#!/usr/bin/env python3
"""
Smart stop_server script for samuTrain (Standard Library Version)

Stops the samuTrain server gracefully using multiple methods:
1. HTTP API shutdown (if available)
2. Process detection by command line patterns
3. PID file detection
4. Port-based process detection
5. Fallback: kill all Python processes (last resort)

Usage: python utils/stop_server_simple.py [--force] [--verbose]
"""

import os
import sys
import time
import signal
import subprocess
import argparse
import socket
import urllib.request
import urllib.error
import json
from typing import List, Optional

# Constants
SERVER_PORT = 8000
SERVER_HOST = "127.0.0.1"
TIMEOUT_SECONDS = 10
PID_FILE = "samutrain.pid"

def log_message(message: str, verbose: bool = True):
    """Print message if verbose mode is enabled"""
    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] {message}")

def check_server_running() -> bool:
    """Check if server is responding on port 8000"""
    try:
        with urllib.request.urlopen(f"http://{SERVER_HOST}:{SERVER_PORT}/api/statistics", timeout=2) as response:
            return response.status == 200
    except:
        return False

def shutdown_via_api(verbose: bool = True) -> bool:
    """Attempt graceful shutdown via HTTP API"""
    log_message("Attempting graceful shutdown via HTTP API...", verbose)
    
    try:
        # Try to call a shutdown endpoint if it exists
        req = urllib.request.Request(
            f"http://{SERVER_HOST}:{SERVER_PORT}/api/shutdown",
            data=b'',
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                log_message("✅ Server accepted shutdown request", verbose)
                return True
    except urllib.error.URLError:
        pass
    
    # If no dedicated shutdown endpoint, try to trigger graceful shutdown
    try:
        # Send SIGTERM to the server process via API (if implemented)
        data = json.dumps({"signal": "SIGTERM"}).encode('utf-8')
        req = urllib.request.Request(
            f"http://{SERVER_HOST}:{SERVER_PORT}/api/signal",
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                log_message("✅ Server accepted SIGTERM signal", verbose)
                return True
    except:
        pass
    
    log_message("❌ HTTP API shutdown not available", verbose)
    return False

def find_processes_by_name(name_pattern: str) -> List[int]:
    """Find process IDs by name pattern using platform-specific commands"""
    pids = []
    
    if sys.platform == "win32":
        # Windows: use tasklist
        try:
            result = subprocess.run(
                ['tasklist', '/FI', f'IMAGENAME eq {name_pattern}', '/FO', 'CSV'],
                capture_output=True, text=True, timeout=5
            )
            lines = result.stdout.split('\n')
            for line in lines[1:]:  # Skip header
                if line.strip() and '"' in line:
                    parts = line.split('"')
                    if len(parts) >= 2:
                        pid_str = parts[1].strip()
                        if pid_str.isdigit():
                            pids.append(int(pid_str))
        except:
            pass
    else:
        # Unix-like systems: use pgrep or ps
        try:
            # Try pgrep first
            result = subprocess.run(
                ['pgrep', '-f', name_pattern],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for pid_str in result.stdout.strip().split('\n'):
                    if pid_str.strip().isdigit():
                        pids.append(int(pid_str.strip()))
        except:
            # Fallback to ps
            try:
                result = subprocess.run(
                    ['ps', 'aux'],
                    capture_output=True, text=True, timeout=5
                )
                lines = result.stdout.split('\n')
                for line in lines[1:]:  # Skip header
                    if name_pattern.lower() in line.lower():
                        parts = line.split()
                        if len(parts) >= 2 and parts[1].isdigit():
                            pids.append(int(parts[1]))
            except:
                pass
    
    return pids

def find_server_processes() -> List[int]:
    """Find samuTrain server processes by command line patterns"""
    all_pids = set()
    
    # Look for various patterns
    patterns = ['python', 'python.exe', 'python3', 'python3.exe']
    
    for pattern in patterns:
        pids = find_processes_by_name(pattern)
        for pid in pids:
            # Check if this Python process is running samuTrain
            if is_samutrain_process(pid):
                all_pids.add(pid)
    
    return list(all_pids)

def is_samutrain_process(pid: int) -> bool:
    """Check if a process is running samuTrain"""
    try:
        if sys.platform == "win32":
            # Windows: use wmic to get command line
            result = subprocess.run(
                ['wmic', 'process', 'where', f'processid={pid}', 'get', 'commandline'],
                capture_output=True, text=True, timeout=5
            )
            cmdline = result.stdout.lower()
        else:
            # Unix-like systems: use ps
            result = subprocess.run(
                ['ps', '-p', str(pid), '-o', 'args='],
                capture_output=True, text=True, timeout=5
            )
            cmdline = result.stdout.lower()
        
        # Check for samuTrain patterns
        samutrain_patterns = [
            'main.py',
            'samutrain',
            'uvicorn.*8000',
            'fastapi',
            'src.main'
        ]
        
        return any(pattern in cmdline for pattern in samutrain_patterns)
        
    except:
        return False

def find_process_by_port() -> Optional[int]:
    """Find process listening on port 8000"""
    try:
        if sys.platform == "win32":
            # Windows: use netstat
            result = subprocess.run(
                ['netstat', '-ano'],
                capture_output=True, text=True, timeout=5
            )
            lines = result.stdout.split('\n')
            for line in lines:
                if f':{SERVER_PORT}' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid_str = parts[-1].strip()
                        if pid_str.isdigit():
                            return int(pid_str)
        else:
            # Unix-like systems: use lsof
            result = subprocess.run(
                ['lsof', '-i', f':{SERVER_PORT}'],
                capture_output=True, text=True, timeout=5
            )
            lines = result.stdout.split('\n')
            for line in lines[1:]:  # Skip header
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    return int(parts[1])
    except:
        pass
    return None

def read_pid_file() -> Optional[int]:
    """Read PID from file if it exists"""
    try:
        if os.path.exists(PID_FILE):
            with open(PID_FILE, 'r') as f:
                pid = int(f.read().strip())
            return pid
    except:
        pass
    return None

def kill_process(pid: int, signal_type: int = signal.SIGTERM, verbose: bool = True) -> bool:
    """Kill a process by PID"""
    try:
        os.kill(pid, signal_type)
        log_message(f"Sent {signal_type} to process {pid}", verbose)
        
        # Wait for process to terminate
        for i in range(TIMEOUT_SECONDS):
            try:
                # Check if process still exists
                os.kill(pid, 0)  # Signal 0 doesn't kill, just checks existence
                time.sleep(1)
            except OSError:
                # Process is gone
                log_message(f"✅ Process {pid} terminated", verbose)
                return True
        
        # Process still alive, try SIGKILL
        if signal_type != signal.SIGKILL:
            log_message(f"Process {pid} still alive, sending SIGKILL...", verbose)
            return kill_process(pid, signal.SIGKILL, verbose)
        else:
            log_message(f"❌ Failed to kill process {pid}", verbose)
            return False
            
    except OSError as e:
        if e.errno == 3:  # No such process
            log_message(f"Process {pid} not found", verbose)
            return True
        else:
            log_message(f"Error killing process {pid}: {e}", verbose)
            return False

def stop_server_processes(force: bool = False, verbose: bool = True) -> bool:
    """Stop server processes using various detection methods"""
    processes_stopped = False
    
    # Method 1: Find processes by command line patterns
    server_pids = find_server_processes()
    if server_pids:
        log_message(f"Found {len(server_pids)} server processes by command line", verbose)
        for pid in server_pids:
            signal_type = signal.SIGKILL if force else signal.SIGTERM
            if kill_process(pid, signal_type, verbose):
                processes_stopped = True
    
    # Method 2: Find process by port
    if not processes_stopped:
        port_pid = find_process_by_port()
        if port_pid:
            log_message(f"Found process {port_pid} listening on port {SERVER_PORT}", verbose)
            signal_type = signal.SIGKILL if force else signal.SIGTERM
            if kill_process(port_pid, signal_type, verbose):
                processes_stopped = True
    
    # Method 3: Read PID file
    if not processes_stopped:
        pid = read_pid_file()
        if pid:
            log_message(f"Found PID {pid} in {PID_FILE}", verbose)
            signal_type = signal.SIGKILL if force else signal.SIGTERM
            if kill_process(pid, signal_type, verbose):
                processes_stopped = True
                # Clean up PID file
                try:
                    os.remove(PID_FILE)
                except:
                    pass
    
    return processes_stopped

def kill_all_python_processes(verbose: bool = True) -> bool:
    """Last resort: kill all Python processes (dangerous!)"""
    log_message("🚨 LAST RESORT: Killing all Python processes", verbose)
    log_message("This will stop ALL Python scripts, not just samuTrain!", verbose)
    
    try:
        if sys.platform == "win32":
            # Windows: use taskkill
            result = subprocess.run(
                ['taskkill', '/F', '/IM', 'python.exe'],
                capture_output=True, text=True, timeout=10
            )
            result = subprocess.run(
                ['taskkill', '/F', '/IM', 'python3.exe'],
                capture_output=True, text=True, timeout=10
            )
        else:
            # Unix-like systems: use pkill
            subprocess.run(['pkill', '-f', 'python'], timeout=10)
        
        log_message("✅ Sent kill signal to all Python processes", verbose)
        return True
        
    except Exception as e:
        log_message(f"Error killing Python processes: {e}", verbose)
        return False

def wait_for_server_stop(verbose: bool = True) -> bool:
    """Wait for server to actually stop responding"""
    log_message("Waiting for server to stop...", verbose)
    
    for i in range(TIMEOUT_SECONDS):
        if not check_server_running():
            log_message("✅ Server has stopped", verbose)
            return True
        time.sleep(1)
        if verbose and i % 2 == 0:
            log_message(f"Waiting... ({i}/{TIMEOUT_SECONDS}s)", verbose)
    
    log_message("❌ Server still responding after timeout", verbose)
    return False

def main():
    parser = argparse.ArgumentParser(description="Stop samuTrain server")
    parser.add_argument("--force", action="store_true", help="Force kill with SIGKILL")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    parser.add_argument("--quiet", action="store_true", help="Quiet output (overrides verbose)")
    args = parser.parse_args()
    
    verbose = args.verbose and not args.quiet
    
    log_message("🛑 samuTrain Server Stopper", verbose)
    log_message("=" * 40, verbose)
    
    # Check if server is running
    if not check_server_running():
        log_message("ℹ️ Server is not running or not responding", verbose)
        return 0
    
    # Method 1: Try graceful API shutdown
    if shutdown_via_api(verbose):
        if wait_for_server_stop(verbose):
            log_message("🎉 Server stopped gracefully via API", verbose)
            return 0
    
    # Method 2: Stop detected processes
    if stop_server_processes(args.force, verbose):
        if wait_for_server_stop(verbose):
            log_message("🎉 Server stopped via process termination", verbose)
            return 0
    
    # Method 3: Last resort - kill all Python processes
    if args.force or input("Kill all Python processes? (y/N): ").lower().startswith('y'):
        if kill_all_python_processes(verbose):
            if wait_for_server_stop(verbose):
                log_message("🎉 Server stopped via Python process kill", verbose)
                return 0
    
    log_message("❌ Failed to stop server", verbose)
    return 1

if __name__ == "__main__":
    sys.exit(main())
