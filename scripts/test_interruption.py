# What is this file for: Test script for graceful shutdown functionality
# When it is created: 2026-03-25

import time
import threading
import signal
import sys
import os

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from scripts.interruption_manager import get_interruption_manager, TrainingContext, reset_interruption_manager


def long_running_task():
  """Simulate a long-running training task"""
  print("Starting long-running task...")
  
  for i in range(10):
    if get_interruption_manager().is_shutdown_requested():
      print(f"Task interrupted at iteration {i}")
      return False
    
    print(f"Working... iteration {i}")
    time.sleep(1)
  
  print("Task completed successfully")
  return True


def test_graceful_shutdown():
  """Test graceful shutdown functionality"""
  print("=== Testing Graceful Shutdown ===")
  
  # Reset interruption manager
  reset_interruption_manager()
  manager = get_interruption_manager()
  
  # Add shutdown callback
  def shutdown_callback(reason):
    print(f"Shutdown callback: {reason}")
  
  manager.add_shutdown_callback(shutdown_callback)
  
  # Test 1: Normal completion
  print("\n--- Test 1: Normal completion ---")
  with TrainingContext():
    result = long_running_task()
  print(f"Result: {result}")
  
  # Test 2: Interrupt after 3 seconds
  print("\n--- Test 2: Interrupt after 3 seconds ---")
  reset_interruption_manager()
  manager = get_interruption_manager()
  manager.add_shutdown_callback(shutdown_callback)
  
  def interrupt_after_delay():
    time.sleep(3)
    print("Sending interrupt signal...")
    manager.request_shutdown("Test interrupt")
  
  # Start interruption timer
  interrupt_thread = threading.Thread(target=interrupt_after_delay)
  interrupt_thread.daemon = True
  interrupt_thread.start()
  
  with TrainingContext():
    result = long_running_task()
  print(f"Result: {result}")
  
  # Test 3: Signal handling
  print("\n--- Test 3: Signal handling ---")
  reset_interruption_manager()
  manager = get_interruption_manager()
  manager.add_shutdown_callback(shutdown_callback)
  
  def signal_after_delay():
    time.sleep(2)
    print("Sending SIGINT...")
    # Simulate signal
    manager._signal_handler(signal.SIGINT)
  
  signal_thread = threading.Thread(target=signal_after_delay)
  signal_thread.daemon = True
  signal_thread.start()
  
  with TrainingContext():
    result = long_running_task()
  print(f"Result: {result}")
  
  print("\n=== All tests completed ===")


if __name__ == "__main__":
  test_graceful_shutdown()
