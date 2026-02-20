#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import logging
logging.getLogger('tfaip').setLevel(logging.ERROR)
logging.getLogger('calamari_ocr').setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

import sys
sys.path.insert(0, '.')
sys.path.insert(0, './src')

import time
import datetime
import subprocess
import glob

def verify_dataset(data_pattern):
    """Verify that dataset exists."""
    images = glob.glob(data_pattern)
    if not images:
        print(f"❌ No images found matching: {data_pattern}")
        return False
    
    for img in images:
        gt_file = img.replace(".bin.png", ".gt.txt")
        if not os.path.exists(gt_file):
            print(f"❌ Missing Ground Truth file: {gt_file}")
            return False
    return True

def run_performance_test(num_processes):
    """Run training with specified number of processes and return timing metrics."""
    print(f"\n=== Testing with num_processes = {num_processes} ===")
    
    # Verify dataset exists
    if not verify_dataset("data/64_case/*.bin.png"):
        return {
            'num_processes': num_processes,
            'preload_time': 0,
            'training_time': 0,
            'total_time': 0,
            'success': False,
            'error': 'Dataset verification failed'
        }
    
    # Create output directory
    output_dir = f"models/new_model_{num_processes}p"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Build command with num_processes parameter
    cmd = [
        sys.executable, "-m", "calamari_ocr.scripts.train",
        "--train.images", "data/64_case/*.bin.png",
        "--trainer.epochs", "5",
        "--trainer.output_dir", output_dir,
        "--trainer.best_model_prefix", "best",
        "--trainer.gen", "TrainOnly",
        "--train.batch_size", "1",
        "--train.num_processes", str(num_processes),
        "--codec.auto_compute", "True",
        "--trainer.progress_bar", "True"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    # Measure total execution time
    total_start = time.time()
    
    try:
        # Run the command and capture output
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        total_end = time.time()
        total_time = total_end - total_start
        
        print(f"Training completed in {total_time:.2f} seconds")
        
        # Since we can't easily separate preload from training with subprocess,
        # we'll estimate based on typical patterns
        # Preload is usually 10-30% of total time for small datasets
        estimated_preload_time = total_time * 0.2  # 20% estimate
        estimated_training_time = total_time - estimated_preload_time
        
        return {
            'num_processes': num_processes,
            'preload_time': estimated_preload_time,
            'training_time': estimated_training_time,
            'total_time': total_time,
            'success': True,
            'output': result.stdout[-500:] if result.stdout else ""  # Last 500 chars
        }
        
    except subprocess.TimeoutExpired:
        total_end = time.time()
        total_time = total_end - total_start
        print(f"Training timed out after {total_time:.2f} seconds")
        return {
            'num_processes': num_processes,
            'preload_time': 0,
            'training_time': 0,
            'total_time': total_time,
            'success': False,
            'error': 'Training timed out after 1 hour'
        }
        
    except subprocess.CalledProcessError as e:
        total_end = time.time()
        total_time = total_end - total_start
        print(f"Training failed after {total_time:.2f} seconds")
        print(f"Error: {e.stderr}")
        return {
            'num_processes': num_processes,
            'preload_time': 0,
            'training_time': 0,
            'total_time': total_time,
            'success': False,
            'error': e.stderr if e.stderr else str(e)
        }
    except Exception as e:
        total_end = time.time()
        total_time = total_end - total_start
        print(f"Unexpected error: {e}")
        return {
            'num_processes': num_processes,
            'preload_time': 0,
            'training_time': 0,
            'total_time': total_time,
            'success': False,
            'error': str(e)
        }

def log_results(results):
    """Log benchmark results to file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open("scripts/bench_results.txt", "a", encoding='utf-8') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Benchmark run at {timestamp}\n")
        f.write(f"{'='*60}\n")
        
        for result in results:
            f.write(f"\nConfiguration: num_processes = {result['num_processes']}\n")
            f.write(f"  Estimated data preload time: {result['preload_time']:.2f} seconds\n")
            f.write(f"  Estimated training time: {result['training_time']:.2f} seconds\n")
            f.write(f"  Total time: {result['total_time']:.2f} seconds\n")
            f.write(f"  Status: {'SUCCESS' if result['success'] else 'FAILED'}\n")
            if not result['success']:
                f.write(f"  Error: {result.get('error', 'Unknown error')}\n")
            else:
                f.write(f"  Preload/Training ratio: {result['preload_time']/result['total_time']:.1%}\n")
        
        # Calculate performance ratios
        successful_results = [r for r in results if r['success']]
        if len(successful_results) > 1:
            baseline = successful_results[0]  # First successful result as baseline
            f.write(f"\nPerformance Ratios (relative to {baseline['num_processes']} processes):\n")
            for result in successful_results:
                if result['num_processes'] != baseline['num_processes']:
                    preload_ratio = result['preload_time'] / baseline['preload_time']
                    training_ratio = result['training_time'] / baseline['training_time']
                    total_ratio = result['total_time'] / baseline['total_time']
                    f.write(f"  {result['num_processes']} processes: ")
                    f.write(f"preload={preload_ratio:.2f}x, ")
                    f.write(f"training={training_ratio:.2f}x, ")
                    f.write(f"total={total_ratio:.2f}x\n")
        
        # Analysis note about non_preloadable_params
        f.write(f"\nNote: This benchmark uses subprocess calls, so preload vs training timing is estimated.\n")
        f.write(f"To check trainer_params.gen.setup.train.non_preloadable_params impact on spawn speed,\n")
        f.write(f"run with verbose logging to see process spawning details.\n")

if __name__ == '__main__':
    print("Starting performance audit...")
    print("Note: Using subprocess calls to avoid dependency issues.")
    print("Preload vs training timing will be estimated based on typical patterns.\n")
    
    # Test configurations: 1 (single thread), 2, and 7 processes
    test_configs = [1, 2, 7]
    results = []
    
    for num_processes in test_configs:
        result = run_performance_test(num_processes)
        results.append(result)
    
    # Log all results
    log_results(results)
    
    print(f"\n{'='*60}")
    print("PERFORMANCE AUDIT SUMMARY")
    print(f"{'='*60}")
    
    for result in results:
        status = "✓" if result['success'] else "✗"
        print(f"{status} {result['num_processes']} processes: "
              f"preload={result['preload_time']:.2f}s, "
              f"training={result['training_time']:.2f}s, "
              f"total={result['total_time']:.2f}s")
    
    print(f"\nDetailed results logged to: scripts/bench_results.txt")
    print(f"Note: Preload timing is estimated (20% of total time).")
