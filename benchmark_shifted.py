"""
Benchmark script for testing optimized _compute_shifted functions.
Run this on the machine with good GPU to compare performance.
"""

import time
import torch
import numpy as np
from typing import Sequence, Optional
from processing import _compute_shifted, _compute_shifted_optimized_medium, _compute_shifted_jit, _compute_shifted_vectorized

def benchmark_functions():
    """
    Benchmark all versions of _compute_shifted with realistic data sizes.
    """
    
    print("🚀 Starting benchmark for shifted matrix functions...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test parameters similar to your phoneme data
    test_configs = [
        {"n_samples": 5000, "n_features": 15, "n_delays": 104, "name": "Small (5K samples)"},
        {"n_samples": 15000, "n_features": 15, "n_delays": 104, "name": "Medium (15K samples)"},
        {"n_samples": 30000, "n_features": 15, "n_delays": 104, "name": "Large (30K samples)"},
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"Samples: {config['n_samples']}, Features: {config['n_features']}, Delays: {config['n_delays']}")
        print(f"{'='*60}")
        
        # Create test data
        n_samples = config['n_samples']
        n_features = config['n_features']
        n_delays = config['n_delays']
        
        # Generate realistic data
        torch.manual_seed(42)  # For reproducibility
        features = torch.randn(n_samples, n_features, device=device, dtype=torch.float32)
        delays = list(range(n_delays))
        delays_tensor = torch.tensor(delays, device=device, dtype=torch.int64)
        
        # Test with random subset of indices (common in your use case)
        subset_size = min(n_samples // 2, 10000)
        indices_subset = torch.randperm(n_samples, device=device)[:subset_size]
        
        print(f"Memory usage before test: {torch.cuda.memory_allocated()/1e6:.1f} MB" if torch.cuda.is_available() else "CPU mode")
        
        # Functions to test
        functions_to_test = [
            {
                'name': 'Original _compute_shifted',
                'func': _compute_shifted,
                'args': (features, delays, None),
                'warmup': True
            },
            {
                'name': 'Optimized Medium',
                'func': _compute_shifted_optimized_medium,
                'args': (features, delays, None),
                'warmup': True
            },
            {
                'name': 'JIT Compiled',
                'func': _compute_shifted_jit,
                'args': (features, delays_tensor, None),
                'warmup': True
            },
            {
                'name': 'Vectorized',
                'func': _compute_shifted_vectorized,
                'args': (features, delays_tensor, None),
                'warmup': True
            }
        ]
        
        # Test with subset indices too
        functions_subset = [
            {
                'name': 'Original (subset)',
                'func': _compute_shifted,
                'args': (features, delays, indices_subset.cpu().numpy()),
                'warmup': False
            },
            {
                'name': 'Optimized Medium (subset)',
                'func': _compute_shifted_optimized_medium,
                'args': (features, delays, indices_subset.cpu().numpy()),
                'warmup': False
            },
            {
                'name': 'JIT (subset)',
                'func': _compute_shifted_jit,
                'args': (features, delays_tensor, indices_subset),
                'warmup': False
            }
        ]
        
        all_functions = functions_to_test + functions_subset
        results = []
        
        for func_config in all_functions:
            func_name = func_config['name']
            func = func_config['func']
            args = func_config['args']
            
            try:
                # Warmup run (especially important for JIT)
                if func_config['warmup']:
                    print(f"  Warming up {func_name}...")
                    _ = func(*args)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                
                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Actual benchmark
                print(f"  Testing {func_name}...")
                
                start_time = time.time()
                start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
                result = func(*args)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                end_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
                elapsed = end_time - start_time
                memory_used = (end_mem - start_mem) / 1e6  # MB
                
                # Verify output shape
                expected_rows = n_samples if 'subset' not in func_name else subset_size
                expected_shape = (expected_rows, n_delays, n_features)
                
                if result.shape != expected_shape:
                    print(f"    ❌ Shape mismatch! Expected {expected_shape}, got {result.shape}")
                else:
                    print(f"    ✅ Shape correct: {result.shape}")
                
                results.append({
                    'name': func_name,
                    'time': elapsed,
                    'memory': memory_used,
                    'shape': result.shape
                })
                
                print(f"    ⏱️  Time: {elapsed:.4f}s")
                print(f"    🧠 Memory: {memory_used:+.1f} MB")
                
            except Exception as e:
                print(f"    ❌ Error in {func_name}: {e}")
                results.append({
                    'name': func_name,
                    'time': float('inf'),
                    'memory': 0,
                    'error': str(e)
                })
        
        # Summary for this configuration
        print(f"\n📊 Summary for {config['name']}:")
        print("-" * 50)
        
        # Sort by time (excluding errors)
        valid_results = [r for r in results if r['time'] != float('inf')]
        valid_results.sort(key=lambda x: x['time'])
        
        if valid_results:
            fastest = valid_results[0]
            print(f"🏆 Fastest: {fastest['name']} ({fastest['time']:.4f}s)")
            
            for result in valid_results:
                speedup = result['time'] / fastest['time']
                print(f"    {result['name']:25s}: {result['time']:.4f}s ({speedup:.2f}x slower)")
        
        # Memory usage
        print(f"\n💾 Memory Usage:")
        for result in valid_results:
            print(f"    {result['name']:25s}: {result['memory']:+6.1f} MB")

def verify_correctness():
    """
    Verify that all implementations produce the same results.
    """
    print("\n🔍 Verifying correctness...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Small test case
    torch.manual_seed(123)
    features = torch.randn(1000, 5, device=device, dtype=torch.float32)
    delays = list(range(20))
    delays_tensor = torch.tensor(delays, device=device, dtype=torch.int64)
    
    # Get results from all functions
    result_original = _compute_shifted(features, delays, None)
    result_optimized = _compute_shifted_optimized_medium(features, delays, None)
    result_jit = _compute_shifted_jit(features, delays_tensor, None)
    result_vectorized = _compute_shifted_vectorized(features, delays_tensor, None)
    
    # Compare results
    tolerance = 1e-5
    
    print(f"Original shape: {result_original.shape}")
    print(f"Optimized shape: {result_optimized.shape}")
    print(f"JIT shape: {result_jit.shape}")
    print(f"Vectorized shape: {result_vectorized.shape}")
    
    # Check if results are close
    checks = [
        ("Original vs Optimized", torch.allclose(result_original, result_optimized, atol=tolerance)),
        ("Original vs JIT", torch.allclose(result_original, result_jit, atol=tolerance)),
        ("Original vs Vectorized", torch.allclose(result_original, result_vectorized, atol=tolerance)),
    ]
    
    for name, is_close in checks:
        status = "✅ PASS" if is_close else "❌ FAIL"
        print(f"{name}: {status}")
    
    return all(check[1] for check in checks)

if __name__ == "__main__":
    print("🔬 GPU Benchmark for Shifted Matrix Functions")
    print("=" * 60)
    
    # First verify correctness
    if verify_correctness():
        print("✅ All implementations produce identical results!")
        
        # Run performance benchmark
        benchmark_functions()
        
        print(f"\n🎉 Benchmark complete!")
        print(f"GPU Memory after test: {torch.cuda.memory_allocated()/1e6:.1f} MB" if torch.cuda.is_available() else "")
        
    else:
        print("❌ Implementations don't match! Check the code before benchmarking.")
