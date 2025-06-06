"""
Simple benchmark with 100 repetitions for better statistics
"""

import time
import torch
from tqdm import tqdm
from utils.processing import _compute_shifted, _compute_shifted_optimized_medium, _compute_shifted_vectorized
from load import load_data
import config
import numpy as np

# Load data once
preprocessed_data_path = f'saves/preprocessed_data/External/tmin{config.tmin}_tmax{config.tmax}/'
subject_1, subject_2, samples_info = load_data(
                                session=21,
                                stim='Phones-Discrete-Phonet',
                                band='Theta',
                                sr=config.sr,
                                delays=config.delays,
                                preprocessed_data_path=preprocessed_data_path,
                                praat_executable_path=config.praat_executable_path,
                                situation='External'
                                )
eeg_subject_1, eeg_subject_2, info = subject_1['EEG'], subject_2['EEG'], subject_1['info']
stims_subject_1 = np.hstack([subject_1['Phones-Discrete-Phonet']])
# relevant_indexes_1 = samples_info['keep_indexes1'].copy()
relevant_indexes_1 = np.arange(stims_subject_1.shape[0])  # Use all indices for simplicity

def simple_benchmark():
    print("🚀 100x Benchmark for Shifted Matrix Functions")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Setup data
    features = torch.tensor(stims_subject_1, device=device, dtype=torch.float32)
    n_samples, n_features = features.shape 
    delays = config.delays
    delays_tensor = torch.tensor(delays, device=device)
    relevant_indices_tensor = torch.tensor(relevant_indexes_1, device=device, dtype=torch.int64)
    
    print(f"Data: {n_samples} samples, {n_features} features, {len(delays)} delays")
    print(f"Using {len(relevant_indexes_1)} subset indices")
    print("-" * 50)
    
    # Functions to test
    functions = [
        ("Original", lambda: _compute_shifted(features, delays, relevant_indices_tensor)),
        ("Optimized", lambda: _compute_shifted_optimized_medium(features, delays, relevant_indices_tensor)),
        ("Vectorized", lambda: _compute_shifted_vectorized(features, delays_tensor, relevant_indices_tensor))
    ]
    
    # Collect results
    results = {name: [] for name, _ in functions}
    
    for i in tqdm(range(100), desc="Running benchmarks"):
        for name, func in functions:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                start = time.time()
                result = func()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.time() - start
                
                results[name].append(elapsed)
                del result
                
            except Exception as e:
                print(f"\n❌ {name}: {str(e)[:50]}...")
                results[name].append(float('inf'))
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Show statistics
    print("\n📊 FINAL STATISTICS (100 runs)")
    print("=" * 50)
    print(f"{'Function':<12} {'Min':<8} {'Max':<8} {'Avg':<8} {'Std':<8}")
    print("-" * 50)
    
    for name in results:
        times = [t for t in results[name] if t != float('inf')]
        if times:
            min_t, max_t = min(times), max(times)
            avg_t = sum(times) / len(times)
            std_t = (sum((t - avg_t)**2 for t in times) / len(times))**0.5
            print(f"{name:<12} {min_t:<8.3f} {max_t:<8.3f} {avg_t:<8.3f} {std_t:<8.3f}")
        else:
            print(f"{name:<12} {'ERROR':<8}")
    
    # Best average
    valid_results = {name: times for name, times in results.items() 
                    if times and all(t != float('inf') for t in times)}
    
    if valid_results:
        best_name = min(valid_results.keys(), 
                       key=lambda name: sum(valid_results[name]) / len(valid_results[name]))
        best_avg = sum(valid_results[best_name]) / len(valid_results[best_name])
        print(f"\n🏆 Best average: {best_name} ({best_avg:.3f}s)")

def verify_correctness():
    print("🔍 Quick Correctness Check")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    torch.manual_seed(123)
    features = torch.randn(1000, 5, device=device, dtype=torch.float32)
    delays = list(range(10))
    delays_tensor = torch.tensor(delays, device=device)
    
    # Random selection of relevant indices
    relevant_indices = torch.randint(0, features.shape[0], (100,), device=device, dtype=torch.int64)
    
    try:
        r1 = _compute_shifted(features, delays, relevant_indices)
        r2 = _compute_shifted_optimized_medium(features, delays, relevant_indices)
        r4 = _compute_shifted_vectorized(features, delays_tensor, relevant_indices)
        
        checks = [torch.allclose(r1, r2, atol=1e-5), torch.allclose(r2, r4, atol=1e-5), 
                 torch.allclose(r1, r4, atol=1e-5)]
        
        if all(checks):
            print("✅ All functions produce identical results!\n")
            return True
        else:
            print("❌ Results don't match!")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    if verify_correctness():
        simple_benchmark()
