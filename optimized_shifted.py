"""
Optimized versions of the shifted matrix computation for different use cases.
Specifically designed for delay arrays of length ~104.
"""

import torch
from typing import Optional, Sequence
import numpy as np

def _compute_shifted_optimized_medium(
    feats_t: torch.Tensor,
    delays: Sequence[int],
    indices_to_keep: Optional[Sequence[int]] = None
) -> torch.Tensor:
    """
    Optimized for medium-sized delay arrays (~100 delays).
    Uses memory-efficient chunking with vectorized operations.
    
    Parameters
    ----------
    feats_t : torch.Tensor
        Input features tensor of shape (n_samples, n_features).
    delays : Sequence[int]
        Delays to apply to the features.
    indices_to_keep : Optional[Sequence[int]]
        Specific indices to compute the shifted matrix for.

    Returns
    -------
    torch.Tensor
        Shifted matrix of shape (n_rows, n_delays, n_features).
    """
    n_samples, n_features = feats_t.shape
    device = feats_t.device
    
    # Convert to tensor once
    if not isinstance(delays, torch.Tensor):
        delays = torch.tensor(delays, device=device, dtype=torch.int64)
    
    if indices_to_keep is not None:
        if not isinstance(indices_to_keep, torch.Tensor):
            idx = torch.tensor(indices_to_keep, device=device, dtype=torch.int64)
        else:
            idx = indices_to_keep.to(device=device, dtype=torch.int64)
        n_rows = idx.shape[0]
    else:
        idx = torch.arange(n_samples, device=device, dtype=torch.int64)
        n_rows = n_samples
    
    n_delays = delays.shape[0]
    
    # Pre-allocate output
    result = torch.zeros((n_rows, n_delays, n_features), 
                        dtype=feats_t.dtype, device=device)
    
    # Process in chunks to balance memory vs speed
    chunk_size = min(32, n_delays)  # Adjust based on your GPU memory
    
    for start_delay in range(0, n_delays, chunk_size):
        end_delay = min(start_delay + chunk_size, n_delays)
        delay_chunk = delays[start_delay:end_delay]
        
        # Vectorized computation for this chunk
        idx_shifted = idx.unsqueeze(1) - delay_chunk.unsqueeze(0)
        valid_mask = (idx_shifted >= 0) & (idx_shifted < n_samples)
        
        # Only process if there are valid indices
        if valid_mask.any():
            idx_clipped = idx_shifted.clamp(0, n_samples - 1)
            chunk_result = feats_t[idx_clipped]
            chunk_result.masked_fill_(~valid_mask.unsqueeze(-1), 0.0)
            result[:, start_delay:end_delay, :] = chunk_result
    
    return result


@torch.jit.script
def _compute_shifted_jit(
    feats_t: torch.Tensor,
    delays: torch.Tensor,
    indices_to_keep: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    JIT-compiled version for maximum speed.
    Call this after the first few runs for best performance.
    
    Parameters
    ----------
    feats_t : torch.Tensor
        Input features tensor of shape (n_samples, n_features).
    delays : torch.Tensor
        Delays to apply to the features (must be tensor for JIT).
    indices_to_keep : Optional[torch.Tensor]
        Specific indices to compute the shifted matrix for.

    Returns
    -------
    torch.Tensor
        Shifted matrix of shape (n_rows, n_delays, n_features).
    """
    n_samples, n_features = feats_t.shape
    device = feats_t.device
    
    if indices_to_keep is not None:
        idx = indices_to_keep
        n_rows = idx.shape[0]
    else:
        idx = torch.arange(n_samples, device=device, dtype=torch.int64)
        n_rows = n_samples
    
    # Vectorized computation
    idx_shifted = idx.unsqueeze(1) - delays.unsqueeze(0)
    valid_mask = (idx_shifted >= 0) & (idx_shifted < n_samples)
    
    # Pre-allocate and fill
    result = torch.zeros((n_rows, delays.shape[0], n_features), 
                        dtype=feats_t.dtype, device=device)
    
    if valid_mask.any():
        idx_clipped = idx_shifted.clamp(0, n_samples - 1)
        gathered = feats_t[idx_clipped]
        gathered = torch.where(valid_mask.unsqueeze(-1), gathered, 
                              torch.zeros_like(gathered))
        result = gathered
    
    return result


def _compute_shifted_fully_vectorized(
    feats_t: torch.Tensor,
    delays: torch.Tensor,
    indices_to_keep: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Fully vectorized version - fastest but uses most memory.
    Best for smaller datasets or when memory is not a constraint.
    
    Parameters
    ----------
    feats_t : torch.Tensor
        Input features tensor of shape (n_samples, n_features).
    delays : torch.Tensor
        Delays to apply to the features.
    indices_to_keep : Optional[torch.Tensor]
        Specific indices to compute the shifted matrix for.

    Returns
    -------
    torch.Tensor
        Shifted matrix of shape (n_rows, n_delays, n_features).
    """
    n_samples, n_features = feats_t.shape
    device = feats_t.device
    
    if indices_to_keep is not None:
        n_rows = indices_to_keep.shape[0]
        # Create broadcasting-friendly indices
        base_indices = indices_to_keep.view(-1, 1)
    else:
        n_rows = n_samples
        base_indices = torch.arange(n_samples, device=device).view(-1, 1)
    
    # Compute all shifted indices at once using broadcasting
    # Shape: (n_rows, n_delays)
    idx_shifted = base_indices - delays.view(1, -1)
    
    # Create valid mask
    valid_mask = (idx_shifted >= 0) & (idx_shifted < n_samples)
    
    # Clamp indices to valid range
    idx_clipped = idx_shifted.clamp(0, n_samples - 1)
    
    # Gather all features at once
    # Shape: (n_rows, n_delays, n_features)
    result = feats_t[idx_clipped]
    
    # Apply mask
    result = torch.where(valid_mask.unsqueeze(-1), result, 
                        torch.zeros_like(result))
    
    return result


def benchmark_shifted_functions(
    n_samples: int = 10000,
    n_features: int = 50,
    n_delays: int = 104,
    n_indices: Optional[int] = None,
    device: str = "cuda"
) -> None:
    """
    Benchmark different shifted matrix implementations.
    
    Parameters
    ----------
    n_samples : int
        Number of time samples
    n_features : int  
        Number of features
    n_delays : int
        Number of delays (default 104 as you mentioned)
    n_indices : Optional[int]
        Number of indices to keep (if None, use all)
    device : str
        Device to run on
    """
    import time
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Create test data
    feats = torch.randn(n_samples, n_features, device=device)
    delays = torch.arange(-n_delays//2, n_delays//2, device=device)
    
    if n_indices is not None:
        indices = torch.randperm(n_samples, device=device)[:n_indices]
    else:
        indices = None
    
    print(f"Benchmarking with:")
    print(f"  Samples: {n_samples}, Features: {n_features}, Delays: {n_delays}")
    print(f"  Indices to keep: {n_indices if n_indices else 'All'}")
    print(f"  Device: {device}")
    print()
    
    # Test functions
    functions = [
        ("Medium-sized optimized", _compute_shifted_optimized_medium),
        ("Fully vectorized", _compute_shifted_fully_vectorized),
    ]
    
    results = {}
    
    for name, func in functions:
        try:
            # Warm up
            for _ in range(3):
                if "jit" in name.lower():
                    _ = func(feats, delays, indices)
                else:
                    _ = func(feats, delays.tolist() if "optimized" in name else delays, indices)
            
            # Benchmark
            torch.cuda.synchronize() if device.type == "cuda" else None
            start_time = time.time()
            
            for _ in range(10):
                if "jit" in name.lower():
                    result = func(feats, delays, indices)
                else:
                    result = func(feats, delays.tolist() if "optimized" in name else delays, indices)
            
            torch.cuda.synchronize() if device.type == "cuda" else None
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            results[name] = avg_time
            
            print(f"{name:25}: {avg_time*1000:.2f} ms")
            print(f"  Output shape: {result.shape}")
            
        except Exception as e:
            print(f"{name:25}: ERROR - {e}")
    
    # Test JIT version separately (needs tensor inputs)
    try:
        # Warm up JIT
        for _ in range(3):
            _ = _compute_shifted_jit(feats, delays, indices)
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        start_time = time.time()
        
        for _ in range(10):
            result = _compute_shifted_jit(feats, delays, indices)
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        results["JIT compiled"] = avg_time
        
        print(f"{'JIT compiled':25}: {avg_time*1000:.2f} ms")
        print(f"  Output shape: {result.shape}")
        
    except Exception as e:
        print(f"{'JIT compiled':25}: ERROR - {e}")
    
    print()
    if results:
        fastest = min(results, key=results.get)
        print(f"Fastest: {fastest} ({results[fastest]*1000:.2f} ms)")


if __name__ == "__main__":
    # Run benchmark with your typical parameters
    print("=== Benchmark for typical use case (104 delays) ===")
    benchmark_shifted_functions(
        n_samples=10000,
        n_features=50, 
        n_delays=104,
        n_indices=5000,  # Using subset of indices
        device="cuda"
    )
