import numpy as np
 
def dependence_vector(src_iter, dst_iter):
    return tuple(d-s for s,d in zip(src_iter,dst_iter))
 
def is_loop_parallelisable(dep_vectors, loop_dim):
    """Check if outer loop can be parallelised (no dep on outer dim)."""
    for dv in dep_vectors:
        if dv[loop_dim]!=0: return False
    return True
 
def pluto_schedule(dep_matrix, n_loops):
    """Simplified Pluto-style loop transformation (affine scheduling)."""
    schedules=[]
    for i in range(n_loops):
        basis=np.eye(n_loops,dtype=int)
        legal=all(basis[i]@np.array(d)>=0 for d in dep_matrix)
        schedules.append((i, 'parallel' if legal else 'sequential'))
    return schedules
 
# Example: matrix multiply C[i][j] += A[i][k] * B[k][j]
# Dependencies: (0,0,1) for k-loop (read-after-write on same k)
deps = [(0,0,1)]  # distance vector (di, dj, dk)
for loop, axis in enumerate(['i-loop','j-loop','k-loop']):
    par = is_loop_parallelisable(deps, loop)
    print(f"  {axis}: {'✓ PARALLEL' if par else '✗ SEQUENTIAL'}")
 
# Tiling transformation
def tile_loop(N, T):
    print(f"\nTiled loop (N={N}, T={T}):")
    print("  for ii in range(0,N,T):    # tile outer (parallel)")
    print("    for i in range(ii,min(ii+T,N)):  # tile inner")
 
tile_loop(1024, 32)
print("\nPolyhedral schedule for matmul:")
sch=pluto_schedule([(0,0,1)],3)
for loop,mode in sch: print(f"  Loop {loop}: {mode}")
