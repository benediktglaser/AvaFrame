#!/usr/bin/env python3
import time
import numpy as np
import avaframe.com4FlowPy.sycl.sycl_core as sycl_core
import avaframe.com4FlowPy.flowCore as flowCore

GRID_SIZE = 2500
NUM_RELEASE_CELLS = 10

print(f"Creating a synthetic DEM of size {GRID_SIZE}x{GRID_SIZE} with {NUM_RELEASE_CELLS} release cells...")

# Create a downslope diagonal DEM
dem = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        dem[r, c] = 2000.0 - 5.0 * r - 3.0 * c

rel = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
for i in range(10, 10 + NUM_RELEASE_CELLS):
    rel[i, i] = 1.0

varParams = {
    "varUmaxBool": False,
    "varUmaxArray": None,
    "varAlphaBool": False,
    "varAlphaArray": None,
    "varExponentBool": False,
    "varExponentArray": None,
}

# Warm-up
print("Warming up SYCL core (JIT compilation)...")
_, _, _ = sycl_core.run_sycl_calculation(
    dem, rel, None, 30.0, 3.0, 3e-4, 270.0, -9999.0, 10.0,
    False, False, varParams, False, False, None, None, "cpu"
)

# Sycl
start_sycl = time.perf_counter()
z_sycl, f_sycl, c_sycl = sycl_core.run_sycl_calculation(
    dem, rel, None, 30.0, 3.0, 3e-4, 270.0, -9999.0, 10.0,
    False, False, varParams, False, False, None, None, "cpu"
)
end_sycl = time.perf_counter()
sycl_duration = end_sycl - start_sycl
print(f"SYCL execution time: {sycl_duration:.6f} seconds")

# Python
outputs = ['zDelta', 'flux', 'cellCounts']
args = [
    dem, None, rel, 30.0, 3.0, 3e-4, 270.0, -9999.0, 10.0,
    False, False, varParams, False, False, None, None, outputs
]

start_py = time.perf_counter()
results_py = flowCore.calculation(args)
end_py = time.perf_counter()
py_duration = end_py - start_py
print(f"Python execution time: {py_duration:.6f} seconds")

print(f"Grid size:              {GRID_SIZE} x {GRID_SIZE} ({GRID_SIZE*GRID_SIZE} cells)")
print(f"Release cells:          {NUM_RELEASE_CELLS}")
print(f"Python execution:       {py_duration:.4f} s")
print(f"SYCL execution:         {sycl_duration:.4f} s")
speedup = py_duration / sycl_duration
print(f"Speedup factor:         {speedup:.2f}x faster with SYCL")
