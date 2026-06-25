#!/usr/bin/env python3
import sys
import numpy as np
import avaframe.com4FlowPy.sycl.sycl_core as sycl_core
import avaframe.com4FlowPy.flowCore as flowCore


# DEM with merging paths (diagonal slope)
dem = np.zeros((5, 5), dtype=np.float32)
for r in range(5):
    for c in range(5):
        dem[r, c] = 100.0 - 15.0 * r - 5.0 * c

rel = np.zeros((5, 5), dtype=np.float32)
rel[1, 1] = 1.0

varParams = {
    "varUmaxBool": False,
    "varUmaxArray": None,
    "varAlphaBool": False,
    "varAlphaArray": None,
    "varExponentBool": False,
    "varExponentArray": None,
}

print("\nInputs:")
print("DEM \n", dem)
print("Release Grid \n", rel)


# --- Run SYCL implementation ---
print("\nInvoking C++/SYCL core...")
z_sycl, f_sycl, c_sycl = sycl_core.run_sycl_calculation(
    dem,
    rel,
    None,
    30.0,
    3.0,
    3e-4,
    270.0,
    -9999.0,
    10.0,
    False,
    False,
    varParams,
    False,
    False,
    None,
    None,
    "cpu"
)


# --- Run Python implementation ---
print("\nInvoking Python core...")
outputs = ['zDelta', 'flux', 'cellCounts']
args = [
    dem,
    None,
    rel,
    30.0,
    3.0,
    3e-4,
    270.0,
    -9999.0,
    10.0,
    False,
    False,
    varParams,
    False,
    False,
    None,
    None,
    outputs
]

results_py = flowCore.calculation(args)
z_py = results_py[0]
f_py = results_py[1]
c_py = results_py[2]


# --- Compare Results ---

# Clean up unvisited/no-data values for cleaner comparison
# Python initializes unvisited flux elements to -9999.0, whereas C++ starts them at 0.0.
z_sycl_cleaned = np.where(z_sycl == -9999.0, 0.0, z_sycl)
z_py_cleaned = np.where(z_py == -9999.0, 0.0, z_py)
f_sycl_cleaned = np.where(f_sycl == -9999.0, 0.0, f_sycl)
f_py_cleaned = np.where(f_py == -9999.0, 0.0, f_py)

print("\n--- zDelta ---")
print("SYCL:\n", z_sycl_cleaned)
print("Python:\n", z_py_cleaned)
z_match = np.allclose(z_sycl_cleaned, z_py_cleaned, atol=1e-5)
print(f"Match: {z_match}")

print("\n--- flux ---")
print("SYCL:\n", f_sycl_cleaned)
print("Python:\n", f_py_cleaned)
f_match = np.allclose(f_sycl_cleaned, f_py_cleaned, atol=1e-5)
print(f"Match: {f_match}")

print("\n--- counts ---")
print("SYCL:\n", c_sycl)
print("Python:\n", c_py)
c_match = np.array_equal(c_sycl, c_py)
print(f"Match: {c_match}")

if z_match and f_match and c_match:
    print("\nSUCCESS")
    sys.exit(0)
else:
    print("\nFAILURE")
