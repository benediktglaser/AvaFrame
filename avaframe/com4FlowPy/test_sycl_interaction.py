#!/usr/bin/env python3
import numpy as np
import avaframe.com4FlowPy.sycl.sycl_core as sycl_core


dem = np.zeros((5, 5), dtype=np.float32)
for r in range(5):
    for c in range(5):
        dem[r, c] = 100.0 - 15.0 * r - 5.0 * c

rel = np.zeros((5, 5), dtype=np.float32)
rel[1, 1] = 1.0

print("\nInputs:")
print("DEM \n", dem)
print("Release Grid \n", rel)


print("\nInvoking C++/SYCL core with realistic parameters...")
z, f, c = sycl_core.run_sycl_calculation(
    dem,
    rel,
    0.0,
    0,
    0.0,
    0.0,
    0.0,
    0.0,
    "cpu"
)


print("zDelta\n", z)
print("flux\n", f)
print("counts\n", c)
