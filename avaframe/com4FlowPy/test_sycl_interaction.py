#!/usr/bin/env python3
import os
import sys
import numpy as np
import avaframe.com4FlowPy.sycl.sycl_core as sycl_core
import avaframe.com4FlowPy.flowCore as flowCore

device_type = os.environ.get("ACPP_TARGETS", sys.argv[1] if len(sys.argv) > 1 else "cpu").lower()
device_type = "gpu" if ("gpu" in device_type or "cuda" in device_type) else "cpu"
print(f"Running unit tests on device target: '{device_type}'")

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

def compare_results(name, sycl_res, py_res, tol=1e-5):
    sycl_cleaned = np.where(sycl_res == -9999.0, 0.0, sycl_res)
    py_cleaned = np.where(py_res == -9999.0, 0.0, py_res)
    match = np.allclose(sycl_cleaned, py_cleaned, atol=tol)
    print(f"{name} Match: {match}")
    if not match:
        print("SYCL:")
        print(sycl_cleaned)
        print("Python:")
        print(py_cleaned)
    return match

# ==========================================
# TEST 1: Base Correctness (No Forest/Infra)
# ==========================================
print("\n--- RUNNING TEST 1: Base Correctness ---")
z_sycl, f_sycl, c_sycl, b_sycl, fo_sycl = sycl_core.run_sycl_calculation(
    dem, rel, None, 30.0, 3.0, 3e-4, 270.0, -9999.0, 10.0,
    False, False, varParams, False, False, None, None, device_type
)

outputs = ['zDelta', 'flux', 'cellCounts']
args = [
    dem, None, rel, 30.0, 3.0, 3e-4, 270.0, -9999.0, 10.0,
    False, False, varParams, False, False, None, None, outputs
]
res_py = flowCore.calculation(args)
z_py, f_py, c_py = res_py[0], res_py[1], res_py[2]

t1_z = compare_results("zDelta", z_sycl, z_py)
t1_f = compare_results("flux", f_sycl, f_py)
t1_c = np.array_equal(c_sycl, c_py)
print(f"Counts Match: {t1_c}")

test1_success = t1_z and t1_f and t1_c

# ==========================================
# TEST 2: Infrastructure Correctness
# ==========================================
print("\n--- RUNNING TEST 2: Infrastructure Backtracking ---")
infra = np.zeros((5, 5), dtype=np.float32)
infra[3, 3] = 4.0
infra[4, 2] = 2.0

z_sycl, f_sycl, c_sycl, b_sycl, fo_sycl = sycl_core.run_sycl_calculation(
    dem, rel, infra, 30.0, 3.0, 3e-4, 270.0, -9999.0, 10.0,
    True, False, varParams, False, False, None, None, device_type
)

args[1] = infra
args[9] = True
res_py = flowCore.calculation(args)
z_py, f_py, c_py, _, b_py = res_py[0], res_py[1], res_py[2], res_py[3], res_py[4]

t2_z = compare_results("zDelta", z_sycl, z_py)
t2_f = compare_results("flux", f_sycl, f_py)
t2_c = np.array_equal(c_sycl, c_py)
print(f"Counts Match: {t2_c}")
t2_b = compare_results("backcalc", b_sycl, b_py)

test2_success = t2_z and t2_f and t2_c and t2_b

# ==========================================
# TEST 3: Forest Interaction Correctness
# ==========================================
print("\n--- RUNNING TEST 3: Forest Interaction ---")
forestArray = np.zeros((5, 5), dtype=np.float32)
forestArray[2, 2] = 1.0
forestArray[3, 2] = 1.0

forestParams = {
    "forestInteraction": True,
    "forestModule": "forestFriction",
    "maxAddedFriction": 10.0,
    "minAddedFriction": 2.0,
    "velThForFriction": 5.0,
    "maxDetrainment": 0.5,
    "minDetrainment": 0.1,
    "velThForDetrain": 3.0,
    "fFrLayerType": "absolute",
    "skipForestDist": 1.0,
}

z_sycl, f_sycl, c_sycl, b_sycl, fo_sycl = sycl_core.run_sycl_calculation(
    dem, rel, None, 30.0, 3.0, 3e-4, 270.0, -9999.0, 10.0,
    False, True, varParams, False, False, forestArray, forestParams, device_type
)

args[1] = None
args[9] = False
args[10] = True
args[14] = forestArray
args[15] = forestParams

res_py = flowCore.calculation(args)
z_py, f_py, c_py = res_py[0], res_py[1], res_py[2]
# In forestInteraction mode, calculation returns 13 elements. forestIntArray is at index 12.
fo_py = res_py[12]

t3_z = compare_results("zDelta", z_sycl, z_py)
t3_f = compare_results("flux", f_sycl, f_py)
t3_c = np.array_equal(c_sycl, c_py)
print(f"Counts Match: {t3_c}")
t3_fo = compare_results("forestInt", fo_sycl, fo_py)

test3_success = t3_z and t3_f and t3_c and t3_fo

# ==========================================
# Final Verdict
# ==========================================
print("\n==========================================")
if test1_success and test2_success and test3_success:
    print("ALL TESTS PASSED SUCCESSFULLY!")
    sys.exit(0)
else:
    print("SOME TESTS FAILED!")
    sys.exit(1)
