import pathlib, glob
import numpy as np
import avaframe.in2Trans.rasterUtils as IOf

# Locate output peak files directory
peak_dir = pathlib.Path('data/avaParabola/Outputs/com4FlowPy/peakFiles')
if not peak_dir.exists() or len(list(peak_dir.glob('res_*'))) == 0:
    peak_dir = pathlib.Path('avaframe/data/avaParabola/Outputs/com4FlowPy/peakFiles')

run_dirs = sorted(list(peak_dir.glob('res_*')))

print("=" * 70)
print("VERIFYING SIMULATION OUTPUT RASTERS ACROSS RUNS")
print("=" * 70)
print(f"Found {len(run_dirs)} output simulation run folder(s):")
for r in run_dirs:
    print(f" - {r}")

if len(run_dirs) < 2:
    print("\nNeed at least 2 simulation run folders to compare!")
    exit()

def get_raster_data(folder_path, keyword):
    matches = list(folder_path.glob(f'*{keyword}*'))
    if not matches:
        raise FileNotFoundError(f"No file matching '*{keyword}*' found in {folder_path}")
    return IOf.readRaster(matches[0])['rasterData']

# Load arrays from Run 1
run1_path = run_dirs[0]
run1_z = get_raster_data(run1_path, 'zdelta')
run1_c = get_raster_data(run1_path, 'cellCounts')

# Compare against all subsequent runs
for i in range(1, len(run_dirs)):
    run2_path = run_dirs[i]
    run2_z = get_raster_data(run2_path, 'zdelta')
    run2_c = get_raster_data(run2_path, 'cellCounts')

    print("\n" + "-" * 70)
    print(f"Comparing: [{run1_path.name}] vs [{run2_path.name}]")
    print("-" * 70)

    # 1. Cell Visit Counts Check (equal_nan=True handles NaN == NaN correctly)
    count_match = np.array_equal(run1_c, run2_c, equal_nan=True)
    c1_clean = np.nan_to_num(run1_c, nan=-9999.0)
    c2_clean = np.nan_to_num(run2_c, nan=-9999.0)
    count_mismatches = np.sum(c1_clean != c2_clean)
    print(f"1. Cell Visit Counts (cellCounts):")
    print(f"   - Exact Match:                 {count_match}")
    print(f"   - Mismatched Pixels:           {count_mismatches} / {run1_c.size}")

    # 2. zDelta Check (Kinetic Energy / Velocity Height)
    z1_clean = np.nan_to_num(run1_z, nan=0.0)
    z2_clean = np.nan_to_num(run2_z, nan=0.0)
    z_diff = np.abs(z1_clean - z2_clean)
    print(f"2. Kinetic Energy Height (zdelta):")
    print(f"   - Max Absolute Difference:     {np.max(z_diff):.8e}")
    print(f"   - Mismatched Pixels (> 1e-5):  {np.sum(z_diff > 1e-5)} / {run1_z.size}")

print("=" * 70)