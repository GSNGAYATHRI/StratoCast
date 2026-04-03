import h5py

file_path = "NISAR_L1_PR_RIFG_001_030_A_019_002_2000_SH_20081012T060911_20081012T060925_20081127T061000_20081127T061014_D00404_N_F_J_001.h5"  # Replace with your actual filename

with h5py.File(file_path, "r") as f:
    def explore(name, obj):
        print(name)

    f.visititems(explore)

import h5py

file_path = "NISAR_L1_PR_RIFG_001_030_A_019_002_2000_SH_20081012T060911_20081012T060925_20081127T061000_20081127T061014_D00404_N_F_J_001.h5"

with h5py.File(file_path, "r") as f:
    # Quick inventory of datasets (not groups)
    def show_dset(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(name, obj.shape, obj.dtype)
    f.visititems(show_dset)

import numpy as np

def read_clean(ds):
    """Load an HDF5 dataset, replace _FillValue with NaN, return float32 array."""
    arr = ds[()].astype("float32")
    fv = ds.attrs.get("_FillValue")
    if fv is not None:
        arr = np.where(arr == fv, np.nan, arr)
    return arr

import pandas as pd

file_path = "NISAR_L1_PR_RIFG_001_030_A_019_002_2000_SH_20081012T060911_20081012T060925_20081127T061000_20081127T061014_D00404_N_F_J_001.h5"

with h5py.File(file_path, "r") as f:
    geo = f["science/LSAR/RIFG/metadata/geolocationGrid"]
    sw  = f["science/LSAR/RIFG/swaths/frequencyA"]

    # Geolocation / geometry
    x   = read_clean(geo["coordinateX"])            # e.g. easting
    y   = read_clean(geo["coordinateY"])            # e.g. northing
    inc = read_clean(geo["incidenceAngle"])         # degrees
    hgt = read_clean(geo["heightAboveEllipsoid"])   # m
    bperp = read_clean(geo["perpendicularBaseline"])  # m (if present)

    # Interferogram side
    coh  = read_clean(sw["interferogram/HH/coherenceMagnitude"])
    mask = read_clean(sw["interferogram/mask"])     # 0/1 or similar

    # Optional: wrapped phase, DEM on the radar grid
    dem  = read_clean(sw["interferogram/digitalElevationModel"])

print("x shape:", x.shape)
print("coh shape:", coh.shape)

print("Shapes:")
print("x:", x.shape)
print("y:", y.shape)
print("inc:", inc.shape)
print("hgt:", hgt.shape)
print("bperp:", bperp.shape)
print("coh:", coh.shape)
print("mask:", mask.shape)
print("dem:", dem.shape)

with h5py.File(file_path, "r") as f:
    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(name)
    f.visititems(show)
import h5py

file_path = "NISAR_L1_PR_RIFG_001_030_A_019_002_2000_SH_20081012T060911_20081012T060925_20081127T061000_20081127T061014_D00404_N_F_J_001.h5"

with h5py.File(file_path, "r") as f:
    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(name)
    f.visititems(show)

import h5py
import numpy as np
import pandas as pd

def read_clean(ds):
    arr = ds[()].astype("float32")
    for key in ["_FillValue", "fillValue", "noDataValue"]:
        if key in ds.attrs:
            arr[arr == ds.attrs[key]] = np.nan
    return arr

file_path = "NISAR_L1_PR_RIFG_001_030_A_019_002_2000_SH_20081012T060911_20081012T060925_20081127T061000_20081127T061014_D00404_N_F_J_001.h5"

with h5py.File(file_path, "r") as f:
    sw = f["science/LSAR/RIFG/swaths/frequencyA"]

    wrapped = read_clean(sw["interferogram/HH/wrappedInterferogram"])
    dem     = read_clean(sw["interferogram/digitalElevationModel"])
    mask    = read_clean(sw["interferogram/mask"])

    # Pixel offsets (your ML prediction target!)
    sr_off  = read_clean(sw["pixelOffsets/HH/slantRangeOffset"])
    at_off  = read_clean(sw["pixelOffsets/HH/alongTrackOffset"])
    corrpk  = read_clean(sw["pixelOffsets/HH/correlationSurfacePeak"])

# Check shapes
print(wrapped.shape, dem.shape, mask.shape, sr_off.shape, at_off.shape)

# Flatten
wrapped_f = wrapped.flatten()
dem_f     = dem.flatten()
mask_f    = mask.flatten()
sr_f      = sr_off.flatten()
at_f      = at_off.flatten()
corr_f    = corrpk.flatten()

import h5py
import numpy as np
import pandas as pd

def read_clean(ds):
    arr = ds[()].astype("float32")
    for key in ["_FillValue", "fillValue", "noDataValue"]:
        if key in ds.attrs:
            arr[arr == ds.attrs[key]] = np.nan
    return arr

file_path = "NISAR_L1_PR_RIFG_001_030_A_019_002_2000_SH_20081012T060911_20081012T060925_20081127T061000_20081127T061014_D00404_N_F_J_001.h5"

with h5py.File(file_path, "r") as f:
    sw = f["science/LSAR/RIFG/swaths/frequencyA"]

    sr_off = read_clean(sw["pixelOffsets/HH/slantRangeOffset"])
    at_off = read_clean(sw["pixelOffsets/HH/alongTrackOffset"])
    corrpk = read_clean(sw["pixelOffsets/HH/correlationSurfacePeak"])

# Flatten
sr_f = sr_off.flatten()
at_f = at_off.flatten()
corr_f = corrpk.flatten()

# Drop NaNs
valid = np.isfinite(sr_f) & np.isfinite(at_f) & np.isfinite(corr_f)

df = pd.DataFrame({
    "slant_range_offset": sr_f[valid],
    "along_track_offset": at_f[valid],
    "correlation_peak": corr_f[valid]
})

df.to_csv("nisar_pixel_offsets_clean.csv", index=False)

print("Saved -> nisar_pixel_offsets_clean.csv")
