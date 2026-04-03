import h5py

file_path = "MT1MADS_1.06_000_9_17_I_2013_12_21_08_06_35_2013_12_21_08_15_18_11318_11318_117_67_67_BL2_00_RR_1.00_L2A.h5"  # Replace with your actual filename

with h5py.File(file_path, "r") as f:
    def explore(name, obj):
        print(name)

    f.visititems(explore)

import h5py
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 1. LOAD MEGHA-TROPIQUES HDF5 FILE
# ------------------------------------------------------------
file_path = "MT1MADS_1.06_000_9_17_I_2013_12_21_08_06_35_2013_12_21_08_15_18_11318_11318_117_67_67_BL2_00_RR_1.00_L2A.h5"   # <- change this to your Megha-Tropiques file

with h5py.File(file_path, "r") as f:
    lat = f["ScienceData/Latitude"][:]              # 2D
    lon = f["ScienceData/Longitude"][:]             # 2D
    rain_rate = f["ScienceData/Rain_Rate"][:]       # 2D
    rain_flag = f["ScienceData/Rain_Flag"][:]       # 2D
    surface_flag = f["ScienceData/Surface_Flag"][:] # 2D
    acq_time = f["ScienceData/Scan_FirstSampleAcq_Time"][:]  # bytes / string

print("Shapes:")
print("lat        :", lat.shape)
print("lon        :", lon.shape)
print("rain_rate  :", rain_rate.shape)
print("rain_flag  :", rain_flag.shape)
print("surface_flag:", surface_flag.shape)
print("acq_time   :", acq_time.shape, acq_time.dtype)

# ------------------------------------------------------------
# 2. CLEAN NUMERIC ARRAYS (lat, lon, rain_rate, flags)
# ------------------------------------------------------------
def clean_numeric(arr):
    arr = arr.astype(float)
    # remove typical fill values like -999, -8888, and huge sentinels
    arr[arr < -100] = np.nan
    arr[arr > 1e10] = np.nan
    return arr

lat = clean_numeric(lat)
lon = clean_numeric(lon)
rain_rate = clean_numeric(rain_rate)
rain_flag = clean_numeric(rain_flag)
surface_flag = clean_numeric(surface_flag)

H, W = lat.shape
N = H * W

# ------------------------------------------------------------
# 3. HANDLE ACQUISITION TIME (BYTE STRING → TEXT)
# ------------------------------------------------------------

# Many products store this as a single bytes string (one timestamp for whole swath)
# We'll decode it and repeat for every pixel.

def decode_bytes_to_str(x):
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("ascii", errors="ignore").strip("\x00")
    return str(x)

if acq_time.size == 1:
    # scalar / length-1 array: one time stamp for entire granule
    ts = acq_time[0] if acq_time.ndim > 0 else acq_time
    ts_str = decode_bytes_to_str(ts)
    acq_flat = np.repeat(ts_str, N)
else:
    # multiple timestamps: decode each and broadcast / flatten
    flat_times = acq_time.flatten()
    flat_times = np.array([decode_bytes_to_str(t) for t in flat_times])
    # if fewer times than pixels, just repeat in a cycle
    if flat_times.size == N:
        acq_flat = flat_times
    else:
        acq_flat = np.resize(flat_times, N)

# ------------------------------------------------------------
# 4. FLATTEN ALL GRIDS INTO ONE TABLE
# ------------------------------------------------------------

df = pd.DataFrame({
    "latitude":      lat.flatten(),
    "longitude":     lon.flatten(),
    "rain_rate":     rain_rate.flatten(),
    "rain_flag":     rain_flag.flatten(),
    "surface_flag":  surface_flag.flatten(),
    "acquisition_time": acq_flat
})

print("Raw dataframe:", df.shape)
print(df.head())

# ------------------------------------------------------------
# 5. DROP PIXELS WITH NO DATA (NaN rain_rate or coords)
# ------------------------------------------------------------

df_clean = df.dropna(subset=["latitude", "longitude", "rain_rate"])
print("After cleaning:", df_clean.shape)

# ------------------------------------------------------------
# 6. SAVE TO CSV
# ------------------------------------------------------------

output_csv = "meghatropiques_clean.csv"
df_clean.to_csv(output_csv, index=False)

print("Saved cleaned dataset →", output_csv)