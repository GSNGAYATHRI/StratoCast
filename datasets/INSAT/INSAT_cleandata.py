import h5py
import numpy as np
import pandas as pd

file_path = "3RIMG_11NOV2025_0015_L2G_GPI_V01R00.h5"

with h5py.File(file_path, "r") as f:
    GPI = f["GPI"][:]           # (H, W)
    lat = f["latitude"][:]      # (H,)
    lon = f["longitude"][:]     # (W,)
    time = f["time"][:]         # typically an integer/time tag

print("Shapes:")
print("GPI:", GPI.shape)
print("lat:", lat.shape)
print("lon:", lon.shape)
print("time:", time)