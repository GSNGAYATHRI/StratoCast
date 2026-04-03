import h5py

file_path = "MT1MADS_1.06_000_9_17_I_2013_12_21_08_06_35_2013_12_21_08_15_18_11318_11318_117_67_67_BL2_00_RR_1.00_L2A.h5"  # Replace with your actual filename

with h5py.File(file_path, "r") as f:
    def explore(name, obj):
        print(name)

    f.visititems(explore)