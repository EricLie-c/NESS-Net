import os
import sys

# Add MATLAB directories to search path
# Note: You can remove this section if MATLAB and OpenCV paths are already configured
matlab_dirs = [
    r"D:\MATLAB\bin\win64",
    r"D:\MATLAB\runtime\win64",
    r"D:\MATLAB\extern\bin\win64",
    r"D:\MATLAB\sys\os\win64"
]

for dir_path in matlab_dirs:
    if os.path.exists(dir_path):
        os.environ["PATH"] = dir_path + ";" + os.environ["PATH"]
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(dir_path)

# Import and use NAMLab
import NAMLab

# Your code goes here
# Example usage:
# command = NAMLab.namLab("input_folder output_folder mode iterations")

command = NAMLab.namLab("C:/Users/Admin/Desktop/srp/DFM-NET-NAM-Edge/dataset/NJU2K/RGB C:/Users/Admin/Desktop/srp/DFM-NET-NAM-Edge/dataset/NJU2K/RGB-lab 1 60")

# Example with BSDS500: NAMLab.namLab("../BSDS500/images ../BSDS500/precomputedResults 1 60")