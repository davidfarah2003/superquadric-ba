import argparse
import numpy as np
import csv
import os

def npy_to_csv(npy_path, csv_path=None):
    # Load the numpy array
    arr = np.load(npy_path)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Array must be 2D and square, got shape {arr.shape}")
    
    # check if array is symmetric
    if not np.allclose(arr, arr.T):
        print("Warning: Array is not symmetric.")
    else:
        print("Array is symmetric.")


    # plot array values >0 distribution and save the plot
    # import matplotlib.pyplot as pltf
    # pltf.hist(arr.flatten()[arr.flatten() > 0], bins=50)
    # pltf.title("Distribution of Array Values")
    # pltf.xlabel("Value")
    # pltf.ylabel("Frequency")
    # pltf.savefig(os.path.splitext(npy_path)[0] + "_distribution.png")
    
    # Save the array to CSV
    # # Determine output path
    # if csv_path is None:
    #     csv_path = os.path.splitext(npy_path)[0] + ".csv"

    # # Write to CSV
    # with open(csv_path, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     for row in arr:
    #         writer.writerow(row)
    # print(f"Saved CSV to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a 2D square .npy file to a .csv file.")
    parser.add_argument("--npy_file", help="Path to the input .npy file")
    parser.add_argument("--csv_file", help="Optional output .csv file path")
    args = parser.parse_args()
    npy_to_csv(args.npy_file, args.csv_file)
