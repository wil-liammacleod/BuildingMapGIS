import laspy
import numpy as np

def main():
    path = "data/Ontario/McMaster/raw/ON_Niagara_20210525_NAD83CSRS_UTM17N_1km_E587_N4790_CLASS_standard.laz"
    print(f"Reading {path}...")
    
    with laspy.open(path) as fh:
        print(f"Header point count: {fh.header.point_count}")
        las = fh.read()
        
    classes = las.classification
    unique_classes, counts = np.unique(classes, return_counts=True)
    
    print("\nPoint classification distribution:")
    for cls, count in zip(unique_classes, counts):
        percentage = (count / len(classes)) * 100
        print(f"Class {cls}: {count} points ({percentage:.2f}%)")

if __name__ == "__main__":
    main()
