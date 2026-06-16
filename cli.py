from Dataset.dataset import Dataset
import argparse

def main():

    parser = argparse.ArgumentParser(
        description = """
 Prototype Segmentation CT
 """, formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument(
        "dicom_file",
        help = "Path to the .dcm file"
    )

    
    args = parser.parse_args()

    full_dataset = Dataset()
    
    dataset_normalized = full_dataset.dataset_normalize(args.dicom_file)