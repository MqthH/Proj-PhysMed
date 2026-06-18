import argparse
from src.datasets.dataset import Dataset


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CT image segmentation prototype.\nPrototype de segmentation d'images CT."
    )

    parser.add_argument(
        "ct_directory",
        help="Path to the directory containing the CT .dcm slices.\nChemin vers le dossier contenant les coupes CT au format .dcm.",
    )

    parser.add_argument(
        "--seg-file",
        required=True,
        help="Path to the DICOM SEG .dcm file.\nChemin vers le fichier de segmentation DICOM SEG au format .dcm.",
    )

    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Skip some invalid DICOM files instead of stopping.\nIgnorer certains fichiers DICOM invalides au lieu d'arrêter le programme.",
    )

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    dataset_loader = Dataset(strict=not args.non_strict)

    try:
        result = dataset_loader.analyze(
            ct_directory=args.ct_directory,
            segmentation_file=args.seg_file,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        parser.error(str(error))

    print("Dataset loaded successfully.")
    print(f"CT slices: {len(result['ct_images'])}")
    print(f"CT array shape: {result['ct_images'].shape}")
    print(f"SEG array shape: {result['segmentation'].shape}")
    print(
        "Matched segmentation frames: "
        f"{len(result['matching_indices'])}"
    )


if __name__ == "__main__":
    main()