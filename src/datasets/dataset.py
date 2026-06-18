from pathlib import Path

import numpy as np
from pydicom import dcmread
from pydicom.errors import InvalidDicomError


class Dataset:
    def __init__(self, strict: bool = True):
        self.strict = strict

    def analyze(self, ct_directory: str | Path, segmentation_file: str | Path,) -> dict:
        """
        Load and normalize a CT series, load its DICOM SEG file,
        and match each segmentation frame to the nearest CT slice.

        Parameters
        ----------
        ct_directory:
            Directory containing the CT .dcm slices.

        segmentation_file:
            Path to the DICOM SEG .dcm file.
        """

        ct_directory = Path(ct_directory).expanduser()
        segmentation_file = Path(segmentation_file).expanduser()

        self._validate_paths(ct_directory, segmentation_file)

        ct_files = self._find_ct_files(ct_directory)
        ct_images = self._load_ct_images(ct_files)

        segmentation, segmentation_positions = self._load_segmentation(
            segmentation_file
        )

        matching_indices = self._match_segmentation_to_ct(
            segmentation_positions,
            ct_files,
        )

        return {
            "ct_images": ct_images,
            "segmentation": segmentation,
            "matching_indices": matching_indices,
            "ct_files": [str(path) for _, path in ct_files],
            "ct_positions": [z_position for z_position, _ in ct_files],
            "segmentation_positions": segmentation_positions,
        }

    def _validate_paths(self, ct_directory: Path, segmentation_file: Path,) -> None:
        # These error raises will stop the program
        if not ct_directory.exists():
            raise FileNotFoundError(
                f"CT directory does not exist: {ct_directory}"
            )

        if not ct_directory.is_dir():
            raise NotADirectoryError(
                f"CT path must be a directory: {ct_directory}"
            )

        if not segmentation_file.exists():
            raise FileNotFoundError(
                f"Segmentation file does not exist: {segmentation_file}"
            )

        if not segmentation_file.is_file():
            raise ValueError(
                f"Segmentation path must be a file: {segmentation_file}"
            )

    def _find_ct_files(self, ct_directory: Path) -> list[tuple[float, Path]]:
        """
        Find CT DICOM files and sort them by their Z position.
        """

        ct_files = []

        for file_path in ct_directory.rglob("*.dcm"):
            try:
                dataset = dcmread(file_path, stop_before_pixels=True)
            except InvalidDicomError:
                if self.strict:
                    raise ValueError(f"Invalid DICOM file: {file_path}")
                continue

            # Ignore non-CT DICOM files.
            if getattr(dataset, "Modality", None) != "CT":
                continue

            image_position = getattr(dataset, "ImagePositionPatient", None,)

            if image_position is None:
                if self.strict:
                    raise ValueError(f"CT file has no ImagePositionPatient: {file_path}")
                continue

            z_position = float(image_position[2])
            ct_files.append((z_position, file_path))

        if not ct_files:
            raise ValueError(
                f"No CT DICOM files were found in: {ct_directory}"
            )

        ct_files.sort(key=lambda item: item[0])

        return ct_files

    def _load_ct_images(self, ct_files: list[tuple[float, Path]],) -> np.ndarray:
        """
        Load and normalize every CT slice between 0 and 1.
        """

        normalized_images = []

        for _, file_path in ct_files:
            dataset = dcmread(file_path)

            image = dataset.pixel_array.astype(np.float32)

            minimum = np.min(image)
            maximum = np.max(image)

            if maximum == minimum:
                normalized_image = np.zeros_like(image, dtype=np.float32)
            else:
                normalized_image = ((image - minimum) / (maximum - minimum))

            normalized_images.append(normalized_image)

        return np.stack(normalized_images)

    def _load_segmentation(self, segmentation_file: Path,) -> tuple[np.ndarray, list[float]]:
        """
        Load the DICOM SEG pixel data and each frame's Z position.
        """

        dataset = dcmread(segmentation_file)

        segmentation = dataset.pixel_array

        if not hasattr(dataset, "PerFrameFunctionalGroupsSequence"):
            raise ValueError(
                "The segmentation DICOM does not contain "
                "PerFrameFunctionalGroupsSequence."
            )

        segmentation_positions = []

        for frame in dataset.PerFrameFunctionalGroupsSequence:
            try:
                position = (
                    frame
                    .PlanePositionSequence[0]
                    .ImagePositionPatient[2]
                )
            except (AttributeError, IndexError):
                raise ValueError(
                    "A segmentation frame does not contain a valid "
                    "ImagePositionPatient value."
                )

            segmentation_positions.append(float(position))

        return segmentation, segmentation_positions

    def _match_segmentation_to_ct(self, segmentation_positions: list[float], ct_files: list[tuple[float, Path]],) -> list[int]:
        """
        Match each segmentation frame to the closest CT slice.
        """

        ct_positions = np.array(
            [z_position for z_position, _ in ct_files],
            dtype=np.float32,
        )

        matching_indices = []

        for segmentation_z in segmentation_positions:
            distances = np.abs(ct_positions - segmentation_z)
            closest_index = int(np.argmin(distances))
            matching_indices.append(closest_index)

        return matching_indices
