
==========================================================================
Main Preprocessing Pipeline- Split3D-GS
==========================================================================
# Workflow

1. Load diffusion MRI volume
2. Intensity Normalization
3. Outlier Detection
4. Noise Reduction
5. Adaptive Gaussian Initialization
6. Covariance Preparation
7. Save Preprocessed Volume

==========================================================================


import os
import argparse
import nibabel as nib
import numpy as np

from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter

from sklearn.preprocessing import MinMaxScaler


# ==========================================================
# Utility Functions
# ==========================================================

def create_directory(folder):

    if not os.path.exists(folder):
        os.makedirs(folder)


# ==========================================================
# Load NIfTI Volume
# ==========================================================

class DiffusionMRIReader:

    def __init__(self, filename):

        self.filename = filename

    def load(self):

        image = nib.load(self.filename)

        volume = image.get_fdata()

        affine = image.affine

        header = image.header

        print("--------------------------------------------------")
        print("Input Volume")
        print("--------------------------------------------------")
        print("Shape :", volume.shape)
        print("Datatype :", volume.dtype)
        print()

        return volume, affine, header


# ==========================================================
# Intensity Normalization
#
# Paper:
# Min-Max normalization (Equation 11)
# ==========================================================

class IntensityNormalization:

    def __call__(self, volume):

        scaler = MinMaxScaler()

        shape = volume.shape

        volume = scaler.fit_transform(
            volume.reshape(-1,1)
        )

        volume = volume.reshape(shape)

        return volume


# ==========================================================
# Outlier Removal
#
# Paper:
# z-score threshold > 3
# ==========================================================

class OutlierRemoval:

    def __init__(self,
                 threshold=3.0):

        self.threshold = threshold

    def __call__(self, volume):

        mean = np.mean(volume)

        std = np.std(volume)

        z = (volume-mean)/(std+1e-8)

        outliers = np.abs(z) > self.threshold

        print("Outliers Removed :", outliers.sum())

        volume[outliers] = mean

        return volume


# ==========================================================
# Noise Reduction
#
# Preserves anatomical continuity
# ==========================================================

class NoiseReduction:

    def __call__(self, volume):

        volume = median_filter(volume,
                               size=3)

        volume = gaussian_filter(volume,
                                 sigma=1.0)

        return volume


# ==========================================================
# Adaptive Gaussian Initialization
# ==========================================================

class GaussianInitialization:

    def __init__(self,
                 sigma=1.2):

        self.sigma = sigma

    def __call__(self, volume):

        gaussian = gaussian_filter(
            volume,
            sigma=self.sigma
        )

        return gaussian


# ==========================================================
# Covariance Preparation
# ==========================================================

class CovariancePreparation:

    def __call__(self, volume):

        xyz = np.argwhere(volume > 0)

        covariance = np.cov(xyz.T)

        print()

        print("Covariance Matrix")

        print(covariance)

        print()

        return covariance


# ==========================================================
# Save Volume
# ==========================================================

class VolumeWriter:

    def save(self,
             volume,
             affine,
             header,
             output):

        image = nib.Nifti1Image(
            volume,
            affine,
            header
        )

        nib.save(image, output)

        print("------------------------------------------")
        print("Saved :", output)
        print("------------------------------------------")


# ==========================================================
# Main Preprocessing Pipeline
# ==========================================================

class PreprocessingPipeline:

    def __init__(self):

        self.normalize = IntensityNormalization()

        self.outlier = OutlierRemoval()

        self.noise = NoiseReduction()

        self.gaussian = GaussianInitialization()

        self.covariance = CovariancePreparation()

        self.writer = VolumeWriter()

    def process(self,
                input_file,
                output_directory):

        create_directory(output_directory)

        volume, affine, header = \
            DiffusionMRIReader(
                input_file
            ).load()

        print("Step 1 : Intensity Normalization")

        volume = self.normalize(volume)

        print("Done")

        print()

        print("Step 2 : Outlier Removal")

        volume = self.outlier(volume)

        print("Done")

        print()

        print("Step 3 : Noise Reduction")

        volume = self.noise(volume)

        print("Done")

        print()

        print("Step 4 : Adaptive Gaussian Initialization")

        gaussian = self.gaussian(volume)

        print("Done")

        print()

        print("Step 5 : Covariance Preparation")

        covariance = self.covariance(
            gaussian
        )

        output_file = os.path.join(
            output_directory,
            "preprocessed.nii.gz"
        )

        self.writer.save(
            gaussian,
            affine,
            header,
            output_file
        )

        np.save(
            os.path.join(
                output_directory,
                "covariance.npy"
            ),
            covariance
        )

        print()

        print("=======================================")
        print("Preprocessing Completed Successfully")
        print("=======================================")


# ==========================================================
# Command Line
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Input diffusion MRI (.nii.gz)"
    )

    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output directory"
    )

    args = parser.parse_args()

    pipeline = PreprocessingPipeline()

    pipeline.process(
        args.input,
        args.output
    )
