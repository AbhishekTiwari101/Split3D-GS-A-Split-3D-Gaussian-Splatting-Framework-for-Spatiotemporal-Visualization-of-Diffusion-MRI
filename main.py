"""
===========================================================
Split3D-GS
A Split 3D Gaussian Splatting Framework
for Spatiotemporal Visualization of Diffusion MRI

Author : Abhishek Tiwari
===========================================================
"""

import time
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
from scipy.linalg import svd
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim


############################################################
# Step 1 : Load diffusion MRI dataset
############################################################

class DiffusionDataset:

    def __init__(self, filename):
        self.filename = filename

    def load(self):

        img = nib.load(self.filename)
        data = img.get_fdata()

        print("Dataset Loaded")
        print("Shape :", data.shape)

        return data


############################################################
# Step 2 : Intensity Normalization
############################################################

class Normalization:

    @staticmethod
    def minmax(volume):

        vmin = volume.min()
        vmax = volume.max()

        volume = (volume-vmin)/(vmax-vmin+1e-8)

        return volume


############################################################
# Step 3 : Outlier Removal
############################################################

class OutlierRemoval:

    @staticmethod
    def zscore(volume, threshold=3):

        mean = np.mean(volume)
        std = np.std(volume)

        z = (volume-mean)/(std+1e-8)

        volume[np.abs(z) > threshold] = mean

        return volume


############################################################
# Step 4 : Voxel-wise Gaussian Generation
############################################################

class GaussianGenerator:

    def __init__(self, sigma=1.0):

        self.sigma = sigma

    def generate(self, volume):

        gaussian_volume = gaussian_filter(volume,
                                          sigma=self.sigma)

        return gaussian_volume


############################################################
# Step 5 : Covariance Matrix Estimation
############################################################

class CovarianceEstimator:

    @staticmethod
    def estimate(volume):

        coords = np.argwhere(volume > 0)

        covariance = np.cov(coords.T)

        return covariance


############################################################
# Step 6 : Split Gaussian using SVD
############################################################

class SplitGaussian:

    @staticmethod
    def decompose(covariance):

        U, S, V = svd(covariance)

        eig1 = U[:, 0]
        eig2 = U[:, 1]

        split_covariance = np.diag(S)

        return eig1, eig2, split_covariance


############################################################
# Step 7 : Moment Preservation
############################################################

class MomentPreservation:

    @staticmethod
    def preserve(volume):

        mass = np.sum(volume)

        mean = np.mean(volume)

        variance = np.var(volume)

        preserved = (volume-mean)

        preserved = preserved*np.sqrt(
            variance/(np.var(preserved)+1e-8)
        )

        preserved += mean

        preserved *= mass/(np.sum(preserved)+1e-8)

        return preserved


############################################################
# Step 8 : Adaptive Gaussian Optimization
############################################################

class AdaptiveGaussian:

    def __init__(self,
                 iterations=100,
                 lr=0.01):

        self.iterations = iterations
        self.lr = lr

    def optimize(self, volume):

        sigma = 1.0

        for i in range(self.iterations):

            smoothed = gaussian_filter(volume, sigma)

            loss = np.mean((smoothed-volume)**2)

            sigma -= self.lr*loss

            sigma = np.clip(sigma,
                            0.5,
                            5)

        return gaussian_filter(volume, sigma)


############################################################
# Step 9 : Surface Bounding
############################################################

class SurfaceBounding:

    @staticmethod
    def apply(volume):

        x, y, z = np.indices(volume.shape)

        cx = volume.shape[0]/2
        cy = volume.shape[1]/2
        cz = volume.shape[2]/2

        radius = min(cx, cy, cz)

        mask = (x-cx)**2 + \
               (y-cy)**2 + \
               (z-cz)**2 <= radius**2

        bounded = np.zeros_like(volume)

        bounded[mask] = volume[mask]

        return bounded


############################################################
# Step 10 : OpenGL / CUDA Rendering (Placeholder)
############################################################

class Renderer:

    def render(self, volume):

        print("--------------------------------")
        print("Rendering Gaussian Splats")
        print("Backend : OpenGL / CUDA")
        print("Frame Ready")
        print("--------------------------------")

        return volume


############################################################
# Evaluation Metrics
############################################################

class Evaluation:

    @staticmethod
    def mse(reference, prediction):

        return mean_squared_error(
            reference.flatten(),
            prediction.flatten()
        )

    @staticmethod
    def ssim(reference, prediction):

        return ssim(reference,
                    prediction,
                    data_range=1.0)

    @staticmethod
    def boundary_sharpness(volume):

        gx, gy, gz = np.gradient(volume)

        grad = np.sqrt(gx**2 +
                       gy**2 +
                       gz**2)

        return np.mean(grad)

    @staticmethod
    def point_density(volume):

        return np.sum(volume > 0)/volume.size


############################################################
# Complete Split3D-GS Pipeline
############################################################

class Split3DGS:

    def __init__(self):

        self.normalizer = Normalization()

        self.gaussian = GaussianGenerator()

        self.optimizer = AdaptiveGaussian()

        self.renderer = Renderer()

    def run(self, filename):

        start = time.time()

        ####################################################
        # Load Dataset
        ####################################################

        dataset = DiffusionDataset(filename)

        volume = dataset.load()

        ####################################################
        # Normalization
        ####################################################

        volume = self.normalizer.minmax(volume)

        ####################################################
        # Outlier Removal
        ####################################################

        volume = OutlierRemoval.zscore(volume)

        ####################################################
        # Gaussian Generation
        ####################################################

        gaussian = self.gaussian.generate(volume)

        ####################################################
        # Covariance Estimation
        ####################################################

        covariance = CovarianceEstimator.estimate(gaussian)

        ####################################################
        # Split Gaussian
        ####################################################

        eig1, eig2, split = SplitGaussian.decompose(
            covariance
        )

        ####################################################
        # Moment Preservation
        ####################################################

        gaussian = MomentPreservation.preserve(
            gaussian
        )

        ####################################################
        # Optimization
        ####################################################

        gaussian = self.optimizer.optimize(
            gaussian
        )

        ####################################################
        # Surface Bounding
        ####################################################

        gaussian = SurfaceBounding.apply(
            gaussian
        )

        ####################################################
        # Rendering
        ####################################################

        rendered = self.renderer.render(
            gaussian
        )

        ####################################################
        # Evaluation
        ####################################################

        mse = Evaluation.mse(volume,
                             rendered)

        score = Evaluation.ssim(volume,
                                rendered)

        sharpness = Evaluation.boundary_sharpness(
            rendered
        )

        density = Evaluation.point_density(
            rendered
        )

        runtime = time.time()-start

        print("\n=========== Results ===========")

        print(f"MSE                 : {mse:.5f}")
        print(f"SSIM                : {score:.4f}")
        print(f"Boundary Sharpness  : {sharpness:.4f}")
        print(f"Point Density       : {density:.4f}")
        print(f"Rendering Time      : {runtime:.3f} sec")

        print("===============================")

        return rendered


############################################################
# Main
############################################################

if __name__ == "__main__":

    pipeline = Split3DGS()

    output = pipeline.run(
        "HCP_subject.nii.gz"
    )

model = SplitGaussianSplatter(num_splats=30000, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

for epoch in range(100):
    rendered = model(dwi_volume, fa_map)
    loss = model.compute_total_loss(dwi_volume, rendered)
    loss.backward()
    optimizer.step()
