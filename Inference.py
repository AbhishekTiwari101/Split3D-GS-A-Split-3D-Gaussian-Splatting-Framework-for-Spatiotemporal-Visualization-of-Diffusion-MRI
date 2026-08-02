===============================================================
Split3D-GS Inference Pipeline
--------
1. Load pretrained Split3D-GS model
2. Load preprocessed diffusion MRI
3. Estimate covariance matrices
4. Split Gaussian via eigenspace decomposition
5. Preserve statistical moments
6. Adaptive Gaussian optimization
7. Surface bounded Gaussian splatting
8. Differentiable rendering
9. Save visualization
10. Compute quantitative metrics
===============================================================

import argparse
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from split_gaussian import SplitGaussianSplatter
from renderer import GaussianRenderer
from metrics import (
    compute_mse,
    compute_ssim,
    boundary_sharpness,
    point_density
)


class Split3DGSInference:

    def __init__(self,
                 checkpoint,
                 device="cuda"):

        self.device = device

        ####################################################
        # Load trained network
        ####################################################

        self.model = SplitGaussianSplatter(
            volume_shape=(96,96,96),
            device=device
        )

        checkpoint = torch.load(
            checkpoint,
            map_location=device
        )

        self.model.load_state_dict(
            checkpoint["model"]
        )

        self.model.eval()

        ####################################################
        # Gaussian Renderer
        ####################################################

        self.renderer = GaussianRenderer(
            device=device
        )

    ########################################################

    def load_volume(self, filename):

        image = nib.load(filename)

        volume = image.get_fdata()

        volume = volume.astype(np.float32)

        volume = (
            volume-volume.min()
        )/(volume.max()-volume.min()+1e-8)

        return torch.from_numpy(
            volume
        ).to(self.device)

    ########################################################

    @torch.no_grad()

    def inference(self,
                  volume):

        start = time.time()

        ###############################################
        # Forward Pass
        ###############################################

        gaussian = self.model(volume)

        ###############################################
        # Differentiable Rendering
        ###############################################

        rendered = self.renderer(gaussian)

        ###############################################
        # Evaluation
        ###############################################

        mse = compute_mse(
            volume,
            rendered
        )

        score = compute_ssim(
            volume,
            rendered
        )

        sharpness = boundary_sharpness(
            rendered
        )

        density = point_density(
            rendered
        )

        runtime = time.time()-start

        metrics = {

            "MSE": mse,

            "SSIM": score,

            "BoundarySharpness": sharpness,

            "PointDensity": density,

            "InferenceTime": runtime

        }

        return rendered, metrics


##############################################################

def main(args):

    engine = Split3DGSInference(

        checkpoint=args.weights,

        device=args.device

    )

    volume = engine.load_volume(
        args.input
    )

    rendered, metrics = engine.inference(
        volume
    )

    Path(args.output).mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(

        rendered.cpu(),

        Path(args.output)/"splats.pt"

    )

    print("\n============================")

    print(" Split3D-GS Inference")

    print("============================")

    for k,v in metrics.items():

        print(f"{k:20s}: {v:.5f}")

    print("============================")


##############################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        type=str
    )

    parser.add_argument(
        "--weights",
        default="checkpoints/best_model.pth"
    )

    parser.add_argument(
        "--output",
        default="results"
    )

    parser.add_argument(
        "--device",
        default="cuda"
    )

    args = parser.parse_args()

    main(args)

# Inference workflow
Input dMRI Volume (.nii.gz)
          │
          ▼
Load Pre-trained Split3D-GS Network
          │
          ▼
Voxel-wise Gaussian Parameter Estimation
          │
          ▼
Covariance Matrix Construction
          │
          ▼
Split Gaussian using SVD
          │
          ▼
Moment Preservation
          │
          ▼
Adaptive σ Estimation (FA-guided)
          │
          ▼
Surface Bounding Constraint
          │
          ▼
Differentiable Gaussian Rendering
          │
          ▼
Rendered 3D Gaussian Volume
          │
          ▼
Evaluation:
 • MSE
 • SSIM
 • Boundary Sharpness
 • Point Cloud Density
 • Inference Time
