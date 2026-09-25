"""
Inference script for Split3D-GS
Produces the visualisation that corresponds to the quantitative results
reported in the paper (Tables 3-6).
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from gaussian_splatter import SplitGaussianSplatter


def load_volume(path: str, device: str = "cuda") -> torch.Tensor:
    """Load a preprocessed dMRI volume (FA-weighted or tensor magnitude)."""
    vol = np.load(path).astype(np.float32)
    vol = torch.from_numpy(vol).to(device)
    # Normalise to [0,1] as described in the paper (Eq. 10)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    return vol


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Running inference on {device}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    volume = load_volume(args.volume, device=device)
    FA = None
    if args.fa is not None:
        FA = load_volume(args.fa, device=device).flatten()
        # Ensure FA has the same number of elements as splats (simple resize)
        if FA.numel() != args.num_splats:
            FA = torch.nn.functional.interpolate(
                FA.view(1, 1, -1), size=args.num_splats, mode="linear", align_corners=False
            ).view(-1)

    # ------------------------------------------------------------------
    # 2. Create model (paper settings)
    # ------------------------------------------------------------------
    model = SplitGaussianSplatter(
        num_splats=args.num_splats,
        volume_size=volume.shape,
        device=device
    ).to(device)

    # Optional: load a checkpoint trained with the moment-preserving loss
    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")

    model.eval()

    # ------------------------------------------------------------------
    # 3. Verify moment preservation (paper claim)
    # ------------------------------------------------------------------
    if args.verify_moments:
        SplitGaussianSplatter.verify_moment_preservation(num_samples=1000, device=device)

    # ------------------------------------------------------------------
    # 4. Inference – anisotropic rendering
    # ------------------------------------------------------------------
    with torch.no_grad():
        rendered = model(volume, FA=FA)

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "rendered_volume.npy", rendered.cpu().numpy())
    print(f"Saved rendered volume → {out_dir / 'rendered_volume.npy'}")

    # Quick statistics that match paper metrics
    print("\nInference statistics (for quick sanity check):")
    print(f"  Rendered shape      : {tuple(rendered.shape)}")
    print(f"  Min / Max / Mean    : {rendered.min():.4f} / {rendered.max():.4f} / {rendered.mean():.4f}")
    print(f"  Non-zero voxels     : {(rendered > 0.01).sum().item()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split3D-GS Inference")
    parser.add_argument("--volume", type=str, required=True,
                        help="Path to preprocessed volume (.npy)")
    parser.add_argument("--fa", type=str, default=None,
                        help="Optional FA map (.npy) for adaptive σ")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional model checkpoint")
    parser.add_argument("--num_splats", type=int, default=50000)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--verify_moments", action="store_true",
                        help="Run the numerical moment-preservation test")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    main(args)






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
