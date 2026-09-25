"""
Split3D-GS: Moment-Preserving Anisotropic Gaussian Splatting
for Interactive Visualization of Diffusion MRI

Paper-synchronous implementation (closed-form moment preservation,
eigenspace split, adaptive FA scaling, surface bounding).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import svd, eigh


class SplitGaussianSplatter(nn.Module):
    """
    Implements the method described in the paper:
    - Eigenspace decomposition (Eq. 3)
    - Closed-form moment-preserving weights (Eq. 13)
    - Adaptive σ based on FA (Eq. 12)
    - Surface bounding (Eq. 6-7)
    - Anisotropic Gaussian rendering (replaces the previous placeholder)
    """

    def __init__(self, num_splats: int = 50000, volume_size=(96, 96, 96), device="cuda"):
        super().__init__()
        self.device = device
        self.num_splats = num_splats
        self.volume_size = volume_size

        # Learnable Gaussian parameters
        self.xyz = nn.Parameter(torch.empty(num_splats, 3).uniform_(-1.0, 1.0))
        self.scale = nn.Parameter(torch.ones(num_splats, 3) * 0.05)
        self.opacity = nn.Parameter(torch.ones(num_splats, 1) * 0.5)

        # Quaternion for rotation
        q = torch.zeros(num_splats, 4, device=device)
        q[:, 0] = 1.0
        self.quaternion = nn.Parameter(q)

        # Adaptive parameters (paper: σ0 = 1.0, k = 1.5)
        self.register_buffer("sigma0", torch.tensor(1.0))
        self.register_buffer("k_fa", torch.tensor(1.5))

    # ------------------------------------------------------------------
    # Utility functions matching the paper
    # ------------------------------------------------------------------
    def quaternion_to_rotation(self, q: torch.Tensor) -> torch.Tensor:
        q = F.normalize(q, dim=-1)
        w, x, y, z = q.unbind(-1)
        R = torch.stack([
            1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w,
            2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w,
            2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y
        ], dim=-1).view(-1, 3, 3)
        return R

    def covariance_matrix(self) -> torch.Tensor:
        """Σ = R S² Rᵀ  (paper Eq. 2-3)"""
        R = self.quaternion_to_rotation(self.quaternion)
        S = torch.diag_embed(F.softplus(self.scale))
        Sigma = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)
        return Sigma

    def split_gaussian(self, Sigma: torch.Tensor):
        """
        Eigenspace decomposition (paper Eq. 3).
        Returns U, λ1, λ2 (largest two eigenvalues) and closed-form weights.
        """
        # Use eigh for numerical stability (symmetric positive-definite)
        L, U = eigh(Sigma)                       # L ascending
        # Take the two largest eigenvalues
        lambda1 = L[:, -1]                       # largest
        lambda2 = L[:, -2]                       # second largest
        U_primary = U[:, :, -1]                  # corresponding eigenvectors
        U_secondary = U[:, :, -2]

        # Closed-form moment-preserving weights (paper Eq. 13)
        w1 = lambda1 / (lambda1 + lambda2 + 1e-12)
        w2 = 1.0 - w1
        return U_primary, U_secondary, lambda1, lambda2, w1, w2

    def adaptive_sigma(self, FA: torch.Tensor) -> torch.Tensor:
        """σ_i = σ0 · (1 + k · FA)  (paper Eq. 12)"""
        return self.sigma0 * (1.0 + self.k_fa * FA)

    def surface_bounding_mask(self) -> torch.Tensor:
        """x² + y² + z² ≤ r²  (paper Eq. 6-7)"""
        r = min(self.volume_size) / 2.0
        dist2 = torch.sum(self.xyz ** 2, dim=-1)
        return (dist2 <= r ** 2).float()

    # ------------------------------------------------------------------
    # Anisotropic Gaussian rendering (paper-compliant, pure PyTorch)
    # ------------------------------------------------------------------
    def render_anisotropic(self, volume: torch.Tensor, FA: torch.Tensor = None) -> torch.Tensor:
        """
        Differentiable anisotropic Gaussian splatting.
        Approximates the continuous integral of the split kernels
        on the discrete volume grid (suitable for inference).
        """
        B, D, H, W = 1, *volume.shape[-3:] if volume.dim() == 3 else volume.shape
        device = volume.device

        Sigma = self.covariance_matrix()
        U1, U2, lam1, lam2, w1, w2 = self.split_gaussian(Sigma)

        if FA is None:
            FA = torch.ones(self.num_splats, device=device) * 0.5
        sigma = self.adaptive_sigma(FA).unsqueeze(-1)          # (N,1)

        # Surface bounding
        mask = self.surface_bounding_mask()                    # (N,)
        opacity = torch.sigmoid(self.opacity).squeeze(-1) * mask

        # Create coordinate grid
        zs = torch.linspace(-1, 1, D, device=device)
        ys = torch.linspace(-1, 1, H, device=device)
        xs = torch.linspace(-1, 1, W, device=device)
        grid_z, grid_y, grid_x = torch.meshgrid(zs, ys, xs, indexing="ij")
        coords = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (D,H,W,3)

        # Render by accumulating anisotropic kernels (vectorised over splats in batches)
        rendered = torch.zeros(D, H, W, device=device)
        batch_size = 2048                                       # memory-friendly

        for start in range(0, self.num_splats, batch_size):
            end = min(start + batch_size, self.num_splats)
            xyz_b = self.xyz[start:end]                         # (B,3)
            op_b  = opacity[start:end]                          # (B,)
            sig_b = sigma[start:end]                            # (B,1)
            w1_b  = w1[start:end].unsqueeze(-1)                 # (B,1)
            w2_b  = w2[start:end].unsqueeze(-1)

            # Distance to each splat centre
            delta = coords.unsqueeze(0) - xyz_b.view(-1, 1, 1, 1, 3)  # (B,D,H,W,3)

            # Anisotropic quadratic form using the two principal directions
            # Simplified but paper-aligned: project onto primary & secondary axes
            # (full 3×3 Mahalanobis can be substituted if needed)
            proj1 = (delta * U1[start:end].view(-1, 1, 1, 1, 3)).sum(-1)  # (B,D,H,W)
            proj2 = (delta * U2[start:end].view(-1, 1, 1, 1, 3)).sum(-1)

            # Split-kernel contribution (paper Eq. 4)
            q1 = (proj1 ** 2) / (2 * (sig_b * lam1[start:end].view(-1, 1, 1, 1) + 1e-8))
            q2 = (proj2 ** 2) / (2 * (sig_b * lam2[start:end].view(-1, 1, 1, 1) + 1e-8))

            gauss = w1_b.view(-1, 1, 1, 1) * torch.exp(-q1) + \
                    w2_b.view(-1, 1, 1, 1) * torch.exp(-q2)

            # Opacity-weighted accumulation
            rendered = rendered + (op_b.view(-1, 1, 1, 1) * gauss).sum(0)

        # Normalise for visualisation stability
        rendered = rendered / (rendered.max() + 1e-8)
        return rendered

    def forward(self, volume: torch.Tensor, FA: torch.Tensor = None) -> torch.Tensor:
        """Inference entry point – returns the rendered anisotropic volume."""
        return self.render_anisotropic(volume, FA)

    # ------------------------------------------------------------------
    # Moment verification (paper Section 3.2)
    # ------------------------------------------------------------------
    @staticmethod
    def verify_moment_preservation(num_samples: int = 10000, device="cpu"):
        """
        Numerical verification claimed in the paper:
        max relative error on second-moment matrix < 2.1e-7
        """
        torch.manual_seed(42)
        errors = []
        for _ in range(num_samples):
            # Random SPD matrix
            A = torch.randn(3, 3, device=device)
            Sigma = A @ A.T + 0.1 * torch.eye(3, device=device)
            L, U = eigh(Sigma)
            lam1, lam2 = L[-1], L[-2]
            w1 = lam1 / (lam1 + lam2)
            w2 = 1.0 - w1
            # Reconstructed second moment
            Sigma_hat = w1 * (lam1 * U[:, -1: ] @ U[:, -1: ].T) + \
                        w2 * (lam2 * U[:, -2:-1] @ U[:, -2:-1].T)
            rel_err = torch.norm(Sigma - Sigma_hat) / (torch.norm(Sigma) + 1e-12)
            errors.append(rel_err.item())
        max_err = max(errors)
        print(f"Moment preservation verification ({num_samples} tensors):")
        print(f"  Maximum relative error = {max_err:.3e}")
        assert max_err < 1e-6, "Moment preservation failed!"
        return max_err







#------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import svd



class SplitGaussianSplatter(nn.Module):

    """
    ----------------------------------------------------------------------
    Split3D-GS

    Main Components
    ----------------
    • Learnable anisotropic Gaussian kernels
    • Quaternion-based rotation
    • SPD covariance estimation
    • Split Gaussian decomposition (SVD)
    • Moment-preserving optimization
    • Adaptive FA-aware Gaussian scaling
    • Surface bounding regularization
    • Differentiable Gaussian rendering

    ----------------------------------------------------------------------
    """

    def __init__(
            self,
            num_splats=50000,
            volume_size=(96,96,96),
            device="cuda"):

        super().__init__()

        self.device=device
        self.num_splats=num_splats
        self.volume_size=volume_size

        ####################################################
        # Gaussian Parameters
        ####################################################

        self.xyz=nn.Parameter(
            torch.empty(num_splats,3).uniform_(-1,1)
        )

        self.scale=nn.Parameter(
            torch.ones(num_splats,3)*0.05
        )

        self.opacity=nn.Parameter(
            torch.ones(num_splats,1)*0.5
        )

        ####################################################
        # Rotation (Quaternion)
        ####################################################

        q=torch.zeros(num_splats,4)
        q[:,0]=1.0

        self.quaternion=nn.Parameter(q)

        ####################################################
        # Adaptive Parameters
        ####################################################

        self.base_sigma=nn.Parameter(
            torch.tensor(0.12)
        )

        self.fa_weight=nn.Parameter(
            torch.tensor(0.45)
        )

        self.boundary_weight=0.10

    # =====================================================
    # Quaternion
    # =====================================================

    def quaternion_to_rotation(self,q):

        q=F.normalize(q,dim=-1)

        w,x,y,z=q.unbind(-1)

        R=torch.stack([

            1-2*y*y-2*z*z,
            2*x*y-2*z*w,
            2*x*z+2*y*w,

            2*x*y+2*z*w,
            1-2*x*x-2*z*z,
            2*y*z-2*x*w,

            2*x*z-2*y*w,
            2*y*z+2*x*w,
            1-2*x*x-2*y*y

        ],dim=-1)

        return R.view(-1,3,3)

    # =====================================================
    # Covariance Matrix
    # Σ = R S² Rᵀ
    # =====================================================

    def covariance_matrix(self):

        R=self.quaternion_to_rotation(self.quaternion)

        S=torch.diag_embed(
            F.softplus(self.scale)
        )

        Sigma=R@S@S.transpose(-1,-2)@R.transpose(-1,-2)

        return Sigma

    # =====================================================
    # Split Gaussian
    # =====================================================

    def split_gaussian(self,Sigma):

        U,L,_=svd(Sigma)

        primary=torch.diag_embed(L[:,0])

        secondary=torch.diag_embed(L[:,1])

        tertiary=torch.diag_embed(L[:,2])

        return U,primary,secondary,tertiary

    # =====================================================
    # Adaptive Sigma
    # =====================================================

    def adaptive_sigma(self,FA):

        return self.base_sigma*(1+self.fa_weight*FA)

    # =====================================================
    # Surface Bounding
    # =====================================================

    def boundary_mask(self):

        radius=min(self.volume_size)/2

        dist=torch.sum(self.xyz**2,dim=-1)

        return (dist<radius**2).float()

    # =====================================================
    # Moment Preservation
    # =====================================================

    def moment_loss(self,target,prediction):

        ###############################################
        # Zero Moment
        ###############################################

        L0=(prediction.sum()-target.sum())**2

        ###############################################
        # First Moment
        ###############################################

        mu_p=prediction.mean()

        mu_t=target.mean()

        L1=(mu_p-mu_t)**2

        ###############################################
        # Second Moment
        ###############################################

        var_p=prediction.var()

        var_t=target.var()

        L2=(var_p-var_t)**2

        return L0+0.5*L1+0.5*L2

    # =====================================================
    # Boundary Regularization
    # =====================================================

    def boundary_loss(self):

        mask=self.boundary_mask()

        return ((1-mask)*self.opacity.squeeze()).mean()

    # =====================================================
    # Differentiable Gaussian Rendering
    # =====================================================

    def render(self,volume,Sigma):

        ##################################################
        # Placeholder
        #
        # Replace with CUDA Gaussian Rasterizer
        #
        ##################################################

        smooth=F.avg_pool3d(

            volume.unsqueeze(0).unsqueeze(0),

            kernel_size=3,

            stride=1,

            padding=1

        ).squeeze()

        rendered=0.65*volume+0.35*smooth

        return rendered

    # =====================================================
    # Forward
    # =====================================================

    def forward(self,volume,FA=None):

        Sigma=self.covariance_matrix()

        U,L1,L2,L3=self.split_gaussian(Sigma)

        if FA is None:

            FA=torch.ones(self.num_splats,
                          device=volume.device)*0.5

        sigma=self.adaptive_sigma(FA)

        rendered=self.render(volume,Sigma)

        return rendered

    # =====================================================
    # Total Loss
    # =====================================================

    def loss(self,target,prediction):

        reconstruction=F.mse_loss(
            prediction,
            target
        )

        structural=1-F.cosine_similarity(

            prediction.flatten(),

            target.flatten(),

            dim=0

        )

        moment=self.moment_loss(
            target,
            prediction
        )

        boundary=self.boundary_loss()

        total=(
            reconstruction
            +0.20*structural
            +0.50*moment
            +0.10*boundary
        )

        return {

            "loss":total,

            "mse":reconstruction,

            "moment":moment,

            "boundary":boundary,

            "structure":structural

        }
