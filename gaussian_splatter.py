import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import svd

# ==========================================================
# Split3D-GS
# Split 3D Gaussian Splatting for Diffusion MRI
# ==========================================================

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
