# Split3D-GS: A Split 3D Gaussian Splatting Framework for Spatiotemporal Visualization of Diffusion MRI

**🌟 State-of-the-Art | Real-time | Anatomically Faithful | Clinically Ready**

[![GitHub stars](https://img.shields.io/badge/stars-★-brightgreen)](https://github.com/abhishek-tiwari/split3d-gs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![OpenGL](https://img.shields.io/badge/OpenGL-4.6+-4285F4.svg)](https://www.opengl.org/)
[![HCP Dataset](https://img.shields.io/badge/Dataset-HCP-blue)](https://www.humanconnectome.org/)

> **"A split-kernel approach that preserves structural moments while delivering 40% lower MSE and real-time spatiotemporal rendering of brain white matter tracts."**

---

## ✨ Highlights & Novelty

This repository presents the **official implementation** of the paper:

> **A Split 3D Gaussian Splatting Framework for Spatiotemporal Visualization of Diffusion MRI**  
> *Abhishek Tiwari, Ankit Vidyarthi, Jaydeep Kishore, Varun Tiwari  
> Q1 SCI Journal

### 🚀 Core Innovations

- **🔬 First Split-Kernel 3D Gaussian Splatting** specifically engineered for anisotropic diffusion tensors
- **🧬 Moment-Preserving Eigenspace Decomposition** — mathematically guarantees preservation of 0th, 1st, and 2nd order statistical moments
- **⏳ True Spatiotemporal Modeling** — captures dynamic evolution of diffusion patterns over time
- **⚡ Adaptive Surface Bounding + Anisotropic Parameter Tuning** — dramatically reduces artifacts and improves boundary sharpness by **28.57%**
- **📈 Closed-Form Optimization** — no heavy neural networks; efficient, interpretable, and GPU-accelerated

### Quantitative Breakthroughs (vs Traditional Gaussian Splatting)

| Metric                    | Improvement       | Value          |
|--------------------------|-------------------|----------------|
| **Mean Squared Error**   | **↓ 40%**         | 0.015          |
| **Structural Similarity**| **↑ 13.33%**      | 0.85           |
| **Point Cloud Density**  | **↑ 24%**         | 3100 pts/mm²   |
| **Boundary Sharpness**   | **↑ 28.57%**      | 4.5/5          |
| **Rendering Speed**      | **↑ 14.82%**      | 38.5 ms/frame  |
| **Memory Usage**         | **↓ 7.89%**       | 3.5 GB         |

**Statistically significant (p < 0.001)** across all metrics on the **Human Connectome Project (HCP)** dataset (100 subjects).

---

## 📊 Visual Results

![Boundary Sharpness Comparison](https://github.com/user-attachments/assets/example-boundary.png)
*Left: Traditional Splatting • Right: Our Split3D-GS (sharper fiber boundaries)*

![Point Cloud Density](https://github.com/user-attachments/assets/pointcloud-density.png)
*Denser, more anatomically faithful reconstructions*

![Spatiotemporal Rendering](https://github.com/user-attachments/assets/spatio-temporal.gif)
*Dynamic visualization of evolving diffusion patterns*

---

## 🛠️ Features

- Full PyTorch + OpenGL implementation
- Real-time interactive visualization with parameter sliders
- Support for HCP and custom dMRI datasets
- Preprocessing pipeline (normalization, outlier removal, tensor fitting)
- Eigenspace decomposition + moment preservation
- Adaptive σ tuning based on Fractional Anisotropy (FA)
- Export high-quality renders and point clouds
- Comprehensive evaluation suite (MSE, SSIM, Boundary Sharpness, etc.)

---

## 📥 Installation

```bash
git clone https://github.com/abhishek-tiwari/split3d-gs.git
cd split3d-gs

# Create conda environment
conda create -n split3d-gs python=3.10
conda activate split3d-gs

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Optional: Build OpenGL extensions
python setup.py install
