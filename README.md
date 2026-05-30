# Adaptive_Skeleton_Aware_Framework
## Overview

This project proposes an **Adaptive Skeleton-Aware Training Framework** built on a 3D CAD-UNet backbone for multiclass segmentation of cerebral vessels in the Circle of Willis (CoW). Standard overlap-based losses (Cross-Entropy + Dice) fail to preserve thin vessel connectivity and recover fine distal branches. This work addresses that by introducing an adaptive weighting scheme for skeleton recall loss that stabilizes training and significantly improves vessel recovery.

The framework is evaluated on the **TopCoW CTA benchmark** and compared against:
- Baseline (CE + Dice)
- Fixed-weight Skeleton Recall Loss (Kirchhoff et al., 2024)
- **Adaptive Skeleton Loss (Proposed)**

---

## Key Results

Performance across **n = 8 held-out CTA validation volumes** (mean ± std):

| Metric | Baseline (CE+Dice) | Fixed Skeleton | Adaptive (Proposed) |
|---|---|---|---|
| Binary Dice | 0.395 ± 0.068 | 0.534 ± 0.069 | **0.727 ± 0.056** |
| Macro Fg Dice | 0.087 ± 0.018 | 0.080 ± 0.014 | **0.350 ± 0.029** |
| Major Artery Dice | 0.126 ± 0.025 | 0.126 ± 0.025 | **0.457 ± 0.040** |
| Small Artery Dice | 0.000 ± 0.000 | 0.029 ± 0.010 | **0.164 ± 0.027** |
| Vein Dice | 0.206 ± 0.103 | 0.085 ± 0.055 | **0.537 ± 0.147** |
| Macro Skel Recall | 0.093 ± 0.018 | 0.106 ± 0.020 | **0.375 ± 0.021** |
| Macro HD95 (mm) | 42.4 ± 16.8 | 54.4 ± 6.5 | **25.4 ± 6.7** |
| Macro ASSD (mm) | 10.9 ± 4.9 | 14.6 ± 2.5 | **6.2 ± 2.0** |
| Classes Recovered (of 32) | 7.4 ± 0.9 | 11.2 ± 1.3 | **25.0 ± 1.7** |

Significance markers indicate paired Wilcoxon signed-rank test results against the proposed method: * p < 0.05, ** p < 0.01, *** p < 0.001.

The proposed framework recovers **3.4× more vessel classes** than the baseline and achieves statistically significant improvements across all metrics (p < 0.01).

---

## Method

### Architecture
- **Backbone:** 3D CAD-UNet with coordinate attention and dilated convolutions
- **Input:** CTA volumes, patch-based sliding window inference
- **Output:** Multiclass vessel segmentation (32 anatomical classes)

### Loss Function
The total loss combines a generic segmentation loss with an adaptive skeleton recall term:

```
L = L_generic + w(t) · L_SkelRecall
```

Where `w(t)` is an adaptive weight that adjusts dynamically during training based on validation Dice feedback, preventing the skeleton recall term from dominating early training and destabilizing convergence.

### Why Adaptive Weighting?
Fixed-weight skeleton supervision was found to destabilize training — the skeleton recall reward dominates when cross-entropy gradients become small, causing negative loss curves and oscillating validation Dice. Adaptive weighting resolves this by scaling the skeleton contribution relative to training progress, consistent with the weight sensitivity analysis of Kirchhoff et al. (2024).

---

## Dataset

- **Dataset:** [TopCoW Challenge](https://topbrain2025.grand-challenge.org/) — CTA and MRA volumes of the Circle of Willis
- **Dataset:** [TopCoW Challenge]() — CTA and MRA volumes of the Circle of Willis
- **Task:** Multiclass segmentation of cerebrovascular anatomy
- **Split:** 8 held-out CTA volumes for validation 

---

## Training Configurations

| Configuration | Loss |
|---|---|
| Baseline | Cross-Entropy + Dice |
| Fixed Skeleton | CE + Dice + Skeleton Recall (fixed w) |
| Adaptive (Proposed) | CE + Dice + Skeleton Recall (adaptive w(t)) |

---

## Requirements

```bash
pip install torch torchvision
pip install scikit-image
pip install nibabel
pip install numpy scipy
```

---

## Repository Structure

```
Adaptive_Skeleton_Aware_Framework/
│
├── CAD_Unet_Multiclass_(Skeleton_Loss)_adaptive_weights.ipynb   # Proposed
├── CAD_Unet_Multiclass_(Skeleton_Loss).ipynb                    # Fixed skeleton
├── CAD_Unet_Multiclass_Baseline.ipynb                           # Baseline
│
└── README.md
```

---

## References

- Kirchhoff et al. (2024). *Skeleton Recall Loss for Connectivity Conserving and Resource Efficient Segmentation of Thin Tubular Structures.* arXiv:2404.03010
- Yijie Dang et al. (2025). *CAD-Unet: A Capsule Network-Enhanced Unet Architecture for Accurate Segmentation of COVID-19 Lung Infections from CT Images* arXiv:2412.06314
- Ronneberger et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI.
- Yang et al. (2023). *Benchmarking the CoW with the TopCoW Challenge.* arXiv:2312.17670

---

## Author

**Prabhjot Kumar** — M.Sc. / Research Project
