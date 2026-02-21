# Research Findings for ICLR 2026 Re-Align Challenge
## Goal: Select 1000 images to maximize representational divergence (minimize CKA) across ~141 diverse vision models

---

## 1. CNN vs ViT Representation Differences: Texture Bias vs Shape Bias

### Original Geirhos et al. (2019) - ICLR
- Established that ImageNet-trained CNNs are biased towards **texture** over shape
- Cue-conflict experiments: images with cat shape + elephant texture → CNNs classify as elephant, humans classify as cat
- Paper: "ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness"

### 2025 Revision: Burgert et al. (arXiv:2509.20234)
- **Major update**: CNNs are NOT inherently texture-biased; they predominantly rely on **local shape features**
- Domain-specific reliance patterns:
  - **Computer vision models** → prioritize shape
  - **Medical imaging models** → emphasize color
  - **Remote sensing models** → stronger texture reliance
- Modern architectures (ConvNeXt, ViTs) substantially mitigate these biases
- **Implication for challenge**: The texture vs shape axis alone may not be the strongest divergence driver between modern models

### Hermann et al. (NeurIPS 2020)
- Data augmentation (not architecture) may be the primary driver of texture vs shape bias
- Less aggressive random crops + naturalistic augmentation → shape-biased models
- **Key insight**: Training procedure matters as much as architecture for feature reliance

### Actionable for challenge:
- **Cue-conflict images** (texture-shape mismatches) will maximally separate old-school CNNs from modern ViTs and CLIP models
- Images where texture and shape give contradictory class signals are high-value targets
- However, since the model zoo likely includes models with diverse training procedures, the training paradigm axis may matter more than architecture alone

---

## 2. Which ImageNet Images Are "Hard" Differently for Different Architectures

### ImageNet-Hard (NeurIPS 2023)
- Benchmark of images that remain difficult even with optimal zooming/framing
- Revealed **positional biases**: center bias in ImageNet-A and ObjectNet
- Models employ "information-discarding strategies" — focusing on specific discriminative regions
- **Key finding**: Proper image framing achieves 98.91% on standard ImageNet but not on ImageNet-Hard

### ImageNet-X (ICLR 2023) — Most relevant for your challenge
- Human annotations of **16 factors of variation** for entire ImageNet-1k validation set:
  - Pose, background, lighting, size, color, texture, occlusion, etc.
- Analyzed **2,200 recognition models** across architectures and training paradigms
- **Critical finding**: Models have **consistent** failure modes across ImageNet-X categories regardless of architecture
- **BUT**: Data augmentation creates unexpected **spill-over effects**:
  - Color-jitter augmentation → improves color/brightness robustness → **hurts pose robustness**
  - These spill-over effects differ across model designs
- **Actionable**: Images with unusual poses where color-jitter-trained models fail but non-augmented models succeed (and vice versa) will cause divergence

### ImageNot (2024)
- Surprising finding: model architecture rankings are **preserved** across drastically different datasets
- Relative improvements between architectures strongly correlate across datasets
- **Implication**: Relative performance is stable, but absolute accuracy drops cause images to fall into different "hard/easy" buckets per architecture

### Remaining Mistakes Analysis (arXiv:2205.04596)
- Nearly half of "errors" on ImageNet involve **semantically similar classes** (e.g., bagel vs dough)
- 40% of top model mistakes are "obviously wrong" — genuine model confusion
- **Actionable**: Fine-grained distinctions between visually similar categories are prime targets

---

## 3. ObjectNet Evaluation: Per-Architecture Performance Gaps

### ObjectNet (NeurIPS 2019)
- 50,000 images with controlled variations in backgrounds, rotations, viewpoints
- **40-45% performance drop** vs ImageNet across all architectures
- Removes dataset biases: object-background correlations, stereotypical rotations, limited occlusion
- Fine-tuning shows only small improvements → biases are deeply embedded

### Battle of the Backbones (2023)
- Large-scale comparison across CNNs and ViTs on diverse vision tasks
- Supervised CNNs on large datasets **still perform best on most tasks**
- Self-supervised backbones competitive when given comparable architectures/data
- **Key gap**: The supervised vs self-supervised split matters more for representation structure than CNN vs ViT

### Actionable for challenge:
- Images with **unusual viewpoints, atypical backgrounds, and non-canonical orientations** will separate models differently
- ObjectNet-style controlled variations are exactly the kind of images that expose architecture-specific biases

---

## 4. CKA Between Different Model Families

### Kornblith et al. (ICML 2019) — Original CKA paper
- CKA is invariant to orthogonal transformations and isotropic scaling
- Can reliably identify correspondences between representations in networks trained from different initializations
- Superior to CCA-based approaches for comparing architectures

### Critical CKA Limitations (relevant to your challenge metric!)
- **Biased CKA** can be artificially driven to maximum values using independent random data with different sample-feature ratios
- Biased CKA is sensitive to differing feature-sample ratios rather than actual stimulus-driven responses
- **Recommendation**: Use debiased CKA for reliable comparisons
- **For your challenge**: Understanding that CKA has known failure modes means you should validate your image selection strategy against these pathologies

### CKA Across Vision Architectures
- ViT-B/32, ViT-L/16, ViT-H/14, ResNet50x1, ResNet152x2 systematically compared
- Typical computation: batch sizes of 1024, sampling 10,240 examples, repeated 20 times
- Different model families show systematically different internal representation hierarchies

### Model Stitching Reveals What CKA Misses
- Task loss matching can be misleading — may indicate high similarity between actually different layers
- **Direct matching** (minimizing distance between representations) is more reliable
- Good networks trained differently (supervised vs self-supervised) can be stitched with minimal performance drop
- **Implication**: CKA may underestimate some representation differences; your strategy should consider what CKA actually measures

---

## 5. Representation Topology Divergence & Representational Similarity Analysis

### RTD (Barannikov et al., ICML 2022)
- Topological data analysis (TDA)-based method for comparing neural network representations
- Measures dissimilarity in multi-scale topology between point clouds
- Sensitive to topological structure — captures aspects CKA misses
- Applications: training dynamics, distribution shift detection, transfer learning

### Data-Driven Typology of Vision Models (Wu et al., 2025, arXiv:2509.21628)
**This is the most directly relevant paper for your challenge strategy:**
- Uses multiple complementary metrics: RSA, Soft Matching, CKA, Linear Predictivity
- **Key findings on what separates model families:**
  1. **Geometry and tuning** (RSA, Soft Matching) → strong family discrimination
  2. **Linear decodability** (Linear Predictivity) → weak separation (broadly shared)
  3. **Supervised ResNets and ViTs** → form **distinct clusters**
  4. **All self-supervised models** → group together **across architectural boundaries**
  5. **Hybrid architectures** (ConvNeXt, Swin) → cluster with **masked autoencoders**
- **Critical insight**: "Emergent computational strategies shaped jointly by architecture and training objective define representational structure beyond surface design categories"
- **Actionable**: To maximize divergence, select images that separate these specific clusters: supervised ResNets vs supervised ViTs vs self-supervised models vs CLIP-like models

### ReSi Benchmark
- Comprehensive evaluation of 24 similarity measures across 14 architectures and 7 datasets
- Provides standardized comparison framework

### Representational Similarity via Interpretable Visual Concepts (arXiv:2503.15699)
- Uses interpretable visual concepts to understand what drives similarity/divergence

---

## 6. The Platonic Representation Hypothesis — And Its Counterexamples

### Huh et al. (ICML 2024)
- **Core claim**: Neural network representations across AI models are converging toward a shared statistical model of reality
- As vision and language models scale larger, they measure distances between datapoints in increasingly similar ways
- **Cross-modal convergence**: Vision and language representations becoming more aligned

### Critical Counterexamples (where convergence breaks down = your opportunity)
- **Dataset-dependent consistency**: Representational similarity between models is NOT universally consistent across different datasets
- **Objective function matters**: Self-supervised vision models show better generalization of representational similarities across datasets vs image classification or image-text models
- **Image-text models** and **image classification models** show **weaker cross-dataset consistency** than self-supervised approaches
- **Key insight for challenge**: The Platonic convergence is strongest for "easy" canonical images. **The divergence you want lives in images where this convergence breaks down** — unusual viewpoints, ambiguous categories, atypical contexts, multi-object scenes

### Actionable strategy:
- Focus on images that are **not well-described by a single "platonic" representation**
- These are images where different model objectives (contrastive, classification, reconstruction, language-guided) lead to fundamentally different similarity judgments
- Multi-object scenes where object prominence is ambiguous
- Images where color, shape, and texture give contradictory category signals

---

## 7. How Data Augmentation Affects Different Architectures Differently

### How to Train Your ViT? (Steiner et al., 2021)
- ViTs require **stronger augmentation** than CNNs due to weaker inductive bias
- ViTs more prone to overfitting on smaller datasets
- Data augmentation + compute can match models trained on orders of magnitude more data

### Augmentation-Specific Architecture Vulnerabilities
- **Color Jitter**: Degrades color feature quality; Planckian Jitter (physics-based) is better for maintaining color discrimination
- **Rotation/Mixing (Mixup)**: Causes variance shifts in ViT positional embeddings, degrading test performance — **specific to ViTs**, CNNs not affected
- **Random crops**: Aggressive crops harm ViTs more due to patch-based processing

### Spill-Over Effects (ImageNet-X)
- Color-jitter-trained models: better on color/brightness variations, **worse on pose changes**
- These effects differ across model designs
- **Actionable**: Images with unusual poses will separate color-jitter-augmented models from non-augmented ones; rotated images will separate ViTs from CNNs

### NoisyMix (2024)
- Noise augmentations in both input and feature spaces improve robustness against ImageNet-C and ImageNet-R for both architectures
- But the degree of improvement differs between CNNs and ViTs

---

## 8. ICLR 2026 Re-Align Workshop & Challenge

### Workshop Details
- **Third edition** of the Re-Align Workshop (following 2024, 2025)
- Pivots from measuring alignment to **what alignment enables**
- Two focus areas: Neural Control and Downstream Behavior
- Paper submission deadline: February 5, 2026
- Camera-ready: April 19, 2026

### Growth Context
- 688 papers submitted to ICLR 2026 on representational alignment (up from 443 in 2025, 303 in 2024)
- 51% average yearly increase in alignment research

### Challenge Component
- New for 2026 edition
- Introduces a Re-Align Challenge alongside traditional paper submissions
- Challenge reports are accepted as submissions

---

## SYNTHESIS: Strategy for Maximizing Representational Divergence

### Image Properties That Maximize Cross-Architecture Disagreement

Based on all research above, here are the image properties most likely to minimize CKA across 141 diverse models:

#### Tier 1: Highest-Impact Properties

1. **Unusual viewpoints and orientations** — ObjectNet shows 40-45% drops; rotations specifically hurt ViTs through positional embedding variance shifts
2. **Texture-shape cue conflicts** — Where texture suggests one class and shape suggests another; maximally separates CNNs, ViTs, and CLIP models
3. **Multi-object scenes with ambiguous prominence** — CLIP prioritizes larger objects (image encoder) vs first-mentioned (text encoder) vs other models' strategies
4. **Fine-grained distinctions between semantically similar classes** — Bagel vs dough, similar dog breeds; nearly half of model disagreements are here
5. **Atypical backgrounds and contexts** — ObjectNet-style removal of object-background correlations

#### Tier 2: Strong Divergence Drivers

6. **Images where color is the primary discriminative cue** — Separates models trained with color-jitter (color-invariant) from those without
7. **Occluded or partially visible objects** — Different architectures handle missing information differently
8. **Images with both local texture patterns AND global shape information** — Local vs global processing differs between CNNs (local → global) and ViTs (global from early layers)
9. **Non-canonical object scales** — Very large or very small objects in frame
10. **Ambiguous/multi-label images** — Where reasonable models could disagree on the primary category

#### Tier 3: Supplementary Divergence

11. **Images from ImageNet-A** (natural adversarial) — Specifically designed to cause model failures, with known center-bias exploitation
12. **High spatial frequency patterns vs low-frequency** — CNNs and ViTs process frequency information differently
13. **Stylized or artistic renderings** — Test texture vs shape processing in extreme cases
14. **Images requiring context-dependent interpretation** — Where background/context changes the meaning of the foreground object

### OOD vs In-Distribution Edge Cases

**Research verdict**: Both contribute, but differently:
- **OOD images** cause larger absolute divergence but may cause ALL models to fail (low signal)
- **In-distribution edge cases** (unusual viewpoints, fine-grained distinctions, atypical contexts) cause models to **disagree with each other** rather than all failing together
- **Best strategy**: Focus on **in-distribution edge cases** and **controlled OOD shifts** (unusual viewpoints/backgrounds within known categories) rather than fully OOD images

### Model Family Clusters to Separate

Based on the typology research, your 141 models likely cluster into:
1. **Supervised ResNets** (distinct cluster)
2. **Supervised ViTs** (distinct cluster)  
3. **Self-supervised models** (cluster together regardless of architecture)
4. **Hybrid architectures** (ConvNeXt, Swin) — cluster with masked autoencoders
5. **CLIP-like vision-language models** (weakest cross-dataset consistency)

Your image selection should aim to find images that produce **different pairwise similarity structures** across ALL of these cluster boundaries, not just CNN vs ViT.

---

## Key Papers to Read in Full

1. **Wu et al. (2025)** — "A Data-driven Typology of Vision Models" (arXiv:2509.21628) — Most directly relevant
2. **Geirhos et al. (2019)** — Texture vs shape bias (ICLR)
3. **Burgert et al. (2025)** — Revision of texture bias (arXiv:2509.20234)
4. **ImageNet-X (2023)** — Factor of variation annotations (ICLR)
5. **Huh et al. (2024)** — Platonic Representation Hypothesis (ICML)
6. **Kornblith et al. (2019)** — CKA metric (ICML)
7. **Barannikov et al. (2022)** — Representation Topology Divergence (ICML)
8. **ObjectNet (2019)** — Bias-controlled evaluation (NeurIPS)
9. **ImageNet-Hard (2023)** — Hardest images benchmark (NeurIPS)
10. **D-BAT / Pathologies of Predictive Diversity** — Ensemble disagreement patterns
