# Red Team Strategy: Maximizing Representational Divergence (1 - avg CKA)

## ICLR 2026 Re-Align Hackathon

---

## 1. Challenge Summary

**Goal:** Select exactly 1000 images from `imagenet_val` or `objectnet` that cause the
fixed set of ~143 vision models to produce **maximally divergent representations**, scored
as `1 - avg CKA` (higher = more divergence = better).

**Key constraint:** CKA is computed on embeddings extracted from a fixed set of **held-out
images** by the organizers, but our 1000 submitted images define the "stimulus set" that
the models must process. The avg CKA is computed across all pairwise model comparisons.

**Available datasets:** ImageNet validation (50,000 images), ObjectNet (~50,000 images).

---

## 2. Understanding CKA and What Drives Divergence

### 2.1 How CKA Works

CKA (Centered Kernel Alignment) compares two sets of representations by examining their
**pairwise similarity structures** (centered Gram matrices) over a shared set of stimuli
[Kornblith et al., 2019]. Given N images and two models producing representations
X (N x d1) and Y (N x d2):

```
CKA(X, Y) = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))
```

**Critical insight for our strategy:** CKA measures whether two models impose the
*same similarity structure* on the stimulus set. Two models have high CKA when they
agree on which images are similar to each other and which are different. Low CKA means
the models organize the same images in fundamentally different ways.

### 2.2 What Makes CKA Low?

CKA will be low when different models **disagree on the relative similarity relationships
among images**. This happens when:

1. **Different features dominate:** Model A clusters images by texture while Model B
   clusters by shape -- the same images end up in different neighborhoods.
2. **Different invariances:** Model A is invariant to rotation but sensitive to background;
   Model B is the opposite -- they organize the same images differently.
3. **Different failure modes:** Model A collapses representations for certain images
   (treats them as identical) while Model B separates them, or vice versa.

### 2.3 The Model Zoo: Why Architecture Diversity is Our Ally

The 143 models span a massive range of architectures:

| Architecture Family | Examples in Model Zoo | Key Representational Properties |
|---|---|---|
| **Pure CNNs** | VGG, ResNet, DenseNet, DarkNet, EfficientNet, MobileNet, RepVGG | Local texture bias, hierarchical features, translation equivariance |
| **Pure Transformers** | ViT, BEiT, DeiT, Swin, CaiT, CrossViT, PiT, XCiT | Global attention, shape bias, uniform layer representations |
| **Hybrid CNN-Transformer** | CoAtNet, MaxViT, CoaT, DaViT, EfficientFormer | Mix of local and global processing |
| **MLP-Mixers** | gMLP, ResMLP, Mixer-B | Token mixing without attention or convolution |
| **NAS-searched** | NASNet, PNASNet, MNASNet, FBNet | Automated architectures with idiosyncratic features |
| **CLIP-aligned** | vit_base_mci_224, vitamin_base_224 | Vision-language alignment, different training objective |
| **SAM/MAE models** | samvit, sam2_hiera, hiera (MAE) | Segmentation/reconstruction pretraining |
| **Self-supervised** | resmlp_12_224.fb_dino | DINO training, cluster-based representations |
| **Exotic/novel** | Sequencer2D (LSTM), MambaOut (SSM), ConvMixer | Highly unusual inductive biases |

This diversity is the key leverage point. The wider the architectural spread, the more
opportunities for representational disagreement on the right stimuli.

---

## 3. Research-Backed Drivers of Representational Divergence

### 3.1 Texture vs. Shape Bias (CNNs vs. Transformers)

**Key finding:** CNNs are biased toward local texture features, while ViTs develop more
shape-sensitive, globally-informed representations [Geirhos et al., 2019; Naseer et al.,
2021]. A 2025 study nuanced this: CNNs rely on *local shape* rather than texture per se,
but the distinction from ViT's global shape processing remains [arXiv:2509.20234].

**Implication for image selection:** Images where texture and shape provide **conflicting
signals** will cause CNNs and Transformers to organize representations differently. This
includes:
- Objects with misleading textures (e.g., a cat-shaped object with wood grain texture)
- Strong, repetitive textures that dominate local receptive fields
- Images where global shape is clear but local texture is ambiguous

### 3.2 Out-of-Distribution: ObjectNet's Controlled Biases

**Key finding:** Models trained on ImageNet show a **40-45% accuracy drop** on ObjectNet
due to its controlled removal of dataset biases [Barbu et al., 2019]:
- Random backgrounds (vs. stereotypical backgrounds in ImageNet)
- Random object rotations (vs. canonical orientations)
- Diverse viewpoints (vs. limited viewpoints)

**Implication:** ObjectNet images systematically break the shortcuts different models
have learned. Since different architectures exploit *different* shortcuts (texture-based
vs. context-based vs. spatial-frequency-based), ObjectNet images will cause architectures
to diverge because they each fall back on their own idiosyncratic fallback features.

### 3.3 Natural Adversarial Examples

**Key finding:** ImageNet-A [Hendrycks et al., 2021] contains naturally occurring images
that cause **90%+ accuracy drops** even in strong models. These are images where spurious
cues are minimized, and different models fail in different ways. Critically, different
architectures (AlexNet, VGG, ResNet, DenseNet) all fail but *on different subsets and
in different ways*.

**Implication:** Images that are "natural adversarial examples" -- naturally hard images
lacking shortcut features -- will produce divergent internal representations because each
model falls back on its own architectural biases when easy features are absent.

### 3.4 Controversial Stimuli

**Key finding:** Golan, Raju, and Kriegeskorte [PNAS, 2020] introduced "controversial
stimuli" -- images explicitly designed to maximize prediction disagreement between neural
networks. These reveal models' distinct inductive biases more effectively than natural
images because they are not constrained to typical image statistics.

**Implication:** While we can't generate synthetic stimuli (limited to ImageNet/ObjectNet),
we should look for images in the catalog that approximate the properties of controversial
stimuli: ambiguous content, unusual visual statistics, or images at the boundary of
multiple categories.

### 3.5 Occlusion, Pose, and Viewpoint Variation

**Key finding:** ViTs maintain higher accuracy under severe occlusions due to flexible
self-attention, while CNNs degrade more sharply [Naseer et al., 2021]. However, both
architectures fail on unusual poses and viewpoints, but in *different ways* -- ViTs
fail differently than CNNs on the same distribution shift [Schott et al., 2022].

**Key finding:** Robustness to pose/viewpoint variation is learned as *class-specific*
adaptations, not general principles [Djolonga et al., 2021]. Different models learn
different class-specific invariances, meaning the same unusual-pose image will trigger
different failure modes across models.

### 3.6 The 42.5% Zone: Images That Differentiate Models

**Key finding:** Only **42.5%** of ImageNet validation images actually differentiate
between model decision boundaries [Recht et al., OpenReview]. 46% are "trivial"
(all models get right) and 11.5% are "impossible" (all models get wrong). The images
in the differentiation zone are most valuable.

**Implication:** Avoid "trivial" images (clear, well-lit, canonical views of common
objects) and "impossible" images (genuinely ambiguous/mislabeled). Focus on the middle
zone where models disagree.

### 3.7 Training Objective Divergence

**Key finding:** Representational similarity between models is **highly dependent on
training objective** [arXiv:2411.05561]. Self-supervised models (DINO, MAE) form
different representational clusters than supervised models, and CLIP-aligned models
form yet another cluster.

**Implication:** Our model zoo includes supervised, self-supervised (DINO, MAE),
and CLIP-aligned models. Images that cause these different training regimes to
disagree maximally will be high-value. Self-supervised models may represent images
based on low-level visual structure, while CLIP models organize by semantic concepts.

---

## 4. Recommended Strategy: Three-Phase Approach

### Phase 1: Broad Empirical Screening (Compute-Heavy, High-Value)

**This is the most important phase.** Rather than relying solely on theory, we should
directly measure CKA divergence.

#### Step 1a: Extract Embeddings from a Diverse Model Subset

Select 10-15 maximally diverse models from the zoo as proxies:
- Pure CNN: `vgg11.tv_in1k`, `resnet101.a1_in1k`, `densenet121.ra_in1k`
- Pure ViT: `deit3_base_patch16_224.fb_in1k`, `beit_base_patch16_224.in22k_ft_in22k`
- Swin: `swin_base_patch4_window12_384.ms_in1k`
- CLIP-aligned: `vit_base_mci_224.apple_mclip`, `vitamin_base_224.datacomp1b_clip`
- Self-supervised: `resmlp_12_224.fb_dino`, `hiera_base_224.mae`
- MLP-Mixer: `mixer_b16_224.goog_in21k`, `gmlp_s16_224.ra3_in1k`
- Exotic: `sequencer2d_l.in1k`, `convmixer_1024_20_ks9_p14.in1k`, `mambaout_base.in1k`
- SAM: `samvit_base_patch16.sa1b`

Extract embeddings for ALL images in both datasets using these ~15 models.

#### Step 1b: Compute Per-Image "Divergence Score"

For each image `i`, compute a divergence contribution score. This is non-trivial because
CKA operates on sets of images, not individual images. Approach:

**Method A: Leave-One-Out Influence**
1. Take a reference set of ~500 random images
2. Compute avg CKA across all model pairs
3. For each candidate image, add it to the reference set and recompute CKA
4. Images that decrease CKA the most are our targets

**Method B: Per-Image Representation Disagreement (faster approximation)**
1. For each image, extract the embedding vector from all N models
2. Compute the pairwise cosine similarity between all model pairs' embeddings for that image
3. Compute the variance of these cosine similarities
4. High variance = models disagree on where this image falls in representation space
5. Alternatively, use the mean pairwise distance (high = more spread = more divergence)

**Method C: Greedy Set Construction**
1. Start with an empty set S
2. For each candidate image, score how much adding it to S would decrease avg CKA
3. Greedily add the image with the highest marginal divergence
4. Repeat until |S| = 1000

**Recommended: Use Method B for initial screening (it's O(n) per image), then Method C
for final refinement of the top ~3000 candidates.**

### Phase 2: Theory-Guided Enrichment

After Phase 1 identifies high-divergence candidates, analyze their properties and use
these patterns to find additional candidates:

#### 2a: ObjectNet Priority Classes

Focus on ObjectNet categories where the bias controls are most extreme:
- **Unusual rotations:** Kitchen objects (plates, cups, pans) shown upside-down or
  sideways
- **Non-stereotypical backgrounds:** Indoor objects on outdoor backgrounds and vice versa
- **Extreme viewpoints:** Objects from directly above/below

#### 2b: ImageNet "Difficult Middle" Classes

Target ImageNet classes in the difficulty middle zone:
- **Fine-grained categories:** Where models disagree on feature importance
  (dog breeds, bird species, similar vehicles)
- **Context-dependent objects:** Objects whose identity depends on scene context
  (e.g., "barber chair" vs. "chair")
- **Multi-object scenes:** Images containing multiple objects from different classes
- **Unusual exemplars:** Atypical instances within a class

#### 2c: Visual Property Filters

Within both datasets, prioritize images with:
- **High visual complexity:** Cluttered scenes with many objects
- **Strong texture-shape conflicts:** Textured surfaces on distinct shapes
- **Unusual lighting/color:** Low contrast, unusual color palettes, strong shadows
- **Partial occlusion:** Objects partially hidden by other objects
- **Unusual scales:** Very small or very large objects relative to the frame

### Phase 3: Ensemble Optimization

#### 3a: Diversity Within the Selected Set

The 1000 images should be **diverse among themselves** -- CKA is computed over the
set, so including many similar images will create redundancy. Use a diversity-aware
selection criterion:

1. After scoring all candidate images, cluster them by visual/embedding similarity
2. Select images from across all clusters, not just the top-scoring ones
3. Ensure coverage across:
   - Different ObjectNet/ImageNet categories
   - Different image complexity levels
   - Different visual properties (color, texture, shape)

#### 3b: Optimal Mix Ratio (ObjectNet vs. ImageNet)

**Recommended starting ratio: ~600 ObjectNet + ~400 ImageNet, then tune empirically.**

Rationale:
- ObjectNet images are inherently more OOD for all ImageNet-trained models, causing
  larger baseline divergence
- But pure ObjectNet may create uniform divergence (all models confused similarly)
- ImageNet validation images in the "differentiation zone" create targeted divergence
  where models specifically disagree
- The mix ensures both baseline divergence (ObjectNet) and targeted model disagreements
  (ImageNet difficult images)

#### 3c: Final Validation

Before submission:
1. Compute CKA across the full model zoo (or a large subset) on the selected 1000 images
2. Try swap experiments: replace individual images and measure CKA change
3. Check for diminishing returns: if the 1000th image adds negligible divergence,
   the set is saturated

---

## 5. Key Image Properties to Target (Ranked by Expected Impact)

| Rank | Property | Why It Causes Divergence | Source Dataset |
|------|----------|------------------------|----------------|
| 1 | **Non-canonical viewpoints** | ViTs maintain invariance, CNNs lose it; different models have different viewpoint-specific training | ObjectNet (controlled) |
| 2 | **Unusual backgrounds** | Background-dependent models (CNNs) vs. object-focused models (ViTs, CLIP) disagree | ObjectNet (controlled) |
| 3 | **Fine-grained ambiguity** | Models rely on different discriminative features for similar categories | ImageNet (dog breeds, birds, similar objects) |
| 4 | **Multi-object scenes** | Different models attend to different objects; CKA captures this structural disagreement | Both |
| 5 | **Strong textures on distinct shapes** | Texture-biased models vs. shape-biased models organize these differently | Both |
| 6 | **Partial occlusion** | ViTs handle occlusion differently than CNNs; SAM models are specifically trained on segmentation | Both |
| 7 | **Unusual scale/crop** | Models with different input resolutions and receptive fields will disagree | Both |
| 8 | **Low-information images** | When features are scarce, models fall back on architecture-specific priors | Both |
| 9 | **High visual complexity/clutter** | Different attention mechanisms parse clutter differently | ImageNet (natural scenes) |
| 10 | **Atypical class instances** | Models that overfit to prototypical features disagree on atypical members | ImageNet |

---

## 6. Architecture-Specific Disagreement Predictions

Based on the model zoo composition, these are the pairings most likely to produce
low CKA, and the images that would exploit their differences:

### Pure CNN vs. Pure Transformer
- **Exploit:** Texture-shape conflicts, global vs. local structure
- **Example images:** Objects with strong repetitive textures but distinct global shape

### Supervised vs. Self-Supervised (DINO/MAE)
- **Exploit:** Category boundaries vs. visual clustering
- **Example images:** Visually similar objects from different categories
  (supervised separates by class; DINO clusters by visual similarity)

### CLIP-aligned vs. Non-CLIP
- **Exploit:** Semantic concept organization vs. visual feature organization
- **Example images:** Visually similar objects with very different names/concepts,
  or visually distinct objects described by the same concept

### SAM models vs. Classification models
- **Exploit:** SAM is trained for segmentation masks, classification models for labels
- **Example images:** Images where foreground/background parsing is ambiguous

### MLP-Mixers vs. Everything Else
- **Exploit:** Fixed token mixing vs. adaptive attention vs. local convolutions
- **Example images:** Images where spatial structure is critical for understanding

---

## 7. Practical Implementation Plan

```
Step 1: Download both datasets (~180 GB total)
Step 2: Select 15 diverse proxy models from the zoo
Step 3: Extract embeddings for all ~100K images × 15 models
Step 4: Compute per-image divergence scores (Method B: pairwise embedding disagreement)
Step 5: Rank all images by divergence score
Step 6: Take top 3000 candidates
Step 7: Apply diversity-aware selection to pick 1000 from top 3000
Step 8: Validate with CKA computation on the full selection
Step 9: Iterative refinement: swap low-contribution images for alternatives
Step 10: Submit
```

### Compute Requirements

- Embedding extraction: ~2-4 hours on a single GPU for 15 models × 100K images
- Per-image scoring: CPU-only, ~30 minutes
- CKA validation: ~1 hour for 1000 images × 15 model pairs
- Total: ~6-8 hours of GPU time

---

## 8. Key References

1. **Kornblith et al. (2019).** "Similarity of Neural Network Representations Revisited."
   ICML. *Original CKA paper.*

2. **Geirhos et al. (2019).** "ImageNet-trained CNNs are biased towards texture."
   ICLR. *Texture vs. shape bias.*

3. **Barbu et al. (2019).** "ObjectNet: A large-scale bias-controlled dataset."
   NeurIPS. *ObjectNet design and 40-45% accuracy drop.*

4. **Hendrycks et al. (2021).** "Natural Adversarial Examples." CVPR.
   *ImageNet-A, natural hard images.*

5. **Golan, Raju, & Kriegeskorte (2020).** "Controversial stimuli." PNAS.
   *Maximizing disagreement between models.*

6. **Raghu et al. (2021).** "Do Vision Transformers See Like CNNs?" NeurIPS.
   *ViT vs. CNN representation differences measured by CKA.*

7. **Naseer et al. (2021).** "Intriguing Properties of Vision Transformers." NeurIPS.
   *ViT robustness to occlusion, shape bias.*

8. **Idrissi et al. (2022).** "ImageNet-X: Understanding Model Mistakes." NeurIPS.
   *Factors of variation causing model failures.*

9. **Valeriani et al. (OpenReview).** "42.5% of ImageNet differentiate models."
   *Characterizing which images separate model decision boundaries.*

10. **Haas et al. (2024).** "Objective drives consistency of representational
    similarity across datasets." arXiv:2411.05561. *Training objective effect on CKA.*

11. **Schott et al. (2022).** "Robustness Limits of SoTA Vision Models."
    NeurIPS. *Architecture-specific failure modes.*

12. **AlKhamissi et al. (2024).** "Correcting CKA alignment." GitHub.
    *Debiased CKA for reliable comparison.*

13. **Kondapaneni et al. (2025).** "Representational Similarity via Interpretable
    Visual Concepts." arXiv:2503.15699. *Per-concept model disagreement.*

---

## 9. Strategic Recommendations Summary

### Top-Line Answer: Use a **data-driven empirical approach** enhanced by theory.

1. **Do NOT rely purely on heuristics.** The interaction between CKA computation
   (over sets of images) and model diversity is too complex for theory alone.

2. **ObjectNet is your primary weapon** (~60% of selections). Its controlled bias
   removal systematically exposes architecture-specific shortcut reliance.

3. **ImageNet's "difficult middle" is your secondary weapon** (~40%). Fine-grained
   ambiguity and atypical exemplars create targeted model disagreements.

4. **Diversity in the selected set matters enormously.** CKA is a set-level metric;
   redundant images dilute the signal. Ensure coverage across visual properties.

5. **Proxy-model screening is essential.** You cannot compute CKA over all 143 models
   for all 100K images, but 15 well-chosen proxies will capture the major axes of
   representational variation.

6. **The mix of CNN, Transformer, CLIP, self-supervised, and exotic architectures
   in the model zoo means that no single type of "hard image" will suffice.** You
   need images that create disagreements across MULTIPLE architectural divides
   simultaneously.

7. **Iterative refinement is cheap and high-value.** After initial selection, swap
   experiments (replace one image, recompute CKA) can yield 5-10% improvement.
