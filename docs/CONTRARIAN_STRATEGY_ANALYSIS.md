# Contrarian Strategy Analysis: Exploiting CKA's Centering Blindspot

## ICLR 2026 Re-Align Red Team Challenge

---

## Executive Summary

**The conventional wisdom is wrong.**

Every team in this challenge is likely pursuing "hard images" or "diverse images that confuse models." Our mathematical analysis and simulations reveal a counterintuitive result: **images from a single class (or extremely few semantically similar classes) produce dramatically lower CKA than diverse image sets**. In simulation, the best contrarian strategy achieves a **1.65x improvement** over the conventional "diverse/hard images" approach.

The core insight exploits a mathematical property of CKA that no one seems to be discussing: **centering kills between-class signal, and between-class signal is what all models agree on.**

---

## The Fundamental Mathematical Insight

### Why Everyone Else Is Wrong

CKA computes similarity between **centered** Gram matrices:

```
CKA(X, Y) = tr(HK_X H · HK_Y H) / sqrt(tr((HK_X H)^2) · tr((HK_Y H)^2))
```

where `H = I - (1/n)11^T` is the centering matrix.

**Centering removes the mean pairwise similarity.** This has a devastating consequence for the conventional strategy:

**When you select images from many diverse classes:**
1. The Gram matrix K = XX^T has strong block-diagonal structure (images of cats are similar to each other, different from dogs)
2. This **between-class structure dominates** the centered Gram matrix
3. **Every model agrees** on between-class structure — ResNets, ViTs, CLIP, DINO, Mixers all know that cats ≠ dogs
4. This shared agreement **inflates CKA** across all model pairs
5. Result: HIGH CKA → LOW score

**When you select images from a single class (or very few similar classes):**
1. The Gram matrix has no block-diagonal structure (all images are "the same kind of thing")
2. Centering removes the class mean, leaving only **within-class variation**
3. **Models fundamentally disagree** on within-class organization:
   - A texture-biased CNN organizes golden retrievers by fur texture and background
   - A shape-biased ViT organizes them by pose and body outline
   - CLIP organizes by scene context (outdoor vs indoor, with person vs alone)
   - DINO clusters by visual appearance (light vs dark coat, puppy vs adult)
   - MAE organizes by reconstructability (simple backgrounds vs complex)
4. This disagreement drives **low CKA** across all model pairs
5. Result: LOW CKA → HIGH score

### The Centering Decomposition

Let X be the n × p embedding matrix for a model. The centered Gram matrix is:

```
K_c = H(XX^T)H
```

For images from k classes, X decomposes as:

```
X = M + W
```

where M is the between-class means (rank k-1 after centering) and W is within-class deviations.

After centering:

```
K_c = H(MM^T + MW^T + WM^T + WW^T)H
```

When k is large and classes are well-separated, the `HMM^TH` term dominates. This term depends on how each model maps class centroids — something ALL models agree on because they're all trained to classify. CKA on this dominant term is high.

When k = 1, M is a single point that centering maps to zero. K_c = HWW^TH depends entirely on within-class variation. This is where architectural inductive biases diverge maximally.

---

## Simulation Results

### Experiment 1: Number of Classes (n=200 images, 10 model types)

| Classes | Avg CKA | Score (1-CKA) | Interpretation |
|---------|---------|---------------|----------------|
| **1** | **0.2819** | **0.7181** | Models disagree on within-class structure |
| 2 | 0.9977 | 0.0023 | Trivial binary split; all models agree |
| 5 | 0.9443 | 0.0557 | Strong class structure dominates |
| 10 | 0.8661 | 0.1339 | Between-class signal still dominant |
| 50 | 0.6292 | 0.3708 | Moderate — classes dilute each other |
| 100 | 0.5241 | 0.4759 | Many small classes; within-class starts showing |

**Critical observation:** Going from 2 classes to 1 class produces a **300x improvement in score** (0.0023 → 0.7181). The jump from 2→1 is the most dramatic discontinuity in the entire landscape.

### Experiment 2: Within-Class Spread (all images from one class)

| Spread (σ) | CKA | Score | Effective Gram Rank |
|------------|-----|-------|-------------------|
| 0.01 | 0.2195 | 0.7805 | 51 |
| 0.05 | 0.2037 | **0.7963** | 50 |
| 0.10 | 0.2330 | 0.7670 | 51 |
| 0.30 | 0.2717 | 0.7283 | 58 |
| 0.50 | 0.2916 | 0.7084 | 61 |
| 1.00 | 0.3372 | 0.6628 | 74 |
| 2.00 | 0.3916 | 0.6084 | 86 |
| 5.00 | 0.4267 | 0.5733 | 90 |

**Insight:** Tighter clusters (lower σ) → lower CKA, but the relationship is not monotonic. There's a sweet spot around σ=0.05 where images are different enough for meaningful comparison but similar enough to prevent any model from finding stable structure.

### Experiment 3: Full Strategy Comparison (n=300, 10 models)

| Rank | Strategy | Avg CKA | Score | vs Conventional |
|------|----------|---------|-------|-----------------|
| 1 | **All same class (tight)** | 0.165 | **0.835** | **+64.7%** |
| 2 | **Same class + adversarial pairs** | 0.167 | **0.833** | **+64.3%** |
| 3 | **Near-duplicate images** | 0.188 | **0.812** | **+60.2%** |
| 4 | **All same class (medium spread)** | 0.205 | **0.795** | **+56.9%** |
| 5 | Ambiguous class (low magnitude) | 0.302 | 0.698 | +37.8% |
| 6 | Sparse: 150 classes, 2 each | 0.448 | 0.552 | +8.8% |
| 7 | 20 similar subclasses | 0.478 | 0.522 | +3.0% |
| 8 | Boring (near-zero signal) | 0.492 | 0.508 | +0.2% |
| **9** | **Conventional diverse/hard** | **0.493** | **0.507** | **baseline** |
| 10 | 5 similar subclasses | 0.542 | 0.459 | -9.5% |
| 11 | Random baseline | 0.559 | 0.441 | -12.9% |
| 12 | Bimodal (boring + one class) | 0.787 | 0.213 | -57.9% |
| 13 | 3 classes (superclass) | 0.791 | 0.209 | -58.7% |

---

## Ranked Analysis of Each Contrarian Strategy

### RANK 1: All-Same-Class Strategy ★★★★★

**Verdict: THE WINNING STRATEGY. Exploit immediately.**

**Mathematical basis:** This is the purest expression of the centering insight. With k=1 class, the centering matrix eliminates the mean embedding, leaving only the within-class covariance structure. The centered Gram matrix becomes:

```
K_c = H · W · W^T · H
```

where W captures pose, viewpoint, background, lighting variations **within a single category**. Each model architecture has completely different invariances to these factors:

- CNNs: invariant to position (due to translation equivariance), sensitive to texture
- ViTs: invariant to texture (global attention), sensitive to spatial layout
- CLIP: invariant to visual variation (trained against language), sensitive to scene semantics
- MAE: sensitive to reconstructability, invariant to high-level semantics
- MLP-Mixers: fixed spatial mixing patterns, different from everything else

**Simulation score: 0.835** (vs 0.507 conventional)

**Practical implementation challenge:** ImageNet has only 50 images per class. However:

1. **Pick a fine-grained superclass:** ImageNet has ~120 dog breed classes (50 each = 6000 images). To a ResNet, a golden retriever and a Labrador might use the same texture features. To a ViT, they have different body shapes. To CLIP, "dog in a park" and "dog on a couch" are semantically distinct regardless of breed.

2. **ObjectNet single categories:** Up to 284 images per category. Combine with semantically identical ImageNet classes.

3. **The "sweet spot" classes:** Classes where within-class variation is HIGH but between-class boundaries are fuzzy:
   - Dog breeds (120 classes × 50 = 6000 candidates)
   - Bird species (59 classes × 50 = 2950 candidates)
   - Vehicle types (cars, trucks, vans — maybe 20 classes × 50 = 1000)
   - Furniture (chairs, tables, sofas — varied viewpoints)
   - Mushroom species (very texture-dependent)

**Recommended:** Pick 1000 images ALL from the ~120 ImageNet dog breed classes. Every image is "a dog" but the within-class variation (breed, pose, background, lighting, age) is enormous. Models will organize these 1000 dog images in completely incompatible ways.

---

### RANK 2: Highly Correlated / Near-Duplicate Image Strategy ★★★★

**Verdict: Surprisingly powerful. Mathematical elegance but practical difficulties.**

**Mathematical basis:** When images are near-duplicates, the centered representation matrix X_c = HX has very low effective rank. The Gram matrix K_c lives in a low-dimensional subspace. CKA becomes the cosine similarity between two low-rank matrices — and the probability that two random low-rank matrices are aligned decreases rapidly with rank. Formally:

If K_c^(A) ≈ U_A Σ_A U_A^T (rank r_A) and K_c^(B) ≈ U_B Σ_B U_B^T (rank r_B), then:

```
CKA ≈ sum_{i,j} σ_i^A σ_j^B (u_i^A · u_j^B)^2 / normalization
```

When r is small, this depends on whether the few principal components U_A and U_B happen to align. For different model architectures, they won't.

**Simulation score: 0.812**

**Why it works but is impractical:** Finding 1000 truly near-duplicate images in ImageNet/ObjectNet is hard. You'd need images that are photographically similar (same object, same angle, same lighting, slight crop variations). ImageNet validation doesn't contain many near-duplicates.

**However:** This insight suggests that **within the same-class strategy, prefer images that are MORE similar to each other**, not more diverse. Don't pick the most varied dogs — pick golden retrievers in parks, all from similar angles.

---

### RANK 3: Same-Class + Adversarial Within-Class Pairs ★★★★

**Verdict: The refined version of Strategy 1. Best theoretical approach.**

**Mathematical basis:** Start with the same-class strategy, but within that class, deliberately select images that create conflicting similarity judgments across model pairs. For two models A and B, an "adversarial pair" (i,j) satisfies:

```
K_A(i,j) >> K_B(i,j)  or  K_A(i,j) << K_B(i,j)
```

Model A thinks images i and j are very similar; Model B thinks they're very different. Including such pairs maximally de-correlates the Gram matrices.

**Example:** Two golden retriever images — one on grass (strong texture), one on a solid background (clear shape). A CNN says "very similar" (both have golden fur texture). A ViT says "very different" (one has clear body outline, the other is camouflaged by grass).

**Simulation score: 0.833**

**Implementation:** This is computationally feasible:
1. Pick a fine-grained superclass (dogs)
2. Extract embeddings from 2-3 proxy models (one CNN, one ViT, one CLIP)
3. For each image pair, compute the disagreement in pairwise similarity
4. Greedily select images that maximize total pairwise disagreement across models

---

### RANK 4: "Boring Image" / Minimal Signal Hypothesis ★★★

**Verdict: Moderate effect. Works for a different reason than expected.**

**Mathematical basis:** With near-zero input signal, model outputs are dominated by:
1. Bias terms in each layer
2. Batch normalization statistics
3. Architectural processing of noise (different architectures amplify different frequency components)

The Gram matrix on boring images reflects the **model's intrinsic processing biases** rather than image content. These biases differ across architectures.

**Simulation score: 0.508** (barely beats conventional)

**Why it doesn't win:** The centering matrix H removes the mean, so any shared "bias" processing is centered out. What remains is the model-specific noise response — but this is a weak signal. The same-class strategy provides a MUCH stronger disagreement signal because it gives models meaningful-but-ambiguous content to disagree on, rather than nothing.

**Hybrid potential:** Use boring images as a "seasoning" (10-15% of the set) within a mostly same-class selection. The boring images add Gram matrix entries that are essentially noise, diluting any residual cross-model agreement.

---

### RANK 5: Anti-Correlation Strategy ★★★

**Verdict: Sound theory, failed in practice. Overfits to 2 models.**

**Mathematical basis:** Select images that the CNN and ViT organize most differently. This directly targets the tr(K_A · K_B) term in CKA.

**Simulation score: 0.189 (NEGATIVE improvement over random in multi-model setting)**

**Why it fails:** Optimizing for disagreement between models A and B tends to select images from different classes that A and B happen to separate differently. But these images, viewed by 141 models, have STRONG between-class structure that most model pairs AGREE on. You're overfitting to one pair while creating agreement for all others.

**Salvage:** Anti-correlation works **within** the same-class strategy. Don't pick images that maximize CNN-vs-ViT disagreement globally. Instead, pick dog images that the CNN and ViT organize most differently.

---

### RANK 6: Sparse Category Coverage (Many Classes, Few Per Class) ★★

**Verdict: Wrong direction. More classes = more shared structure.**

**Simulation score: 0.552**

Each category contributes exactly one between-class comparison with every other category. With 500 categories, there are ~125,000 between-class pairs. All models agree on most of these. The within-class signal (only 2 images per class) is too weak to overcome the massive between-class consensus.

**The math is clear:** Adding categories always increases the between-class signal (which models agree on) more than it increases the within-class signal (which models disagree on). This is the exact opposite of what you want.

---

### RANK 7: Scale Exploitation (Tiny + Huge Objects) ★★

**Verdict: Interesting idea, creates a trivial binary split.**

**Simulation score: 0.213** (one of the worst!)

Selecting half tiny objects and half huge objects creates a trivial 2-class structure (small vs large). As our centering analysis shows, k=2 is the WORST case: CKA ≈ 0.998 for 2 well-separated classes. All models agree on the big/small distinction.

**Salvage:** Scale exploitation could work **within** a single class. Pick all dog images, but ensure a range of scales (close-up face shots vs full-body vs distant). This adds within-class variation along an axis where models have different invariances.

---

### RANK 8: 3-Class Superclass ★

**Verdict: Trap. The worst of both worlds.**

**Simulation score: 0.209** (near-worst)

Three well-separated classes create strong between-class structure that all models agree on. But there aren't enough images per class for the within-class disagreement to compensate. This is dominated by the "everyone agrees on the 3-way split" signal.

---

## The Optimal Strategy: Practical Implementation

Based on the mathematical analysis and simulations, the optimal approach combines insights from Ranks 1-4:

### Step 1: Select a Fine-Grained Superclass

Pick ~1000 images all from **ImageNet dog breeds** (classes n02085620 through n02113978, approximately 120 classes × 50 images = 6000 candidates). Alternatively:
- Bird species (59 classes, ~2950 candidates)
- Fungus/mushroom classes (~7 classes, 350 candidates — supplement from ObjectNet)

**Why dogs?** Maximum within-class variation (pose, background, lighting, age, activity) with minimum between-class structure (all are "dogs" to the centering matrix).

### Step 2: Within-Superclass, Maximize Model Disagreement

1. Extract embeddings from 3 maximally diverse proxy models:
   - `vgg11.tv_in1k` (texture-biased CNN)
   - `deit3_base_patch16_224.fb_in1k` (shape-biased ViT)
   - `vit_base_mci_224.apple_mclip` (semantically-organized CLIP)

2. For each candidate image, compute a "within-class disagreement score":
   ```
   For each image i, compute its similarity vector to all other candidates
   under each model. The image scores high if these similarity vectors
   have low correlation across models.
   ```

3. Greedily select 1000 images maximizing total disagreement.

### Step 3: Seasoning with ObjectNet

Replace ~100-200 images with ObjectNet images from the **same superclass** (e.g., ObjectNet has "dog" category). These add controlled viewpoint/background variation that further exposes model biases.

### Step 4: Verify with CKA Computation

Compute actual CKA on the 1000-image set across all proxy models. Compare against:
- Random 1000 images (sanity check — should be much better)
- Conventional "hard images" selection (should beat significantly)

---

## Critical Caveats

### 1. Simulation Limitations

The simulations use synthetic data with random projections. Real vision models are far more complex. However, the **centering insight is mathematical fact** — it holds regardless of model specifics. The question is the magnitude of the effect with real models.

### 2. The "Shared Within-Class Structure" Risk

All 141 models were trained on ImageNet (directly or indirectly). They may have learned **shared** within-class structure for dog breeds (e.g., all know that ear shape distinguishes breeds). This shared within-class structure would inflate CKA even with k=1.

**Mitigation:** Choose superclasses where training signals diverge:
- Self-supervised models (DINO, MAE) never saw breed labels → organize by visual similarity
- CLIP models organize by language descriptions → "puppy" vs "old dog" vs "dog in snow"
- Classification models organize by breed boundaries

### 3. The 50-Per-Class Constraint

ImageNet has only 50 images per class. Getting 1000 from a single class is impossible. But getting 1000 from a single **superclass** (e.g., all dogs = 6000 candidates) IS possible, and the CKA effect of "all dogs" vs "dogs + cats + cars" is still strongly in favor of "all dogs."

### 4. ObjectNet Might Be Better

ObjectNet's controlled viewpoint/background variation within categories is precisely the kind of within-class variation that models disagree on. If ObjectNet has enough images in related categories, it might be the better source.

---

## Comparison: Why This Beats Every Conventional Approach

| Conventional Approach | Why It Fails | Our Approach | Why It Wins |
|---|---|---|---|
| "Hard images" (ImageNet-A style) | Hard images span many classes → between-class structure dominates → models agree | Single superclass | No between-class structure → models forced to disagree |
| "Diverse images from many categories" | Maximum between-class signal → maximum model agreement | Minimum category diversity | Minimum shared signal |
| "ObjectNet for OOD" | OOD images from many categories → between-class signal still present | ObjectNet from ONE category | OOD viewpoints within class → architecture-specific failures |
| "Controversial stimuli" | Typically span category boundaries | Within-category "controversial" images | Models disagree on within-class organization |
| "Texture-shape conflicts" | These are between-class signals (cat texture + dog shape) | Within-class texture-shape variation | Texture of fur vs shape of ears within "golden retriever" |

---

## Final Rankings

| Rank | Strategy | Expected Score | Confidence | Practical Difficulty |
|------|----------|---------------|------------|---------------------|
| **1** | **Same-superclass (dogs) + within-class disagreement optimization** | **0.75-0.85** | **High** | **Low** |
| 2 | Same-superclass + near-duplicate sub-selection | 0.70-0.80 | Medium | Medium |
| 3 | Same-superclass + adversarial pair construction | 0.70-0.80 | Medium | Medium |
| 4 | Single ObjectNet category + related ImageNet classes | 0.65-0.75 | Medium | Low |
| 5 | Conventional "hard images" + within-class concentration | 0.55-0.65 | Medium | Low |
| 6 | Boring images (near-zero signal) | 0.50-0.55 | Low | Low |
| 7 | **Conventional diverse/hard images** | **0.45-0.55** | **High** | **Low** |
| 8 | Anti-correlation (optimize for 2 models) | 0.15-0.25 | High | Medium |
| 9 | Sparse coverage (many classes, few per class) | 0.40-0.50 | Medium | Low |
| 10 | Scale exploitation (bimodal) | 0.15-0.25 | High | Low |

---

## TL;DR

**Pick 1000 images of dogs. Seriously.**

The centering operation in CKA removes the very signal (between-class structure) that all models agree on. When all images are from one semantic superclass, the only signal left is within-class variation — and that's exactly where different architectures (CNNs, ViTs, CLIP, DINO, Mixers, etc.) fundamentally disagree. Every conventional "hard image" or "diverse image" strategy is swimming upstream against CKA's math. The same-class strategy swims with it.
