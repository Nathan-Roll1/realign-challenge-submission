# CKA and Embedding Dimensionality: A Deep Analysis

## ICLR 2026 Re-Align Hackathon

---

## 1. The Linear CKA Formula — How Dimensions Enter the Math

### 1.1 Setup

Given two representation matrices:
- **X** in R^{n x p1} — model 1's embeddings (n examples, p1 features)
- **Y** in R^{n x p2} — model 2's embeddings (n examples, p2 features)

Both matrices are column-centered (mean subtracted per feature).

### 1.2 The Formula (Two Equivalent Views)

**Feature-space view:**

```
Linear CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
```

Here:
- Y^T X is a **p2 x p1** matrix (cross-covariance of features)
- X^T X is a **p1 x p1** matrix (auto-covariance of X's features)
- Y^T Y is a **p2 x p2** matrix (auto-covariance of Y's features)

**Gram-matrix view (critical insight):**

Using the identity `||Y^T X||_F^2 = tr(XX^T YY^T)`:

```
Linear CKA(X, Y) = tr(K_X * K_Y) / sqrt(tr(K_X^2) * tr(K_Y^2))
```

where `K_X = XX^T` and `K_Y = YY^T` are both **n x n** Gram matrices, regardless of p1 and p2.

### 1.3 Key Mathematical Observation

**The Gram matrix formulation makes CKA's "dimension-free" nature apparent.** Both K_X and K_Y are n x n matrices. The feature dimensions p1 and p2 are "absorbed" into the Gram matrices. CKA compares **how the n examples relate to each other** in each representation space, not the features directly.

This is why CKA was introduced as an improvement over CCA — Kornblith et al. (ICML 2019) proved that CCA and any statistic invariant to invertible linear transformation "cannot measure meaningful similarities between representations of higher dimension than the number of data points." CKA avoids this by only requiring orthogonal invariance.

**In theory, p1 != p2 does NOT inherently lower CKA.** Two models with dimensions 512 and 2048 can achieve CKA = 1.0 if they impose identical similarity structures on the n examples.

---

## 2. But In Practice: The Bias Problem

### 2.1 The Biased HSIC Estimator

The standard ("biased") empirical HSIC estimator is:

```
HSIC_b(K, L) = (1/(n-1)^2) * tr(KHLH)
```

where H = I_n - (1/n) * 11^T is the centering matrix.

This estimator has a **non-zero expectation even when X and Y are independent random matrices**. The bias depends on the dimensionality p and the number of samples n.

### 2.2 How p/n Ratio Creates Artificial Similarity

Murphy et al. (2024, arXiv:2405.01012) demonstrated the core problem:

For random Gaussian matrices X in R^{n x p1} and Y in R^{n x p2}:

1. **HSIC(K_X, K_X)** has expected value that **grows with p1**. Intuitively: with more random features, the Gram matrix K_X = XX^T has more "structure" simply due to the concentration of random projections.

2. **HSIC(K_X, K_Y)** for independent X, Y also has non-zero expectation that depends on **both p1 and p2**.

3. The **CKA ratio** for independent random matrices approaches values that depend on `p1/n` and `p2/n`, NOT on any true alignment.

**The critical consequence:** When p1/n and p2/n are both large, biased CKA will report high similarity even between completely independent random representations. When p1/n != p2/n, the bias is asymmetric, making comparisons across different dimensionalities unreliable.

### 2.3 Concrete Example of the Bias

From Murphy et al.'s experiments:
- Two independent random matrices with the **same** p/n ratio produce moderate CKA (~0.1-0.3 depending on exact ratio)
- Two independent random matrices with **different** p/n ratios can produce CKA values that are **artificially inflated or deflated**
- Biased CKA can be driven to its **maximum value** with independent random data of sufficiently different p/n ratios

### 2.4 What This Means for the Challenge

The challenge uses standard (biased) CKA (this is the standard Kornblith et al. formulation). This means:

| Scenario | Effect on CKA |
|----------|--------------|
| Two models with same p and same data | Reliable comparison |
| Two models with different p but n >> max(p1,p2) | Bias is small, comparison is reasonable |
| Two models with different p and n ~ p | **Bias becomes significant**, CKA inflated for higher-p models |
| Two models with different p and n < max(p1,p2) | **CKA is unreliable**, dominated by p/n ratio effects |

---

## 3. What "embedding": "flatten" Means

### 3.1 The Flatten Operation

Every model in the registry has `"embedding": "flatten"`. This means the raw output tensor from the specified layer is **reshaped to a 1D vector per example** before CKA is computed.

### 3.2 Layer Types and Their Output Shapes

The registry specifies different extraction layers, which produce different output shapes:

| Layer Type | Example Models | Raw Output Shape | After Flatten (= p) |
|-----------|---------------|-----------------|---------------------|
| `global_pool` / `head.global_pool` | ResNets, EfficientNets, ConvNeXts, Swin, etc. | (batch, channels) | channels (e.g., 2048, 1024, 512, 768) |
| `fc_norm` | BEiT, DeiT, EVA, FlexiViT, AIM | (batch, hidden_dim) | hidden_dim (e.g., 768, 1536) |
| `norm` | CaiT, ConViT, gMLP, Mixer, PiT, MViT, ResMLP, Twins, TNT, VOLO, XCiT | (batch, seq_len, hidden_dim) **or** (batch, hidden_dim) | **Potentially seq_len * hidden_dim** |
| `head.norm` | Hiera, MambaOut | (batch, hidden_dim) | hidden_dim |
| `head.bn` | LeViT | (batch, channels) | channels |
| `stages.3.norm` / `norm4` / `norm.1` | PVT, CoAT, CrossViT | (batch, tokens, dim) | **Potentially tokens * dim** |

### 3.3 The Critical Dimensionality Ranges

**Post-pooling models** (most CNN-based): p is in the range **256 to 4096**, typically:
- ResNet-18/34: 512
- ResNet-50/101: 2048
- EfficientNet-B0: 1280
- MobileNets: 960-1280
- Swin-B/ConvNeXt-B: 1024
- VGG: 512 (after conv features global pool)

**ViT-based models with fc_norm** (post-pooling): p is in the range **768 to 1536**, typically:
- ViT-Base: 768
- BEiT-Base: 768
- AIMv2-1B: 1536 (very large model)

**ViT-based models with `norm` (potentially pre-pooling)**: p could be **much larger**:
- If `norm` captures the full sequence before pooling: ViT-Base at 224 with patch16 = 197 tokens x 768 = **151,296** dimensions
- If `norm` captures only the CLS token: 768 dimensions
- This depends on the exact timm implementation and how the challenge organizers extract features

**This is the single most important unknown for dimensionality analysis.** If some models produce flattened embeddings of 150K+ dimensions while others produce 512-2048, the p/n ratio differences would be enormous.

---

## 4. Answering Your Key Questions

### Q1: Does different embedding dimension inherently lower CKA?

**Mathematically, no.** CKA operates on n x n Gram matrices, so p1 != p2 is handled naturally. If two models impose the same similarity structure on examples, CKA = 1.0 regardless of whether p1 = 512 and p2 = 2048.

**Practically, it depends on n.** If the number of evaluation examples n is much larger than both p1 and p2, the bias is negligible and CKA is reliable. If n is small relative to the larger p, the biased CKA estimator produces artificially high or distorted scores that depend on p/n ratios rather than true alignment.

**For the challenge specifically:** Since the evaluation uses a "fixed set of held-out images" with standard biased CKA, the p/n ratio matters. If n = 1000 and some models have p = 2048 (p/n = 2.0) while others have p = 768 (p/n = 0.77), the bias will differ across pairs.

### Q2: How does dimensionality affect the formula components?

Breaking down `||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)`:

**Numerator: ||Y^T X||_F^2**
- Y^T X is p2 x p1. Its Frobenius norm squared = sum of squared entries = sum of all (y_i^T x_j)^2 for features i in Y and j in X.
- This is the total "cross-correlation energy" between X and Y features.
- More features in either X or Y means more terms in this sum.
- For truly aligned representations, additional features contribute positive signal.
- For unaligned representations, additional features contribute noise that scales with p1 * p2 / n.

**Denominator: ||X^T X||_F * ||Y^T Y||_F**
- ||X^T X||_F = sqrt(sum of squared auto-correlations among X's features). Scales roughly as p1.
- ||Y^T Y||_F = sqrt(sum of squared auto-correlations among Y's features). Scales roughly as p2.
- Product scales as p1 * p2.

**The ratio:** For random data, both numerator and denominator scale as ~p1 * p2, but the exact scaling depends on n, creating the bias. For aligned data, the numerator captures true alignment signal that the denominator normalizes correctly.

### Q3: Does CKA "flatten" embeddings before comparison?

**Yes.** The `"embedding": "flatten"` in the registry means the raw layer output is flattened to a 1D vector per image. The CKA computation then receives X in R^{n x p} where p is the total flattened dimension.

**Important nuance:** Flattening a (batch, seq_len, hidden_dim) tensor to (batch, seq_len * hidden_dim) preserves spatial/positional information in the feature dimension. This is different from global average pooling, which discards spatial info. Models that are flattened pre-pooling carry spatial structure in their features, while post-pooling models don't.

This means CKA for pre-pooling models is partially comparing **spatial layouts**, not just feature semantics. Two models that extract similar features but arrange tokens differently would have lower CKA than two models with global-pooled features.

### Q4: Blue Team — should you prefer models with similar embedding dimensions?

**Yes, and your strategy document already identifies this as a risk factor.** Here's the full reasoning:

1. **Bias alignment:** If the challenge uses biased CKA (almost certain), then models with similar p values will have similar p/n ratios, making their CKA scores mutually comparable and avoiding artificial inflation/deflation from the bias.

2. **All ResNet variants use global_pool → similar p:** ResNet-50/101/152 all have p = 2048 after global average pooling. SE-ResNet, ResNeXt, Res2Net, ResNeSt all preserve this (they modify the residual block but keep the final channel count at 2048). This is a **hidden advantage** of the ResNet-clan strategy — you're implicitly controlling for dimensionality.

3. **Cross-family comparisons are worse for two reasons:** Not only do different architectures have different representations (lowering true CKA), they also have different embedding dimensions (adding bias artifacts). ViT-base (768) vs. ResNet-101 (2048) has both problems.

**Recommendation refinement:** Within your Tier 1-3 ResNet selections, verify that all models produce the same embedding dimension after global pooling. Models like `skresnet18` (512-dim, being an 18-layer variant) would have a different dimension than `resnet101` (2048-dim). Consider prioritizing dimensional homogeneity.

Likely dimensions within your selections:
- ResNet-50+ variants: **2048** (resnet101, resnest101e, res2net101, ecaresnet101d, seresnet152d, seresnext101, resnetaa101d, wide_resnet101, cspresnet50, cspresnext50)
- ResNet-26/33 variants: **1024 or 2048** (gcresnet33ts, gcresnext26ts, botnet26t, halonet26t, etc.)
- ResNet-18 variants: **512** (skresnet18 — this is an outlier!)

### Q5: Red Team — can you exploit dimensionality mismatches?

**The Red Team doesn't choose models — they choose images.** But dimensionality matters for the Red Team strategy in a subtler way:

1. **The p/n ratio depends on n (number of your submitted images + held-out images).** Wait — actually, for the Red Team, CKA is computed on **your 1000 submitted images** (the stimulus set). So n = 1000 is fixed by your submission size.

2. **With n = 1000 and models ranging from p = 512 to potentially p = 150,000+:** The p/n ratios range from 0.5 to 150+. This means CKA is in vastly different bias regimes for different model pairs.

3. **You CAN'T directly exploit this** because you don't control which models are used. But you can **select images that amplify model-specific feature activation patterns**, which interact with dimensionality:
   - High-dimensional (pre-pooling) models carry spatial information → images with unusual spatial layouts cause more divergence in these models
   - Low-dimensional (post-pooling) models compress to channel statistics → images with unusual semantic content cause more divergence

4. **For the Red Team scoring (1 - avg CKA):** Since biased CKA on the full 143-model zoo with n = 1000 will naturally produce varying p/n ratios across model pairs, the baseline CKA will already include some bias-driven inflation. Your job is to select images that push the **true alignment** as low as possible; the bias effects are a constant backdrop across all submissions.

---

## 5. The n/p Ratio: Reliability Analysis

### 5.1 Regimes of CKA Reliability

| Regime | n/p Ratio | CKA Behavior |
|--------|-----------|-------------|
| **High samples** | n >> p (e.g., n/p > 10) | CKA is reliable. Bias is negligible. Dimensionality has no meaningful effect. |
| **Moderate samples** | n ~ p (e.g., n/p = 1-5) | CKA works but bias is noticeable. Comparisons across different p values are slightly distorted. |
| **Low samples** | n < p (e.g., n/p < 1) | CKA is unreliable. Dominated by p/n ratio effects. Biased estimator gives artificially high values. |
| **Extreme** | n << p (e.g., n/p < 0.1) | CKA is meaningless. Random matrices of same dimensions give high CKA. |

### 5.2 Estimating n for This Challenge

The challenge says CKA is "computed on embeddings extracted from a fixed set of held-out images." The number n is unknown, but reasonable estimates:

- **Lower bound:** ~500 (enough for statistical stability)
- **Likely range:** 1000-5000 (common in representation analysis benchmarks)
- **Upper bound:** ~10,000 (computational cost of CKA grows as n^2 for Gram matrix computation, times 143^2/2 ≈ 10,000 model pairs)

### 5.3 Implications for Different n Values

**If n = 1000:**
| Model Type | Typical p | p/n | Reliability |
|-----------|-----------|-----|-------------|
| Post-pooling CNN (ResNet-101) | 2048 | 2.05 | Moderate — some bias |
| Post-pooling CNN (ResNet-18) | 512 | 0.51 | Good |
| Post-pooling ViT (BEiT-base) | 768 | 0.77 | Good |
| Pre-pooling ViT (norm layer) | ~150,000 | 150 | **Very unreliable** |
| Post-pooling large model (AIMv2-1B) | 1536 | 1.54 | Moderate |

**If n = 5000:**
| Model Type | Typical p | p/n | Reliability |
|-----------|-----------|-----|-------------|
| Post-pooling CNN (ResNet-101) | 2048 | 0.41 | Good |
| Pre-pooling ViT (norm layer) | ~150,000 | 30 | **Unreliable** |
| Everything else | <2048 | <0.41 | Good |

### 5.4 The Pre-Pooling Problem

If some models (those with `norm`, `stages.3.norm`, `norm.1`, `norm4` layers) truly produce flattened pre-pooling outputs (seq_len * hidden_dim >> n), then:

1. Their self-similarity HSIC(K, K) will be inflated
2. Their CKA with post-pooling models will be distorted by dimension mismatch
3. Their CKA with other pre-pooling models of similar dimension will be artificially high

**However:** The challenge organizers likely accounted for this. The `"embedding": "flatten"` might be applied after the model's own pooling mechanism, not literally to the raw layer output. This is the most reasonable interpretation given that the challenge is designed to produce meaningful CKA comparisons.

---

## 6. Strategic Implications

### 6.1 Blue Team — Updated Recommendations

Your existing ResNet-clan strategy is **well-aligned with dimensionality considerations**:

1. **Most ResNet-50+ variants produce 2048-dim embeddings** after global average pooling → homogeneous p/n ratios → unbiased CKA comparisons within the set.

2. **Watch out for outliers:**
   - `skresnet18` → likely 512-dim (ResNet-18 backbone) — **consider replacing** with a ResNet-50+ variant
   - Verify `botnet26t`, `halonet26t`, `sebotnet33ts`, `sehalonet33ts` dimensions — the `26t` and `33ts` suffixes suggest smaller backbones that might have 1024 or 512 channels

3. **Dimensional homogeneity is a secondary optimization** on top of your primary strategy (architecture family). Within the ResNet clan, prefer models with the same output dimension.

4. **The "CKA bias works in our favor" insight from your strategy doc (Section 1.9) is correct IF all models have similar p.** If you introduce a 512-dim model into a set of 2048-dim models, the bias effect is asymmetric and could lower your mean CKA.

### 6.2 Red Team — Updated Considerations

1. **Your stimulus selection directly controls n.** With n = 1000, the p/n ratios for the model zoo range from ~0.25 (small MobileNets) to ~2.0 (ResNets) to potentially >>1 (pre-pooling ViTs).

2. **The bias creates a "floor" for CKA** — even with maximally divergent images, biased CKA for high-p model pairs won't go to zero. This is a limitation you can't overcome through image selection.

3. **Focus on images that create divergent SIMILARITY STRUCTURES** (not just divergent individual embeddings). CKA measures whether models agree on which images are similar to each other. Your 1000 images should create situations where different architectures impose fundamentally different orderings on the set.

### 6.3 Summary Table

| Factor | Effect on CKA | Blue Team Impact | Red Team Impact |
|--------|--------------|-----------------|----------------|
| p1 = p2 (same dim) | No bias from dimension mismatch | Pick models with same p | N/A |
| p1 >> p2 (mismatched) | Biased CKA distorted | Avoid mixing dims | Can't control models |
| n >> p (many examples) | CKA reliable | Evaluation n is fixed | Your n = 1000 |
| n < p (few examples) | CKA inflated/unreliable | Models with high p are riskier | Some model pairs are inherently biased |
| "flatten" pre-pooling | Creates very high p | Avoid such models | These models add noise to CKA |
| "flatten" post-pooling | Normal p (256-2048) | Standard regime | Standard regime |

---

## 7. Key References

1. **Kornblith, Norouzi, Lee & Hinton (ICML 2019)** — "Similarity of Neural Network Representations Revisited." Original CKA paper. Proves limitations of CCA for high-dim, introduces CKA.

2. **Murphy, Zylberberg & Fyshe (2024, arXiv:2405.01012)** — "Correcting Biased CKA Measures." Demonstrates p/n ratio bias, introduces debiased CKA.

3. **Davari, Horoi, Natik, Lajoie, Wolf & Belilovsky (ICLR 2023)** — "Reliability of CKA as a Similarity Measure in Deep Learning." Shows CKA sensitivity to outliers and linear transformations.

4. **Nguyen, Raghu & Kornblith (ICLR 2021)** — "Do Wide and Deep Networks Learn the Same Things?" Width/depth effects on CKA.

---

## 8. Bottom Line

**For the Re-Align Challenge, embedding dimensionality is a second-order effect that your ResNet-clan strategy already largely handles.** The most important factors for CKA are:

1. **Architecture family** (first-order effect — dominates CKA)
2. **Training objective** (first-order for self-supervised models)
3. **Embedding dimensionality** (second-order — matters through bias when p ~ n)
4. **Preprocessing** (second-order — affects input distribution)

The key practical takeaway: **ensure dimensional homogeneity within your Blue Team selection**, and be aware that the pre-pooling ViT models in the registry (those with `norm` layer extraction) may have anomalous CKA behavior due to very high embedding dimensions. If you're unsure about a model's dimension, run a quick test: load it in timm, pass a dummy input, and check the output shape at the specified layer after flattening.
