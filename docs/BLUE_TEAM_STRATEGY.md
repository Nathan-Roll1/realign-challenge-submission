# Blue Team Strategy: Maximizing Mean Pairwise CKA

## ICLR 2026 Re-Align Hackathon

**Objective:** Select 20 models from the ~148 model registry to maximize mean pairwise CKA (Centered Kernel Alignment) computed on held-out image embeddings.

---

## 1. Literature Review: What Causes Representations to Converge?

### 1.1 The Platonic Representation Hypothesis

Huh, Cheung, Wang & Isola (ICML 2024, arXiv:2405.07987) argue that representations across AI models are **converging toward a shared statistical model of reality**. Key findings:

- Over time, the ways different neural networks represent data are becoming more aligned
- As models scale larger, vision and language models measure distances between datapoints in increasingly similar ways
- This convergence is driven by **scale** (model size), **data diversity**, and **task diversity**

**Implication for us:** Larger, more capable models may converge more, but the pool mostly contains one representative per family, not scale variants.

### 1.2 Architecture Family is the PRIMARY Determinant of CKA Similarity

**Sirikova & Chan (2025, arXiv:2601.17093) — "The Triangle of Similarity"** is the most directly relevant paper. Their framework combines CKA, functional similarity, and sparsity similarity across CNNs, ViTs, and VLMs. Key finding:

> "Architectural family is a **primary determinant** of representational similarity, forming **distinct clusters**."

Models within the same architecture family (e.g., ResNets with ResNets, ViTs with ViTs) have **much higher CKA** than cross-family pairs, even controlling for model size and pretraining dataset.

### 1.3 Training Objective Matters — Sometimes MORE Than Architecture

**Wu, Saha, Bo & Khosla (2025, arXiv:2509.21628) — "A Data-driven Typology of Vision Models"** provides crucial nuance:

- **Metrics preserving geometry (like CKA, RSA)** yield **strong family discrimination** — geometry carries family-specific signatures
- **Supervised ResNets and supervised ViTs form distinct clusters**
- But: **All self-supervised models group together across architecture boundaries** (DINO ResNet clusters with DINO ViT, not with supervised ResNet)
- **Hybrid architectures (ConvNeXt, Swin) cluster with masked autoencoders**, suggesting convergence between architectural modernization and reconstruction-based training

**Implication:** For supervised models, architecture family dominates. For self-supervised models, training objective dominates. Since most models in our pool are supervised, architecture family is the key lever.

### 1.4 Training Objective Drives Cross-Dataset Consistency

**Muttenthaler et al. (2024, OpenReview)** — "Objective drives the consistency of representational similarity across datasets":

- Self-supervised vision models show **more consistent** pairwise representational similarities across datasets
- Supervised image classification models show less consistency
- The training objective is the crucial factor for whether representational similarities generalize

### 1.5 Same Architecture + Different Seeds → Similar (but not identical) Representations

**Kornblith et al. (ICML 2019)** — The foundational CKA paper:

- Networks with **identical architectures** trained from different random initializations learn **similar representations** detectable by CKA
- CKA reliably identifies **layer correspondences** between architecturally similar networks

**Nguyen et al. (Nature Communications 2020)** found "individual differences" persist even with same architecture, arising from under-constrained alignment of category exemplars.

### 1.6 Model Scale: Larger Models Show "Block Structure"

**Nguyen, Raghu & Kornblith (ICLR 2021)** — "Do Wide and Deep Networks Learn the Same Things?":

- Smaller models show higher inter-model similarity
- Very wide or very deep models develop a characteristic **"block structure"** in representations that makes them **dissimilar** to other models
- This block structure is unique to each model's specific width/depth configuration
- **Representations outside the block structure remain similar across architectures**

**Implication:** Don't blindly pick the largest models. Models of **similar, moderate capacity** may have higher pairwise CKA than mixing huge and small models.

### 1.7 Model Stitching: Same Architecture = Functionally Interchangeable

**Bansal, Nakkiran & Barak (NeurIPS 2021)** — Model stitching experiments show:

- Networks of the **same architecture but trained differently** (supervised vs self-supervised) can be stitched together without performance drop
- This confirms same-architecture models learn **functionally compatible** representations

### 1.8 Preprocessing Parameters Affect CKA

CKA is computed on embeddings. Different preprocessing (crop size, normalization) changes the input distribution, which changes activations, which changes CKA. From the registry:

- **Most models use ImageNet normalization:** mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- **Some use [0.5, 0.5, 0.5] normalization** (ViTs, Xception, some others)
- **Some use [0.0, 0.0, 0.0] / [1.0, 1.0, 1.0]** (TResNet, MobileViT, SAM ViT)
- **Crop sizes vary:** 160 to 1024

**Implication:** Models with the same preprocessing will have more comparable activations. However, the hackathon extracts embeddings using each model's own preprocessing, so the CKA comparison happens on the embedding level post-preprocessing. Still, models trained with similar preprocessing learned from similar input distributions.

### 1.9 CKA Biases and Deconfounding

**Alex Murphy et al. (2024, arXiv:2405.01012)** — Biased CKA produces artificially high similarity scores:

- CKA is sensitive to feature-sample ratios
- Debiased CKA corrects for this

**Nguyen et al. (2022, arXiv:2202.00095)** — CKA is confounded by input population structure, leading to spuriously high similarity even between random networks.

**Implication:** The hackathon uses standard CKA, so these biases work in our favor if we pick models with similar embedding dimensionality and structure.

---

## 2. Strategic Analysis of the Model Pool

The registry contains ~148 models, each representing a distinct "family" from timm. All are trained on ImageNet (or fine-tuned from larger datasets to ImageNet).

### 2.1 Key Model Clusters in the Registry

#### Cluster A: ResNet Extended Family (STRONGEST CANDIDATE — ~20 models available)

These all share the fundamental **bottleneck residual block** with minor variations:

| Model | Variation | Preprocessing | Notes |
|-------|-----------|---------------|-------|
| `resnet101.a1_in1k` | Standard ResNet | ImageNet norm, crop 224 | Baseline |
| `resnetaa101d.sw_in12k` | Anti-aliased ResNet | ImageNet norm, crop 224 | Pretrained on larger data |
| `resnetblur50.bt_in1k` | Blur-pooled ResNet | ImageNet norm, crop 224 | Anti-aliasing variant |
| `resnetrs101.tf_in1k` | ResNet-RS | ImageNet norm, crop 192 | Revised training |
| `resnetv2_101.a1h_in1k` | ResNet v2 | **[0.5] norm**, crop 224 | Pre-activation variant |
| `resnext101_32x16d.fb_ssl_yfcc100m_ft_in1k` | ResNeXt | ImageNet norm, crop 224 | **SSL pretrained** |
| `res2net101_26w_4s.in1k` | Res2Net | ImageNet norm, crop 224 | Multi-scale residual |
| `resnest101e.in1k` | ResNeSt | ImageNet norm, crop 256 | Split attention |
| `ecaresnet101d.miil_in1k` | ECA-ResNet | ImageNet norm, crop 224 | Channel attention |
| `ecaresnetlight.miil_in1k` | ECA-ResNet Light | ImageNet norm, crop 224 | Lightweight ECA |
| `seresnet152d.ra2_in1k` | SE-ResNet | ImageNet norm, crop 256 | Squeeze-excitation |
| `seresnext101_32x4d.gluon_in1k` | SE-ResNeXt | ImageNet norm, crop 224 | SE + grouped conv |
| `seresnextaa101d_32x8d.ah_in1k` | SE-ResNeXt-AA | ImageNet norm, crop 224 | SE + anti-alias |
| `skresnet18.ra_in1k` | SK-ResNet | ImageNet norm, crop 224 | Selective kernel |
| `skresnext50_32x4d.ra_in1k` | SK-ResNeXt | ImageNet norm, crop 224 | SK + grouped conv |
| `wide_resnet101_2.tv2_in1k` | Wide ResNet | ImageNet norm, **crop 176** | Width multiplier |
| `cspresnet50.ra_in1k` | CSP-ResNet | ImageNet norm, crop 256 | Cross-stage partial |
| `cspresnext50.ra_in1k` | CSP-ResNeXt | ImageNet norm, crop 256 | CSP + grouped conv |
| `gcresnet33ts.ra2_in1k` | GC-ResNet | ImageNet norm, crop 256 | Global context |
| `gcresnext26ts.ch_in1k` | GC-ResNeXt | ImageNet norm, crop 256 | GC + grouped conv |
| `lambda_resnet26rpt_256.c1_in1k` | Lambda-ResNet | ImageNet norm, crop 256 | Lambda layers |

**Why this cluster:** All share bottleneck residual architecture DNA. All supervised on ImageNet. Most use identical preprocessing. The variations (SE, ECA, SK, GC, CSP) are minor attention or connection modifications that preserve the fundamental residual computation pattern.

#### Cluster B: SE-Net / ResNet-Attention Hybrids

| Model | Notes |
|-------|-------|
| `legacy_senet154.in1k` | Original SE networks |
| `senet154.gluon_in1k` | SE networks (gluon weights) |

These are architecturally very close to SE-ResNets in Cluster A.

#### Cluster C: DarkNet Family

| Model | Notes |
|-------|-------|
| `darknet53.c2ns_in1k` | YOLO backbone |
| `darknetaa53.c2ns_in1k` | Anti-aliased DarkNet |
| `cspdarknet53.ra_in1k` | CSP-DarkNet |
| `cs3darknet_focus_l.c2ns_in1k` | CS3-DarkNet |

Closely related to ResNets (residual connections, convolutional blocks).

#### Cluster D: Supervised Standard-Preprocessing CNNs (broader)

Additional models with ImageNet norm + supervised training that are architecturally "ResNet-adjacent":

| Model | Relation to ResNet |
|-------|-------------------|
| `densenet121.ra_in1k` | Dense residual connections |
| `densenetblur121d.ra_in1k` | Dense + anti-aliasing |
| `dla102.in1k` | Deep layer aggregation (residual-based) |
| `dpn107.mx_in1k` | Dual-path (ResNet + DenseNet hybrid) |
| `selecsls42b.in1k` | Selective long/short skip |
| `repvgg_a0.rvgg_in1k` | Re-parameterized VGG (becomes ResNet-like) |

#### Cluster E: BotNet/HaloNet Hybrids

| Model | Notes |
|-------|-------|
| `botnet26t_256.c1_in1k` | ResNet + bottleneck attention |
| `eca_botnext26ts_256.c1_in1k` | ECA + BotNeXt |
| `halo2botnet50ts_256.a1h_in1k` | Halo + BotNet |
| `halonet26t.a1h_in1k` | Halo attention |
| `sebotnet33ts_256.a1h_in1k` | SE + BotNet |
| `sehalonet33ts.ra2_in1k` | SE + HaloNet |
| `lamhalobotnet50ts_256.a1h_in1k` | Lambda + Halo + BotNet |
| `bat_resnext26ts.ch_in1k` | Bottleneck attention + ResNeXt |

These are ResNet bodies with attention heads — structurally very close to Cluster A.

---

## 3. Strategic Recommendation

### 3.1 The Key Answer: SAME Family, Not Cross-Family

**The literature is unambiguous: to maximize CKA, pick models from the SAME architectural family.**

The research evidence is:
1. Architecture family forms **distinct CKA clusters** (Sirikova & Chan, 2025)
2. Same-architecture models show **highest CKA correspondence** (Kornblith et al., 2019)
3. Same-architecture models can be **stitched without performance drop** (Bansal et al., 2021)
4. Cross-architecture CKA is systematically lower even with same training data/objective

The "Platonic Representation Hypothesis" convergence applies at the frontier of scale — very large models from different families converge. But in our pool of moderate-sized ImageNet classifiers, **architecture dominates**.

### 3.2 Recommended 20-Model Selection: "The ResNet Clan"

Pick the 20 most closely related ResNet variants, prioritizing:
1. **Same fundamental architecture** (bottleneck residual blocks)
2. **Same training objective** (supervised ImageNet classification)
3. **Same preprocessing normalization** (ImageNet mean/std)
4. **Similar crop/resolution** (224-256 range)

#### Tier 1: Core ResNet Variants (highest expected pairwise CKA)

These are the most architecturally similar — they literally are ResNet with minor tweaks:

1. `resnet101.a1_in1k` — Standard ResNet-101
2. `resnetaa101d.sw_in12k` — Anti-aliased ResNet (minor pooling change)
3. `resnetblur50.bt_in1k` — Blur-pooled ResNet (minor pooling change)
4. `res2net101_26w_4s.in1k` — Multi-scale residual (minor block change)
5. `resnest101e.in1k` — Split-attention ResNet (minor attention change)
6. `ecaresnet101d.miil_in1k` — ECA channel attention ResNet
7. `ecaresnetlight.miil_in1k` — Lightweight ECA-ResNet
8. `seresnet152d.ra2_in1k` — SE-ResNet (channel attention)
9. `seresnext101_32x4d.gluon_in1k` — SE-ResNeXt (grouped convolutions)
10. `seresnextaa101d_32x8d.ah_in1k` — SE-ResNeXt with anti-aliasing

#### Tier 2: ResNet Extended Family (very high expected CKA)

11. `skresnext50_32x4d.ra_in1k` — Selective kernel + ResNeXt
12. `skresnet18.ra_in1k` — Selective kernel ResNet
13. `cspresnet50.ra_in1k` — Cross-stage partial ResNet
14. `cspresnext50.ra_in1k` — Cross-stage partial ResNeXt
15. `gcresnet33ts.ra2_in1k` — Global context ResNet
16. `gcresnext26ts.ch_in1k` — Global context ResNeXt

#### Tier 3: ResNet-Bodied Hybrids (high expected CKA)

These use ResNet bodies with attention in the final stages:

17. `botnet26t_256.c1_in1k` — ResNet + bottleneck self-attention
18. `sebotnet33ts_256.a1h_in1k` — SE + BotNet (ResNet body)
19. `sehalonet33ts.ra2_in1k` — SE + Halo attention (ResNet body)
20. `halonet26t.a1h_in1k` — Halo attention (ResNet body)

#### Alternative Tier 3 Swaps

If any Tier 3 model underperforms, swap with these candidates:
- `lambda_resnet26rpt_256.c1_in1k` — Lambda ResNet
- `eca_botnext26ts_256.c1_in1k` — ECA + BotNeXt
- `halo2botnet50ts_256.a1h_in1k` — Halo + BotNet
- `bat_resnext26ts.ch_in1k` — Bottleneck attention + ResNeXt
- `senet154.gluon_in1k` — Classic SE-Net
- `legacy_senet154.in1k` — Legacy SE-Net

### 3.3 Why NOT Cross-Architecture?

Picking 20 models from different families (ViTs, MLP-Mixers, ConvNeXt, ResNets) would be a losing strategy because:

1. **CKA between ViT and ResNet is systematically low** — they develop fundamentally different internal representations despite similar accuracy (Raghu et al., NeurIPS 2021)
2. **MLP-Mixers have unique representation geometry** — they are outliers in CKA space
3. **ViTs show "block structure" at scale** making large ViTs dissimilar even to each other (Nguyen et al., ICLR 2021)
4. The mean pairwise CKA of a mixed set is dragged down by every low cross-family pair

### 3.4 Why NOT All ViT Variants?

The ViT cluster in our pool (BEiT, DeiT, EVA, FlexiViT, etc.) seems tempting, but:

1. ViTs in the registry use **diverse training objectives** (supervised, MIM, CLIP, MAE) — and training objective splits ViTs into sub-clusters
2. Several ViTs have **different preprocessing** ([0.5] vs ImageNet norm vs [0.0])
3. ViTs develop more **uniform layer representations**, but cross-ViT-variant CKA can be lower than cross-ResNet-variant CKA because the architectural variations are more dramatic (patch size, attention patterns, class token vs global pool)

### 3.5 Preprocessing Considerations

Within our recommended 20, note:
- **resnetv2_101** uses [0.5, 0.5, 0.5] normalization — consider excluding it
- **resnext101_32x16d.fb_ssl_yfcc100m_ft_in1k** was SSL pretrained on YFCC100M — per Wu et al. (2025), SSL models cluster separately from supervised models. Consider excluding.
- **wide_resnet101_2.tv2_in1k** uses crop size 176 (outlier) — consider excluding
- **resnetrs101.tf_in1k** uses crop size 192 (outlier) — lower priority

The recommended 20 above already accounts for these concerns by preferring models with standard ImageNet preprocessing.

---

## 4. Experimental Validation Plan

Before submitting, verify the strategy empirically:

1. **Extract embeddings** from all 20 candidate models on a small image set (e.g., 500 ImageNet val images)
2. **Compute pairwise CKA matrix** (20x20 = 190 pairs)
3. **Compute mean pairwise CKA** as the score
4. **Compare against alternative strategies:**
   - A: 20 random models
   - B: 20 ViT-family models
   - C: Top-20 by ImageNet accuracy
   - D: Our ResNet-clan selection
5. **Iteratively swap** the lowest-CKA model out and try alternatives

### Key metrics to track:
- Mean pairwise CKA (submission score)
- Min pairwise CKA (identifies the weakest link)
- Which model pairs have lowest CKA (candidates for removal)

---

## 5. Risk Assessment

| Risk | Mitigation |
|------|------------|
| CKA is computed on held-out images we don't see | ResNet family CKA should be robust across datasets (same architecture = consistent representations) |
| The evaluator uses debiased CKA | Debiased CKA still shows architecture-family clustering; our strategy holds |
| ResNet variants have lower CKA than expected | Empirically validate before submitting; have ViT-family backup |
| Embedding dimensionality mismatch inflates/deflates CKA | The registry specifies per-model embedding layers; check dims match |
| Crop size differences cause divergence | We prioritized models with 224-256 crop sizes |

---

## 6. Summary

### One-line strategy:
**Pick 20 supervised ResNet variants with standard ImageNet preprocessing to maximize within-family CKA.**

### Rationale in three bullets:
1. Architecture family is the single strongest predictor of CKA similarity (Sirikova & Chan, 2025)
2. The ResNet extended family has the most members (~20+) in the model pool, all sharing bottleneck residual block DNA
3. Supervised training + ImageNet normalization is consistent across these models, avoiding the training-objective split that fragments ViTs

### Key citations:
- Kornblith et al. (ICML 2019) — CKA metric; same-architecture convergence
- Huh et al. (ICML 2024) — Platonic representation hypothesis; scale-driven convergence
- Sirikova & Chan (2025) — Architecture family = primary CKA cluster determinant
- Wu et al. (2025) — Training objective drives cross-architecture clustering for SSL; supervised clusters by architecture
- Nguyen, Raghu & Kornblith (ICLR 2021) — Width/depth block structure; moderate-size models most similar
- Bansal, Nakkiran & Barak (NeurIPS 2021) — Model stitching confirms same-architecture compatibility
- Muttenthaler et al. (2024) — Training objective drives cross-dataset consistency
