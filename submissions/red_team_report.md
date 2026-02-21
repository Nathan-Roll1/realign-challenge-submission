---
layout: distill
title: "Exploit the Centering: Maximizing Representational Divergence via Semantic Restriction"
description: Re-Align Challenge Report (Red Team) - Exploiting CKA's centering operation via a fine-grained single-superclass strategy.
date: 2026-02-26
future: true
htmlwidgets: true

authors:
 - name: Anonymous

bibliography: re-align-challenge.bib

toc:
 - name: Introduction
 - name: The Core Mathematical Insight
 - name: Methodology
 - name: Results and Honest Evaluation
 - name: Implications for Representational Alignment
---

## Introduction

The Re-Align Challenge's Red Team track requires selecting an image set that causes representations across ~143 vision models to diverge. Representational alignment is evaluated using Centered Kernel Alignment (CKA) [@kornblith2019similarity].

A naive strategy for minimizing CKA might involve selecting highly diverse or anomalous images to "confuse" models. However, we argue that this approach fundamentally misunderstands what CKA measures. CKA computes the cosine similarity of vectorized, doubly-centered Gram matrices. When an image set spans multiple distinct semantic categories (e.g., dogs, cars, furniture), the uncentered Gram matrix $K$ exhibits strong block-diagonal structure. This between-class structure survives the centering operation $K_c = HKH$ and dominates the overall variance.

Because all models in the registry are competent ImageNet classifiers, they all agree on these coarse categorical distinctions. This shared agreement inflates CKA. To minimize CKA, one must *remove* the signal that models agree upon.

Recent literature has diagnosed these severe confounds in CKA: Cui et al. [@cui2022deconfounded] demonstrated that the metric is deeply confounded by population structure in the input space, and Cloos et al. [@cloos2024differentiable] proved its quadratic dependence on high-variance principal components. While prior work has sought to patch these flaws by proposing debiased [@murphy2024correcting] or deconfounded [@cui2022deconfounded] alternative metrics, we take an adversarial perspective. We demonstrate how to maximally exploit this known vulnerability through semantic restriction.

---

## The Core Mathematical Insight

Let the embedding matrix $X \in \mathbb{R}^{n \times d}$ decompose as $X = M + W$, where $M$ captures between-class means and $W$ captures within-class deviations. After centering, the Gram matrix becomes:

$$K_c = H(MM^T + MW^T + WM^T + WW^T)H$$

When images span $k$ well-separated classes, the $HMM^TH$ term dominates. When $k = 1$ (all images from one class), $M$ reduces to a single point that the centering matrix $H$ maps to zero. The centered Gram matrix becomes $K_c = HWW^TH$, which depends entirely on within-class variation.

It is within this fine-grained variation that architectures fundamentally disagree:
- **CNNs** organize these images by local fur texture and background statistics [@geirhos2019imagenettrained].
- **Vision Transformers (ViTs)** aggregate global information early and organize by body pose and silhouette shape [@raghu2021vision].
- **Vision-Language Models (CLIP)** organize by scene semantics.

By restricting stimuli to a single superclass, we surgically remove the $HMM^TH$ term, isolating the architectural divergence.

![Gram Matrix Comparison](red_team_real_gram_insight.png "Figure 1: The Effect of Semantic Restriction on Gram Matrices (Real ResNet50 Embeddings)")

---

## Methodology

### Stimulus Selection: The "Dog Breeds" Superclass

We selected **domestic dogs and wild canids** as our target superclass. From the ImageNet validation and ObjectNet datasets, we filtered for 125 classes (118 dog breeds + 7 wild canids), yielding 6,250 candidate images. This choice provides high within-class variation (breed, pose, lighting) while maintaining strict semantic uniformity ("a dog").

### Optimization Pipeline

To select the optimal 1,000 images from the 6,250 candidates, we built a computationally efficient proxy optimization pipeline.

1. **Proxy Models**: We selected 11 diverse proxy models spanning major architectures: ResNet, VGG, DenseNet, DeiT, BEiT, XCiT, CLIP, DINO, MAE, MambaOut, and MLP-Mixer.
2. **Divergence Scoring**: We scored all 6,250 candidates based on cross-model divergence and retained the top 5,000.
3. **Simulated Annealing**: We optimized the 1,000-image subset using simulated annealing (SA). A naive CKA computation at each SA iteration costs $O(M N^2)$ where $M$ is the number of models and $N=1,000$. We derived an **incremental Gram matrix update** algorithm: when swapping one image, only a single row and column of the sub-Gram matrix is updated. This reduced per-iteration cost to $O(MN)$, enabling us to run 100,000 SA iterations in under 40 minutes.

---

## Results and Honest Evaluation

Our final submission achieved a predicted proxy CKA of 0.416 (score: $1 - 0.416 = 0.584$) on our 11 proxy models. On the organizers' hidden evaluation set, the submission achieved a score of **0.547**.

**Limitations and Generalization Gap**: We observed a generalization gap between our proxy score and the final evaluation score. We attribute this to two factors in our proxy design:
1. **Proxy Pool Size**: 11 models may have been too small to fully capture the variance of the 143-model evaluation pool.
2. **Resolution Homogeneity**: To simplify embedding extraction, we used 224px crops for all proxy models. However, the evaluation pool includes models utilizing crop sizes ranging from 160px to 1024px. Preprocessing differences strongly influence CKA, and our optimization may have overfit to the 224px regime.

Despite this gap, the semantic restriction strategy successfully produced a winning divergence score, validating the underlying mathematical hypothesis.

---

## Implications for Representational Alignment

Our findings echo a critical nuance in representational alignment research: **CKA similarity scores are highly sensitive to stimulus composition** [@dujmovic2022pitfalls; @ding2021grounding]. Recently, Ciernik et al. [@ciernik2025objective] demonstrated that objective drives the consistency of representational similarity across datasets, challenging the notion of a uniform "Platonic" convergence [@huh2024platonic].

Our Red Team result demonstrates this practically. By treating the known dataset-confound of CKA—which is particularly pronounced when features outnumber samples [@murphy2024correcting]—not as a bug to be patched but as an attack vector to be exploited, we systematically minimized the alignment score. If researchers wish to measure genuine differences in architectural inductive biases, they must evaluate on datasets where coarse semantic structure does not overwhelmingly dominate the variance.
