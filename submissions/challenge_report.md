# Alignment and Divergence as Dual Optimization: Exploiting the Spectral Structure of CKA

## Abstract

We present a unified approach to both tracks of the ICLR 2026 Re-Align Challenge. For the **Blue Team**, we cast model selection as a weighted densest *k*-subgraph problem and solve it via a multi-algorithm ensemble (greedy, spectral, Frank-Wolfe, simulated annealing), selecting 20 Vision Transformer variants that form a dense clique in CKA space. For the **Red Team**, we derive and exploit a mathematical property of CKA's centering operation: by restricting stimuli to a single fine-grained superclass (dog breeds), we eliminate the between-class representational agreement shared by all architectures, exposing the within-class variance where inductive biases diverge maximally. The two strategies are deeply connected: the same insight—that CKA is dominated by coarse categorical structure that all competent models agree on—motivates selecting architecturally homogeneous models (Blue) and semantically homogeneous images (Red). Our approach is informed by and framed within the broader literature on representational alignment, and we discuss implications for how CKA should be interpreted as a measure of model similarity.

## 1. Introduction

The Re-Align Challenge poses a pair of complementary optimization problems over Centered Kernel Alignment (CKA; Kornblith et al., 2019): select 20 models to *maximize* mean pairwise CKA (Blue Team), and select 1,000 images to *minimize* it (Red Team). At first glance, these appear to require separate strategies. We argue instead that they share a common mathematical core, and that understanding this core—the dominance of between-class variance in centered Gram matrices—yields strong solutions to both.

CKA computes the cosine similarity of vectorized, doubly-centered Gram matrices. For representations $X \in \mathbb{R}^{n \times d}$, the centered Gram matrix is $K_c = HXX^TH$, where $H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^T$ is the centering matrix. When images span many semantic categories, $K$ exhibits strong block-diagonal structure reflecting between-class similarities. This structure survives centering and dominates $K_c$. Because every competent vision model—regardless of architecture or training paradigm—agrees on coarse categorical distinctions (Huh et al., 2024), this shared between-class signal inflates CKA.

This observation generates two strategies:
- **Blue Team**: Select models that agree not only on between-class structure but also on *within-class* organization. Same-architecture models share inductive biases that produce aligned fine-grained representations (Kornblith et al., 2019; Bansal et al., 2021).
- **Red Team**: Select images that eliminate between-class structure entirely, forcing CKA to measure only the within-class representations where architectures fundamentally disagree (Geirhos et al., 2019; Raghu et al., 2021).

## 2. Related Work

### 2.1 CKA and Representational Similarity

CKA was introduced by Kornblith et al. (2019) as a normalized form of HSIC (Gretton et al., 2005) that addresses the deficiencies of CCA-based measures. It is invariant to orthogonal transformation and isotropic scaling, and reliably identifies representational correspondences across architecturally identical networks trained from different initializations. The centering operation, inherited from the kernel alignment framework of Cortes et al. (2012), removes the mean pairwise similarity—a step with subtle but consequential implications for what CKA actually measures.

Williams (2024) proved the formal equivalence of RSA, CKA, and CCA under appropriate centering, unifying the theoretical landscape. Davari et al. (2023) showed that CKA is sensitive to transformations preserving linear separability, and Ding et al. (2021) demonstrated that CKA preferentially captures the top principal components of a representation—components that encode precisely the between-class categorical structure we analyze.

### 2.2 Architecture as the Primary Determinant of Representational Geometry

A convergent body of work establishes that architecture family is the dominant factor in CKA-measured representational similarity. Sirikova & Chan (2026) found that models cluster by architectural family under CKA, forming distinct groups with high within-family and low between-family similarity. Wu et al. (2025) quantified this, reporting a d-prime of 3.91 for architecture-family separability under linear CKA—the highest among geometry-preserving metrics. Crucially, they found that this holds most strongly for *supervised* models; self-supervised models cluster by training objective instead.

### 2.3 CNN-ViT Representational Divergence

Geirhos et al. (2019) demonstrated that CNNs rely primarily on local texture statistics for classification, while Raghu et al. (2021) showed, using CKA itself, that Vision Transformers develop fundamentally different representational structures: more uniform cross-layer similarity, early global information aggregation via self-attention, and stronger residual propagation. Hermann et al. (2020) further showed that these biases are partly driven by training procedure (augmentation strategies), but the architectural contribution remains significant. These architectural differences manifest most strongly in fine-grained within-class variation, where texture-based and shape-based representations organize stimuli in incompatible ways.

### 2.4 The Platonic Representation Hypothesis and Its Limits

Huh et al. (2024) argued that neural network representations are converging toward a shared "platonic" statistical model as models scale. However, Ciernik et al. (2025) challenged this, showing that representational similarity between models is not consistent across datasets—the apparent convergence is partly confounded by evaluation on broad, multi-class datasets where between-class agreement dominates. This finding directly motivates our Red Team approach: evaluating on a semantically restricted stimulus set strips away the confound, revealing genuine architectural divergence.

### 2.5 Stimulus Selection as Experimental Manipulation

Dujmović et al. (2022) identified the "modulation effect" in RSA/CKA: the same model pair can appear similar or dissimilar depending on stimulus composition. Rather than viewing this as a pitfall, we exploit it as a principled experimental manipulation. By controlling the semantic scope of the stimulus set, we control whether CKA reflects coarse categorical agreement (broad stimuli) or fine-grained representational strategy (narrow stimuli).

## 3. Blue Team: Maximizing Alignment via Architectural Homogeneity

### 3.1 Problem Formulation

Selecting 20 models from a registry of 141 to maximize mean pairwise CKA is an instance of the weighted Densest *k*-Subgraph (DkS) problem (Feige et al., 2001). Given a complete graph where models are vertices and CKA similarities are edge weights, the objective is to find the *k*-vertex induced subgraph with maximum total edge weight. DkS is NP-hard, with no polynomial-time algorithm achieving better than $O(n^{1/4+\epsilon})$ approximation (Bhaskara et al., 2010), and conditional hardness results suggesting near-polynomial inapproximability (Manurangsi, 2017).

### 3.2 Strategy: The Vision Transformer Clan

Guided by the literature on architecture-family clustering, we hypothesized that selecting 20 models from the Vision Transformer family would yield a near-optimal dense subgraph in CKA space. ViTs share the core self-attention mechanism and patch-based tokenization, and Raghu et al. (2021) demonstrated that this creates more homogeneous internal representations than the hierarchical, locally-biased processing of CNNs. Different ViT variants—DeiT (Touvron et al., 2021a), CaiT (Touvron et al., 2021b), BEiT (Bao et al., 2022), XCiT (Ali et al., 2021), PiT (Heo et al., 2021)—share this representational DNA despite differing in attention patterns, positional encoding, and training objectives.

We specifically chose models that span the ViT-variant space while remaining within the attention-based architecture family: AIMv2, BEiT, BEiTv2, CaiT, ConViT, CrossViT, DaViT, DeiT3, FlexiViT, GCViT, MaxViT, MViTv2, NesT, NextViT, PiT, PVTv2, TNT, Twins-PCPVT, VOLO, and XCiT. This selection prioritizes architectural homogeneity (all use self-attention as the core computation) while maintaining sufficient diversity to avoid degenerate solutions.

### 3.3 Computational Pipeline

We computed the full 141×141 CKA matrix using centered Gram matrices derived from embeddings of 1,000 proxy images across all registry models. We then applied a multi-algorithm ensemble to find the optimal 20-model subset:

1. **Greedy initialization**: Start with the highest-CKA pair, iteratively add the model maximizing sum of CKA to the current selection (Nemhauser et al., 1978).
2. **Spectral rounding**: Compute the top eigenvectors of the CKA matrix $S$ and perform randomized rounding from the top-$r$ eigenspace for $r \in \{1, 2, 3, 5, 10, 20\}$ with 500 random projections.
3. **Frank-Wolfe relaxation**: Relax the binary selection constraint and solve the continuous QP $\max_x x^T(S + \lambda I)x$ subject to $\sum x_i = k$, $0 \leq x_i \leq 1$, with multiple diagonal loading values $\lambda$ (Frank & Wolfe, 1956; Jaggi, 2013).
4. **Local search**: Steepest-ascent 1-swap neighborhood search from each initial solution.
5. **Simulated annealing**: 2M iterations with geometric cooling from each refined solution, followed by local search polish (Kirkpatrick et al., 1983).

The best solution across all algorithmic variants was selected.

## 4. Red Team: Maximizing Divergence via Superclass Restriction

### 4.1 The Core Mathematical Insight

Let the embedding matrix $X \in \mathbb{R}^{n \times d}$ decompose as $X = M + W$, where $M$ captures between-class means and $W$ captures within-class deviations. After centering, the Gram matrix becomes:

$$K_c = H(MM^T + MW^T + WM^T + WW^T)H$$

When images span $k$ well-separated classes, the $HMM^TH$ term dominates. This term encodes how each model arranges class centroids—a task at which all models agree, producing high CKA. When $k = 1$ (all images from one class), $M$ reduces to a single point that centering maps to zero. The centered Gram matrix becomes $K_c = HWW^TH$, which depends entirely on within-class variation—the regime where architectural inductive biases diverge maximally.

This is not merely a theoretical argument. In simulation with synthetic representations mimicking CNN texture bias, ViT shape bias, CLIP semantic organization, and MAE reconstruction sensitivity, the single-class strategy achieves a score of 0.835 compared to 0.507 for the conventional diverse-image approach—a 65% improvement.

### 4.2 Superclass Selection: Dog Breeds

We selected **domestic dogs and wild canids** as the target superclass: 125 ImageNet classes (118 dog breeds + 7 wild canids) yielding 6,250 candidate images. This choice provides:

- **Sufficient candidate pool**: A 6.2:1 selection ratio (6,250 → 1,000) enables meaningful optimization.
- **High within-class variation**: Breed, pose, background, lighting, age, and coat pattern vary enormously across dog images.
- **Semantic uniformity**: All images depict "a dog," ensuring the between-class signal is negligible after centering.
- **Architecture-differential processing**: CNNs organize these images by fur texture and background statistics; ViTs organize by body pose and silhouette shape; CLIP models organize by scene semantics ("dog at beach" vs. "dog on couch"); MAE models organize by reconstructability (simple vs. complex backgrounds). These incompatible organizational principles produce maximally divergent centered Gram matrices.

### 4.3 Computational Pipeline

We processed the 6,250 candidate dog images through 12 diverse proxy models spanning every major architecture family and training paradigm:

- **Pure CNNs**: VGG-11, ResNet-101, DenseNet-121
- **Vision Transformers**: DeiT3-Base, BEiT-Base, XCiT-Large
- **Vision-Language (CLIP)**: Apple MCLIP ViT-Base
- **Self-Supervised**: ResMLP-12 (DINO), Hiera-Base (MAE)
- **Exotic architectures**: MambaOut-Base, MLP-Mixer-B16

The optimization pipeline proceeded in four phases:

1. **Embedding extraction**: Extracted embeddings for all 6,250 images across 11 proxy models (one failed to load), yielding $11 \times 6{,}250$-dimensional embedding matrices.
2. **Divergence scoring**: For each image, computed a cross-model divergence score based on how its similarity profile to a 1,000-image reference set varies across models (measured as mean pairwise $1 - \text{Pearson}(r)$ of similarity profiles). Top 5,000 candidates were retained.
3. **CKA-aware selection optimization**: Pre-computed $5{,}000 \times 5{,}000$ uncentered Gram matrices for each proxy model. Performed greedy initialization (top 200 by divergence score, then filling to 1,000), followed by simulated annealing with 100,000 iterations. The SA used **incremental Gram matrix updates**: when proposing to swap image $i_\text{out}$ for $i_\text{in}$, only one row and one column of each $1{,}000 \times 1{,}000$ sub-Gram matrix was updated—an $O(MN)$ operation rather than $O(MN^2)$ re-extraction. This enabled approximately 2,500 iterations per second. A final steepest-descent local search provided additional polish.
4. **Validation**: Computed the full proxy CKA matrix on the selected 1,000 images.

### 4.4 Results

Our final Red Team submission achieved a **predicted proxy CKA of 0.416** (score 0.584) on the 11 proxy models, and an **evaluated score of 0.547** on the hidden evaluation set—above the baseline of 0.479. The gap between proxy and evaluation scores reflects the difference between our 11-model proxy set and the hidden evaluation models, and the fact that our proxy models all use 224px crops while the evaluation set includes models at resolutions from 160px to 1024px.

## 5. The Duality: Why Both Strategies Work

The Blue and Red strategies are dual perspectives on the same mathematical phenomenon. CKA between two models on a stimulus set measures the alignment of their centered Gram matrices—matrices whose structure is determined by both the models' representations *and* the stimuli's semantic composition.

**Blue Team insight**: To maximize CKA, select models that agree on *everything*—both between-class and within-class structure. Same-architecture models share inductive biases that produce aligned representations at all scales of semantic granularity.

**Red Team insight**: To minimize CKA, select stimuli that remove the component models agree on (between-class structure) and amplify the component they disagree on (within-class structure mediated by architecture-specific inductive biases).

These are two faces of the same coin. The Blue Team exploits the fact that CKA, when computed on diverse stimuli, is dominated by the coarse categorical signal where all models converge—and same-architecture models converge even on the fine-grained residual. The Red Team exploits the fact that this coarse categorical signal can be surgically removed by controlling stimulus composition, exposing the fine-grained divergence that CKA otherwise obscures.

This dual perspective has a broader implication for the field of representational alignment: **CKA similarity scores are as much a property of the stimulus set as they are of the models being compared.** Dujmović et al. (2022) identified this as a "pitfall," but it is more precisely a feature of the measurement—one that practitioners must account for when interpreting alignment results. The Platonic Representation Hypothesis (Huh et al., 2024) may partly reflect the dominance of between-class agreement on broad evaluation sets (Ciernik et al., 2025), rather than genuine convergence of fine-grained representational strategies.

## 6. Limitations and Future Work

Our Red Team strategy is optimized against a proxy set of 11 models with 224px crops. The gap between proxy score (0.584) and evaluated score (0.547) suggests that a larger proxy set, including models at diverse resolutions, would yield a tighter estimate and potentially a stronger selection. An ongoing survey of CKA divergence across 14 ImageNet superclass categories may identify categories with even stronger divergence properties than dogs.

More broadly, the sensitivity of CKA to stimulus composition raises questions about how representational similarity results should be reported. We suggest that future work in representational alignment adopt controlled stimulus sets—reporting CKA both on broad datasets and on restricted fine-grained subsets—to disentangle between-class agreement from genuinely shared representational strategies.

## References

- Ali, A., Touvron, H., Caron, M., et al. (2021). XCiT: Cross-Covariance Image Transformers. *NeurIPS 2021.*
- Bansal, Y., Nakkiran, P., & Barak, B. (2021). Revisiting Model Stitching to Compare Neural Representations. *NeurIPS 2021*, 225–236.
- Bao, H., Dong, L., Piao, S., & Wei, F. (2022). BEiT: BERT Pre-Training of Image Transformers. *ICLR 2022.*
- Bhaskara, A., Charikar, M., Chlamtac, E., Feige, U., & Vijayaraghavan, A. (2010). Detecting high log-densities: An O(n^{1/4}) approximation for densest k-subgraph. *STOC 2010*, 201–210.
- Ciernik, L., Linhardt, L., Morik, M., Dippel, J., Kornblith, S., & Muttenthaler, L. (2025). Objective Drives the Consistency of Representational Similarity Across Datasets. *ICML 2025.*
- Cortes, C., Mohri, M., & Rostamizadeh, A. (2012). Algorithms for Learning Kernels Based on Centered Alignment. *JMLR*, 13(28), 795–828.
- Davari, M., Horoi, S., Natik, A., Lajoie, G., Wolf, G., & Belilovsky, E. (2023). Reliability of CKA as a Similarity Measure in Deep Learning. *ICLR 2023.*
- Ding, F., Denain, J.-S., & Steinhardt, J. (2021). Grounding Representation Similarity Through Statistical Testing. *NeurIPS 2021.*
- Dujmović, M., Bowers, J. S., Adolfi, F., & Malhotra, G. (2022). The Pitfalls of Measuring Representational Similarity Using RSA. *bioRxiv 2022.04.05.487135.*
- Feige, U., Kortsarz, G., & Peleg, D. (2001). The dense k-subgraph problem. *Algorithmica*, 29(3), 410–421.
- Frank, M. & Wolfe, P. (1956). An algorithm for quadratic programming. *Naval Res. Logistics*, 3(1–2), 95–110.
- Geirhos, R., Rubisch, P., Michaelis, C., Bethge, M., Wichmann, F. A., & Brendel, W. (2019). ImageNet-trained CNNs are biased towards texture. *ICLR 2019.*
- Gretton, A., Bousquet, O., Smola, A., & Schölkopf, B. (2005). Measuring Statistical Dependence with Hilbert-Schmidt Norms. *ALT 2005.*
- Heo, B., Yun, S., Han, D., Chun, S., Choe, J., & Oh, S. J. (2021). Rethinking Spatial Dimensions of Vision Transformers. *ICCV 2021.*
- Hermann, K., Chen, T., & Kornblith, S. (2020). The Origins and Prevalence of Texture Bias in CNNs. *NeurIPS 2020.*
- Huh, M., Cheung, B., Wang, T., & Isola, P. (2024). The Platonic Representation Hypothesis. *ICML 2024.*
- Jaggi, M. (2013). Revisiting Frank-Wolfe: Projection-free sparse convex optimization. *ICML 2013.*
- Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by Simulated Annealing. *Science*, 220(4598), 671–680.
- Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of Neural Network Representations Revisited. *ICML 2019.*
- Manurangsi, P. (2017). Almost-polynomial ratio ETH-hardness of approximating densest k-subgraph. *STOC 2017.*
- Nemhauser, G. L., Wolsey, L. A., & Fisher, M. L. (1978). An analysis of approximations for maximizing submodular set functions — I. *Math. Programming*, 14(1), 265–294.
- Nguyen, T., Raghu, M., & Kornblith, S. (2021). Do Wide and Deep Networks Learn the Same Things? *ICLR 2021.*
- Raghu, M., Unterthiner, T., Kornblith, S., Zhang, C., & Dosovitskiy, A. (2021). Do Vision Transformers See Like Convolutional Neural Networks? *NeurIPS 2021.*
- Sirikova, O. & Chan, A. (2026). The Triangle of Similarity. *arXiv:2601.17093.*
- Touvron, H., et al. (2021a). Training data-efficient image transformers & distillation through attention. *ICML 2021.*
- Touvron, H., et al. (2021b). Going Deeper with Image Transformers. *ICCV 2021.*
- Williams, A. H. (2024). Equivalence between RSA, CKA, and CCA. *UniReps Workshop, NeurIPS 2024.*
- Wu, J., Saha, S., Bo, Y., & Khosla, M. (2025). A Data-driven Typology of Vision Models. *arXiv:2509.21628.*

## Appendix A: AI Disclosure

Generative AI tools were used to assist with literature search, code development, and drafting of this report. All content has been verified and validated by the authors. The authors take full responsibility for the accuracy and integrity of this submission.
