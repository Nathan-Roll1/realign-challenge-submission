# Paper Analysis: CKA Subset Selection Optimization

## Problem Restatement

Select S ⊂ {1,...,141}, |S| = 20, to maximize:

```
f(S) = (1/C(20,2)) Σ_{i<j, i,j∈S} CKA(K_i, K_j)
```

where CKA(K_i, K_j) = HSIC(K_i, K_j) / √(HSIC(K_i, K_i) · HSIC(K_j, K_j)).

**Critical reformulation**: Let C be the 141×141 matrix with C_ij = CKA(K_i, K_j). Let x ∈ {0,1}^141 with Σx_i = 20. Then:

```
f(S) ∝ x^T C x - 20    (since CKA(K_i, K_i) = 1, subtract the diagonal)
```

So our problem is: **maximize x^T C x subject to x ∈ {0,1}^141, 1^T x = 20**.

This is the **weighted densest k-subgraph problem** — NP-hard in general, but our instance (n=141, k=20, complete weighted graph) is structured enough for exact or near-exact solutions.

---

## Paper 1: Cortes, Mohri, Rostamizadeh (2012, JMLR)
**"Algorithms for Learning Kernels Based on Centered Alignment"**

### Key Mathematical Insight

The paper defines the **alignment matrix** M ∈ R^{p×p} between p base kernels:

```
M_kl = ⟨K_kc, K_lc⟩_F    (Frobenius inner product of centered kernel matrices)
```

and the alignment vector a ∈ R^p with a target kernel K_Y:

```
a_k = ⟨K_kc, K_Yc⟩_F
```

They show that finding the optimal convex combination K_μ = Σ μ_k K_k to maximize centered alignment with K_Y reduces to:

```
max_μ  (μ^T a) / √(μ^T M μ)    subject to μ ≥ 0
```

**For a linear (unconstrained) combination, the closed-form solution is:**

```
μ* = M^{-1} a / ‖M^{-1} a‖
```

For the convex (non-negative) case, this reduces to a **simple QP** after a variable substitution (ν = μ/‖μ‖), becoming:

```
min_ν  ν^T M ν    subject to  ν^T a = 1, ν ≥ 0
```

### Adaptation to Our Problem

**The M matrix IS our CKA matrix** (up to normalization). Specifically:

```
CKA(K_i, K_j) = M_ij / √(M_ii · M_jj)
```

So the alignment matrix from Cortes et al. is exactly the unnormalized version of the pairwise CKA matrix we're optimizing over.

**However, there's a fundamental difference**: Cortes et al. optimize weights μ to maximize alignment with an **external target** K_Y. We have **no target** — we want to maximize mutual alignment among the selected subset.

**The adaptation**: Our problem is the "targetless" analog. Instead of max μ^T a / √(μ^T M μ), we want:

```
max_x  x^T C x    subject to  x ∈ {0,1}^141, 1^T x = 20
```

The continuous relaxation (replacing x ∈ {0,1} with x ∈ [0,1]) yields:

```
max_μ  μ^T C μ    subject to  0 ≤ μ_i ≤ 1, 1^T μ = 20
```

This is a **non-convex** QP (maximizing a convex quadratic), but:

1. **The top eigenvector of C gives the relaxed solution direction.** Models with large components in the top eigenvector of C are the ones that contribute most to overall alignment. This gives a spectral heuristic.

2. **SDP relaxation**: Replace x x^T with a matrix variable X ≽ 0, giving the semidefinite relaxation:
```
max  ⟨C, X⟩    subject to  diag(X) ≤ 1, ⟨I, X⟩ = 20, X ≽ 0, X ≥ 0
```

### Specific Algorithmic Trick

**The independent alignment algorithm (Section 3.1)** sets μ_k ∝ ⟨K_kc, K_Yc⟩_F. The analog for our targetless problem: set μ_k ∝ Σ_j CKA(K_k, K_j) — i.e., **select the 20 models with the highest row-sums of the CKA matrix.** This is the simplest baseline (greedy by average alignment).

**The alignment maximization algorithm (Section 3.2)** accounts for correlations between kernels via the M matrix. The analog: don't just pick high-CKA models, account for the fact that some high-CKA models are redundant with each other. The QP structure suggests we should look at the **spectral structure of C** to find models that jointly maximize quadratic form.

### Non-obvious Insight

**The key non-obvious result**: The concentration bound (Theorem 10) shows that empirical kernel alignment concentrates around population alignment at rate O(1/√m). This means if you compute CKA on a subsample of your data, the ranking of models is stable. You can do **fast approximate CKA computation on subsamples** to screen candidates before exact evaluation — without worrying about the ranking changing.

Also: the closed-form μ* = M^{-1}a suggests that if we had a target, we could solve the problem exactly. This motivates a **synthetic target** approach: create K_target = (1/20) Σ_{i∈S*} K_i where S* is the current best subset, then use M^{-1}a to find the next improvement.

---

## Paper 2: Zhou et al. (2024, IJCAI/arXiv:2401.11824)
**"Rethinking Centered Kernel Alignment in Knowledge Distillation"**

### Key Mathematical Insight

**Theorem 1**: CKA measures the **cosine similarity between vectorized Gram matrices**:

```
CKA(X, Y) = vec(XX^T)^T vec(YY^T) / (‖vec(XX^T)‖₂ · ‖vec(YY^T)‖₂)
```

**Theorem 2**: CKA decomposes as an upper bound on MMD:

```
CKA(X, Y) = -N · E_{i,j}[(⟨x_i, x_j⟩ - ⟨y_i, y_j⟩)²] + 2
           ≤ -N · (E_{i,j}[⟨x_i, x_j⟩] - E_{i,j}[⟨y_i, y_j⟩])² + 2
```

The inequality is Jensen's. The first term is the **mean squared elementwise difference** between the two kernel matrices; the second is the squared difference of their **mean kernel values**.

### What Drives High CKA Values?

From the exact expression: CKA(K_i, K_j) is high when the **elementwise differences between centered kernel matrices are small**. Specifically:

```
CKA(K_i, K_j) = 2 - N · E_{s,t}[(K_ic(s,t) - K_jc(s,t))²] / (‖K_ic‖_F · ‖K_jc‖_F)
```

This means: **Two models have high CKA when they agree on the pairwise similarity structure of the data.** Not just on which pairs are similar, but on the *magnitude* of similarity for every pair.

### Adaptation: The MMD Perspective for Model Selection

Since maximizing CKA ≈ minimizing pairwise MMD, our problem becomes:

```
Select 20 models to minimize the sum of pairwise MMD² between their kernel matrices.
```

**This reframes the problem geometrically**: Each centered kernel matrix K_ic, when vectorized and normalized to unit norm, is a point on a high-dimensional unit sphere. CKA = cosine similarity = dot product on this sphere. Our problem becomes:

> **Find 20 points on the unit sphere (from 141 candidates) whose pairwise dot products are maximized — i.e., the tightest angular cluster.**

This is the **spherical k-center / min-diameter subset** problem on the sphere, which has known algorithms:

1. **Angular PCA**: Project the 141 normalized kernel vectors onto the top principal components. Points clustered tightly in this low-dimensional projection will have high mutual CKA.

2. **Iterative shrinking**: Start with all 141 models. Repeatedly remove the model with the lowest average cosine similarity to the remaining set, until 20 remain.

### Specific Algorithmic Trick

**The constant term in the decomposition is key.** The "+2" constant means CKA = 2 - (penalty for disagreement). This means:

```
Σ_{i<j} CKA(K_i, K_j) = 190 · 2 - Σ_{i<j} penalty(i,j) = 380 - Σ_{i<j} penalty(i,j)
```

So **maximizing total CKA = minimizing total pairwise kernel disagreement.** The penalty is:

```
penalty(i,j) = N · ‖vec(K_ic) / ‖K_ic‖_F - vec(K_jc) / ‖K_jc‖_F ‖² · (some scaling)
```

This is a Euclidean distance on the unit sphere. So minimizing total penalty = finding the **minimum-diameter k-subset** in Euclidean space, where each point is the normalized vectorized kernel matrix.

### Non-obvious Insight

**The MMD perspective reveals what CKA does NOT capture**: CKA is invariant to isotropic scaling of features. Two models where one produces embeddings 10x larger than the other but with the same relative structure will have CKA = 1. This means models with wildly different embedding magnitudes but identical similarity structures are considered identical by CKA.

**For subset selection, this means**: You should NOT separately consider "magnitude diversity" — CKA already factors that out. Models that differ only in embedding scale are correctly treated as redundant by CKA.

Also: **The Gram matrix interpretation** (CKA = cosine similarity of vectorized Gram matrices) means you can precompute a 141-dimensional "CKA feature vector" for each model (its row in the CKA matrix C) and do the selection in this 141-dimensional space. Since C is PSD, you can eigendecompose it and work in the top-k eigenspace, dramatically reducing the search space.

---

## Paper 3: Lu et al. (2014, Pattern Recognition)
**"Multiple Kernel Clustering Based on Centered Kernel Alignment"**

### Key Mathematical Insight

Lu et al. formulate MKC as jointly optimizing kernel combination weights and cluster assignments. The kernel weights μ are found by maximizing the alignment between the combined kernel and an "ideal" clustering kernel:

```
max_μ  ρ̂(K_μ, K_ideal)    subject to μ ≥ 0, ‖μ‖₁ = 1
```

where K_ideal is derived from cluster assignments (e.g., K_ideal = HH^T where H is the cluster indicator matrix).

This creates an **alternating optimization**:
1. Fix cluster assignments → optimize kernel weights (QP, as in Cortes et al.)
2. Fix kernel weights → optimize cluster assignments (standard k-means or spectral clustering on K_μ)

### Adaptation: Finding the Tightest Cluster of 20

**Yes, their CKA-based clustering can directly find dense subsets.** Here's how:

**Method 1 — Direct spectral clustering of the CKA matrix**:
1. Build C (141×141 CKA matrix)
2. C is itself a kernel/similarity matrix
3. Apply spectral clustering to C with varying numbers of clusters
4. The cluster with the highest intra-cluster average CKA IS a candidate solution
5. If this cluster has ≥20 members, select the 20 with highest within-cluster centrality
6. If <20, merge with the next most similar cluster

**Method 2 — Iterative alignment maximization**:
1. Initialize: pick the 20 models with highest row-sums of C
2. Compute K_center = (1/20) Σ_{i∈S} K_i (the "centroid kernel")
3. Find the 20 models with highest CKA to K_center (re-solve selection)
4. Repeat until convergence

This is analogous to k-means but for finding ONE tight cluster of exactly k=20 models.

### Specific Algorithmic Trick

**The connection between kernel clustering and spectral methods gives us a powerful approach**:

The top eigenvectors of the CKA matrix C encode the cluster structure. Specifically, the **top eigenvector v₁ of C** gives the "average alignment direction" — models with large v₁ components are most central. The **second eigenvector v₂** gives the primary axis of variation.

For finding the tightest 20-cluster, we want models that are:
- Large in v₁ (high average alignment with everyone)
- Small in |v₂|, |v₃|, ... (minimal variation from the cluster center)

So: **rank models by v₁_i - λ·Σ_{j≥2} v_j_i², then pick the top 20.** This penalizes models that are "pulled" toward different sub-clusters.

### Non-obvious Insight

**The alternating optimization can escape local optima that greedy search misses.** Pure greedy search (add the model that most increases mean CKA) is a forward-selection heuristic that never reconsiders past choices. The Lu et al. approach alternates between:
- "Which models best align with the current center?" (selection step)
- "What is the center of the current selection?" (centroid step)

This can **swap** models in and out, unlike greedy. The alternation is guaranteed to converge (monotonic increase in alignment). Combined with multiple random restarts, this searches much more of the combinatorial space.

---

## Paper 4: Cortes, Mohri, Rostamizadeh (ICML 2010)
**"Two-Stage Learning Kernel Algorithms"**

### Key Mathematical Insight

The closed-form weight solution for the linear (unconstrained) combination:

```
μ* = M^{-1} a    (up to normalization)
```

where M_kl = ⟨K_kc, K_lc⟩_F and a_k = ⟨K_kc, K_Yc⟩_F.

For the **convex combination** (μ ≥ 0), this becomes a QP:
```
min_ν  ν^T M ν    subject to  ν^T a = 1, ν ≥ 0
```

The key property: M is PSD (it's a Gram matrix of centered kernels in Frobenius inner product space), so this QP is convex and efficiently solvable.

### Is There an Analogous Closed-Form for Subset Selection?

**Not directly, but there are useful analogs:**

**Relaxation 1 — Continuous weight optimization (no target)**:
Without a target kernel, our "alignment maximization" becomes:
```
max_μ  μ^T M μ / (μ^T μ)    subject to μ ≥ 0
```
This is a **non-negative principal component** problem. The solution is the top eigenvector of M restricted to non-negative entries (computable via non-negative matrix factorization or power iteration with clamping).

**Relaxation 2 — Ratio cut analogy**:
If we define the problem as max μ^T C μ subject to ‖μ‖₀ = 20, ‖μ‖₂ = 1, the relaxation (dropping the sparsity constraint) gives:
```
μ* = top eigenvector of C
```
Then round by selecting the 20 largest-magnitude entries. This is the **spectral relaxation** approach.

**Relaxation 3 — Using the closed-form with a synthetic target**:
1. Compute the average kernel: K_avg = (1/141) Σ_i K_i
2. Set a_k = ⟨K_kc, K_avg_c⟩_F = Σ_j M_kj / 141 (row mean of M)
3. Apply the closed-form: μ* = M^{-1} a = M^{-1} (M · 1/141) = 1/141 · M^{-1} M · 1 = 1/141 · 1

This gives the trivial uniform solution! This proves that **with the "average" as target, all models get equal weight** — confirming that we need a smarter target.

**Better synthetic target**: Use K_target = (1/20) Σ_{i∈S_current} K_i. Then:
```
a_k = (1/20) Σ_{j∈S_current} M_kj
μ* = M^{-1} a
```
Select the top 20 entries of μ* as the new S. Iterate. This is a **fixed-point iteration on the closed-form**, and each step is O(141² + 141) — trivial computation.

### Specific Algorithmic Trick

**The two-stage framework itself is the trick.** Apply it to our problem:

**Stage 1**: Solve the continuous relaxation to get weights μ ∈ R^{141}_+
**Stage 2**: Use μ to guide the discrete selection

The continuous solution μ* gives a "soft ranking" of models. Instead of rounding to the top 20, use μ* as a **probability distribution** and sample multiple candidate subsets (weighted random sampling without replacement). Evaluate each and keep the best. This is a **randomized rounding** technique that often finds better solutions than deterministic rounding.

### Non-obvious Insight

**The M^{-1} factor in the closed-form is a decorrelation step.** It's the equivalent of whitening: it downweights kernels that are similar to many others (because they don't add independent "alignment information").

Applied to our problem: two models might both have high CKA with everyone (high row-sum), but if they're also highly correlated with each other, M^{-1} will reduce their effective contribution. This is **exactly** the redundancy-aware selection that greedy misses.

**Concrete algorithm**:
1. Compute C (CKA matrix, 141×141)
2. Compute C^{-1} (or pseudoinverse if near-singular)
3. For each candidate subset S, score it as: 1_S^T C 1_S (sum of CKA in subset)
4. But the efficient version: iteratively, for the current subset S with centroid alignment vector a_S = C · 1_S / |S|, compute μ = C^{-1} a_S, and let the top-20 entries of μ define the new S
5. This converges in a few iterations and naturally balances "high alignment" with "low redundancy"

---

## Paper 5: Kandola, Shawe-Taylor, Cristianini (2002)
**"Optimizing Kernel Alignment over Combinations of Kernel"**

### Key Mathematical Insight

**Incomplete Cholesky Factorization (ICF)**: For an n×n kernel matrix K, ICF produces a low-rank approximation:

```
K ≈ L L^T    where L ∈ R^{n×r}, r << n
```

This is equivalent to Gram-Schmidt orthogonalization of training points in feature space. The rank r is chosen adaptively based on the eigenspectrum.

**The key efficiency gain for alignment computation:**

```
⟨K_i, K_j⟩_F = Tr(K_i^T K_j) = Tr(L_i L_i^T L_j L_j^T) = ‖L_i^T L_j‖²_F
```

If both L_i ∈ R^{n×r_i}, then L_i^T L_j is r_i × r_j, and its Frobenius norm costs O(n · r_i · r_j) instead of O(n²) for the full kernel product.

### Relevance for Efficient Computation

**Directly applicable for computing the CKA matrix efficiently.**

For our problem: each K_i = X_i X_i^T where X_i ∈ R^{n×d_i}. The Frobenius inner product is:

```
⟨K_i, K_j⟩_F = ‖X_i^T X_j‖²_F
```

This costs O(n · d_i · d_j) — much cheaper than O(n²) if d_i << n.

**But we need centered kernel matrices.** For K_ic = H K_i H:

```
K_ic = H X_i X_i^T H = (H X_i)(H X_i)^T = X̃_i X̃_i^T
```

where X̃_i = H X_i (centered embedding matrix, n × d_i).

Then:
```
HSIC(K_i, K_j) = (1/(n-1)²) ⟨K_ic, K_jc⟩_F = (1/(n-1)²) ‖X̃_i^T X̃_j‖²_F
```

**This is the crucial computational shortcut**: You never need to form the n×n kernel matrices. The CKA between any pair can be computed directly from the centered embedding matrices:

```
CKA(K_i, K_j) = ‖X̃_i^T X̃_j‖²_F / (‖X̃_i^T X̃_i‖_F · ‖X̃_j^T X̃_j‖_F)
```

where X̃_i^T X̃_j is a d_i × d_j matrix.

### Specific Algorithmic Trick

**If embedding dimensions are large, use ICF on the embedding matrices themselves:**

For X_i ∈ R^{n×d_i} with large d_i, compute the SVD X̃_i = U_i Σ_i V_i^T and truncate to rank r:

```
X̃_i ≈ U_i[:,:r] Σ_i[:r,:r] V_i[:,:r]^T
```

Then X̃_i^T X̃_j becomes an r×r matrix (instead of d_i × d_j), and:

```
CKA(K_i, K_j) ≈ ‖(U_i Σ_i)^T (U_j Σ_j)‖²_F / (‖Σ_i²‖_F · ‖Σ_j²‖_F)
```

**Even more aggressively**: represent each kernel K_i by its top-r eigendecomposition K_ic ≈ Σ_{l=1}^r λ_l^(i) u_l^(i) u_l^(i)^T. Then:

```
⟨K_ic, K_jc⟩_F ≈ Σ_{l,m} λ_l^(i) λ_m^(j) (u_l^(i)^T u_m^(j))²
```

This reduces the CKA matrix computation from O(141² · n²) to O(141² · r² · n), where r is the effective rank.

**For 141 models, n ≈ 50K–100K samples**: this could mean the difference between hours and seconds.

### Non-obvious Insight

**The ICF/low-rank structure reveals the effective dimensionality of alignment.** If all 141 kernel matrices can be well-approximated at rank r, then the CKA matrix C lives in a space of dimension at most r². For typical vision models, r might be 50-200 (the number of effective feature dimensions), meaning the 141×141 CKA matrix has rank ≤ min(141, r²).

**This constrains the optimization landscape**: if C has effective rank d << 141, then the optimal subset is determined by only d dimensions. You can project the 141 models into this d-dimensional space and find the tightest cluster there, which is much easier.

---

## Synthesis: Unified Algorithm

Combining insights from all five papers, here is the **non-obvious algorithm** that pure greedy search would miss:

### Phase 1: Efficient CKA Matrix Construction (Paper 5)

```python
# For each model i, center its embedding matrix
X_tilde_i = H @ X_i  # H = I - 11^T/n

# Compute CKA matrix using embedding inner products
# CKA(i,j) = ||X_tilde_i^T @ X_tilde_j||_F^2 / (||X_tilde_i^T @ X_tilde_i||_F * ||X_tilde_j^T @ X_tilde_j||_F)

# Optionally: truncated SVD for each X_tilde_i to accelerate
```

### Phase 2: Spectral Initialization (Papers 1, 3, 4)

```python
# Eigendecompose the CKA matrix
eigenvalues, eigenvectors = eigh(C)

# Use top eigenvector for initial selection
v1 = eigenvectors[:, -1]  # top eigenvector
S_init = argsort(v1)[-20:]  # top 20 by eigenvector loading
```

### Phase 3: Fixed-Point Iteration with Decorrelation (Papers 1, 4)

```python
S = S_init
for iteration in range(max_iter):
    # Compute centroid alignment vector
    a = C[:, S].mean(axis=1)
    
    # Decorrelate using C^{-1} (the Cortes closed-form analog)
    mu = solve(C, a)  # C^{-1} @ a
    
    # Select top 20
    S_new = argsort(mu)[-20:]
    
    if set(S_new) == set(S):
        break
    S = S_new
```

### Phase 4: Local Search Refinement

```python
# For each model in S, try swapping with each model not in S
# Accept swap if it increases x^T C x
# Repeat until no improvement
```

### Why This Beats Greedy

1. **Greedy (forward selection)** adds one model at a time, never reconsidering. It's trapped by early choices.
2. **Phase 2 (spectral)** gives a globally-informed starting point based on the entire CKA matrix structure.
3. **Phase 3 (decorrelated iteration)** uses the M^{-1} trick from Cortes et al. to account for redundancy between models — two models that are both good but similar to each other will be penalized.
4. **Phase 4 (swap search)** handles the discretization gap from the continuous relaxation.

The decorrelation in Phase 3 is the **non-obvious** insight: raw greedy picks the 20 "best aligned on average" models, but these might form two or three clusters that are internally redundant. The C^{-1} weighting naturally spreads the selection to cover the high-CKA region more uniformly.

---

## Summary Table

| Paper | Key Math Object | Direct Applicability | Non-obvious Insight |
|-------|----------------|---------------------|-------------------|
| Cortes 2012 | M matrix (= unnormalized CKA), QP for alignment maximization | The QP structure carries over to subset selection via continuous relaxation | μ* = M^{-1}a decorrelates — finds alignment contribution independent of redundancy |
| Zhou 2024 | CKA = cosine(vec(K_ic), vec(K_jc)), CKA ≥ 2 - N·MMD² | Reframes as "tightest cluster on unit sphere" in vectorized kernel space | CKA is invariant to embedding scale; the constant "+2" means we're minimizing disagreement, not maximizing agreement |
| Lu 2014 | CKA-based kernel clustering with alternating optimization | Alternating between centroid-update and selection naturally searches combinatorial space | Alternation can swap models in/out, escaping greedy's irreversibility |
| Cortes 2010 | Closed-form μ* = M^{-1}a, two-stage framework | Synthetic target + closed-form → fixed-point iteration for subset selection | Uniform weights → trivial with average target; need iterative refinement with current-subset target |
| Kandola 2002 | ICF: K ≈ LL^T, alignment via ‖L_i^T L_j‖²_F | CKA computable in O(n·d²) instead of O(n²) via centered embeddings | Low effective rank of CKA matrix constrains optimization landscape to d << 141 dimensions |
