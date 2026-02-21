# ICLR 2026 Re-Align Hackathon - Complete Reconnaissance Report

**Date:** February 16, 2026
**Source:** https://huggingface.co/spaces/representational-alignment/iclr2026-realign-challenge

---

## 1. LEADERBOARD RESULTS (Live as of Feb 16, 2026)

### Blue Team Leaderboard (Higher is Better - Mean Pairwise CKA)
| Rank | Submitter | Score  | Submitted              |
|------|-----------|--------|------------------------|
| 1    | baseline  | 0.5557 | 2026-02-12T00:53:15Z   |
| 2    | km        | 0.5421 | 2026-02-13T08:10:17Z   |
| 3    | km        | 0.5421 | 2026-02-13T08:31:47Z   |

**Total blue submissions loaded: 3**

### Red Team Leaderboard (Higher is Better - 1 - avg CKA = divergence)
| Rank | Submitter | Score  | Submitted              |
|------|-----------|--------|------------------------|
| 1    | baseline  | 0.4790 | 2026-02-12T01:18:36Z   |

**Total red submissions loaded: 1**

### Key Takeaway
- The baseline Blue Team score is **0.5557** (mean pairwise CKA across 20 models).
- The baseline Red Team score is **0.479** (1 - avg CKA across fixed model set on 1000 selected images).
- Very few submissions so far - the challenge appears early stage.

---

## 2. CHALLENGE RULES SUMMARY

### Blue Team - Maximize Alignment
- **Select exactly 20 models** from the registry (~147 models available in `one_model_per_family.json`)
- Score = **mean pairwise CKA** across all 20 selected models
- Higher CKA = better (models have more aligned representations)
- Models must include correct `model_name` AND `layer_name` (validated against registry)
- Embeddings extracted from a **fixed set of held-out images** (secret, server-side)

### Red Team - Maximize Divergence
- **Select exactly 1000 images** from the stimuli catalog
- Available datasets: `imagenet_val` and `objectnet`
- Score = **1 - avg CKA** across a **fixed set of models** (secret, server-side)
- Higher divergence = better
- The full stimuli catalog has ~150,000+ entries (9.49 MB JSONL file)

### Important
- Must also submit a Markdown report on OpenReview
- Submissions via HuggingFace private datasets
- Processing takes ~20 minutes

---

## 3. CKA COMPUTATION DETAILS

### Core CKA Implementation (`src/cka/compute.py`)

The scoring uses **Linear CKA with biased HSIC** by default:

```python
def linear_cka(x, y, *, unbiased=False, eps=1e-6):
    # x, y are 2D arrays: [num_samples, dim], converted to float64
    k = x @ x.T      # Linear kernel for x
    l = y @ y.T      # Linear kernel for y
    hsic_fn = hsic_unbiased if unbiased else hsic_biased
    hsic_kk = hsic_fn(k, k)
    hsic_ll = hsic_fn(l, l)
    hsic_kl = hsic_fn(k, l)
    denom = sqrt(hsic_kk * hsic_ll)
    return hsic_kl / (denom + eps)
```

### Biased HSIC (default):
```python
def hsic_biased(k, l):
    m = k.shape[0]
    h = eye(m) - (1.0 / m)   # Centering matrix
    return trace(k @ h @ l @ h)
```

### Feature-space equivalent:
```python
def linear_cka_feature(x, y, eps=1e-6):
    x = x - x.mean(axis=0)  # Center features
    y = y - y.mean(axis=0)
    numerator = norm(x.T @ y, 'fro') ** 2
    denom = norm(x.T @ x, 'fro') * norm(y.T @ y, 'fro')
    return numerator / (denom + eps)
```

### Key details:
- Arrays converted to **float64** before computation
- Epsilon = **1e-6** for numerical stability
- Biased HSIC is default (matches original Kornblith et al. 2019 paper)
- Reference: https://arxiv.org/abs/1905.00414

---

## 4. SCORING PIPELINE

### Blue Team Scoring (`src/hackathon/scoring.py`)
```
1. For each pair of selected models (C(20,2) = 190 pairs):
   - Get embeddings for both models on held-out images
   - Compute linear_cka(embeddings_a, embeddings_b)
2. Score = mean of all 190 pairwise CKA values
```

### Red Team Scoring
```
1. Use the FIXED model set (secret, loaded server-side from Modal volume)
2. For each model, get embeddings on the 1000 SUBMITTED images only
3. For each pair of models:
   - Compute linear_cka on the filtered embeddings
4. avg_cka = mean of all pairwise CKA values
5. Score = 1.0 - avg_cka
```

### Modal Backend (Production)
- When `HACKATHON_MODAL_ENABLE=true`, scoring goes through Modal
- Modal app name: `iclr2026-eval` (deployed from private `iclr2026-eval-backend` repo)
- Key functions called: `score_blue_submission`, `score_red_submission`
- Embedding extraction function: `extract_embeddings_s3` (can use S3 storage)
- **Blue heldout images are ALWAYS loaded server-side (secret)**
- **Red model registry is ALWAYS loaded server-side (secret)**
- Embeddings cached per model/layer/dataset version on Modal volume `iclr2026-embeddings`

---

## 5. MODEL REGISTRY

### `configs/one_model_per_family.json` - The 147 Available Models

All models are from **timm** (PyTorch Image Models). Here is the complete list:

```
aimv2_1b_patch14_224.apple_pt
bat_resnext26ts.ch_in1k
beit_base_patch16_224.in22k_ft_in22k
beitv2_base_patch16_224.in1k_ft_in1k
botnet26t_256.c1_in1k
caformer_b36.sail_in1k
cait_m36_384.fb_dist_in1k
coat_lite_medium.in1k
coatnet_0_rw_224.sw_in1k
coatnext_nano_rw_224.sw_in1k
convformer_b36.sail_in1k
convit_base.fb_in1k
convmixer_1024_20_ks9_p14.in1k
convnext_atto.d2_in1k
convnextv2_atto.fcmae
crossvit_15_240.in1k
cs3darknet_focus_l.c2ns_in1k
cspdarknet53.ra_in1k
cspresnet50.ra_in1k
cspresnext50.ra_in1k
darknet53.c2ns_in1k
darknetaa53.c2ns_in1k
davit_base.msft_in1k
deit3_base_patch16_224.fb_in1k
densenet121.ra_in1k
densenetblur121d.ra_in1k
dla102.in1k
dm_nfnet_f0.dm_in1k
dpn107.mx_in1k
eca_botnext26ts_256.c1_in1k
ecaresnet101d.miil_in1k
ecaresnetlight.miil_in1k
edgenext_base.in21k_ft_in1k
efficientformer_l1.snap_dist_in1k
efficientformerv2_l.snap_dist_in1k
efficientnet_b0.ra4_e3600_r224_in1k
efficientnetv2_rw_m.agc_in1k
efficientvit_b0.r224_in1k
ese_vovnet19b_dw.ra_in1k
eva02_base_patch14_224.mim_in22k
fastvit_ma36.apple_dist_in1k
fbnetc_100.rmsp_in1k
fbnetv3_b.ra2_in1k
flexivit_base.1000ep_in21k
focalnet_base_lrf.ms_in1k
gc_efficientnetv2_rw_t.agc_in1k
gcresnet33ts.ra2_in1k
gcresnext26ts.ch_in1k
gcvit_base.in1k
gernet_l.idstcv_in1k
ghostnet_100.in1k
ghostnetv2_100.in1k
gmixer_24_224.ra3_in1k
gmlp_s16_224.ra3_in1k
halo2botnet50ts_256.a1h_in1k
halonet26t.a1h_in1k
haloregnetz_b.ra3_in1k
hardcorenas_a.miil_green_in1k
hgnet_base.ssld_in1k
hgnetv2_b0.ssld_stage1_in22k_in1k
hiera_base_224.mae
hrnet_w18.ms_aug_in1k
inception_next_atto.sail_in1k
lambda_resnet26rpt_256.c1_in1k
lamhalobotnet50ts_256.a1h_in1k
lcnet_050.ra2_in1k
legacy_senet154.in1k
levit_128.fb_dist_in1k
mambaout_base.in1k
maxvit_base_tf_224.in1k
maxxvit_rmlp_nano_rw_256.sw_in1k
maxxvitv2_nano_rw_256.sw_in1k
mixer_b16_224.goog_in21k
mixnet_l.ft_in1k
mnasnet_100.rmsp_in1k
mobilenet_edgetpu_v2_m.ra4_e3600_r224_in1k
mobilenetv1_100.ra4_e3600_r224_in1k
mobileone_s0.apple_in1k
mobilevit_s.cvnets_in1k
mobilevitv2_050.cvnets_in1k
mvitv2_base.fb_in1k
nasnetalarge.tf_in1k
nest_base_jx.goog_in1k
nextvit_base.bd_in1k
nf_regnet_b1.ra2_in1k
nfnet_l0.ra2_in1k
pit_b_224.in1k
pnasnet5large.tf_in1k
poolformer_m36.sail_in1k
poolformerv2_m36.sail_in1k
pvt_v2_b0.in1k
rdnet_base.nv_in1k
regnetv_040.ra3_in1k
regnetx_002.pycls_in1k
regnety_002.pycls_in1k
regnetz_040.ra3_in1k
repghostnet_050.in1k
repvgg_a0.rvgg_in1k
repvit_m0_9.dist_300e_in1k
res2net101_26w_4s.in1k
resmlp_12_224.fb_dino
resnest101e.in1k
resnet101.a1_in1k
resnetaa101d.sw_in12k
resnetblur50.bt_in1k
resnetrs101.tf_in1k
resnetv2_101.a1h_in1k
resnext101_32x16d.fb_ssl_yfcc100m_ft_in1k
rexnet_100.nav_in1k
rexnetr_200.sw_in12k
sam2_hiera_base_plus.fb_r896
samvit_base_patch16.sa1b
sebotnet33ts_256.a1h_in1k
sehalonet33ts.ra2_in1k
selecsls42b.in1k
semnasnet_075.rmsp_in1k
senet154.gluon_in1k
sequencer2d_l.in1k
seresnet152d.ra2_in1k
seresnext101_32x4d.gluon_in1k
seresnextaa101d_32x8d.ah_in1k
skresnet18.ra_in1k
skresnext50_32x4d.ra_in1k
spnasnet_100.rmsp_in1k
swin_base_patch4_window12_384.ms_in1k
swinv2_base_window12_192.ms_in22k
test_byobnet.r160_in1k
tf_efficientnet_b0.aa_in1k
tiny_vit_11m_224.dist_in22k
tinynet_a.in1k
tnt_s_patch16_224
tresnet_l.miil_in1k
twins_pcpvt_base.in1k
vgg11.tv_in1k
visformer_small.in1k
vit_base_mci_224.apple_mclip
vitamin_base_224.datacomp1b_clip
volo_d1_224.sail_in1k
wide_resnet101_2.tv2_in1k
xception41.tf_in1k
xcit_large_24_p16_224.fb_dist_in1k
```

### Model Registry Spec Structure (`blue_team_model_registry.json`)
Each entry in the ~50.9 kB registry contains:
```json
{
    "model_name": "resnet101.a1_in1k",
    "source": "timm",
    "weights": "imagenet",
    "layer": "global_pool",          // REQUIRED in submission
    "embedding": "flatten",
    "preprocess": {
        "resize": 256,
        "crop": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    }
}
```

### Layer Detection Priority (from `generate_blue_registry.py`):
1. `head.global_pool` (hybrid/newer architectures)
2. `global_pool` (standard CNNs)
3. `pooling` (ConvMixer)
4. `fc_norm` (EVA, BEiT, AIMv2)
5. `norm` (ViT/transformer final normalization)
6. `norm.1` (CrossViT)
7. `norm4` (CoaT)
8. `head.norm`
9. `stages.3.norm` (PVT v2)
10. `head.head.bn` (RepViT)
11. `head.bn` (LeViT)
12. `head` (last resort)
13. `avgpool` (torchvision-style)

---

## 6. STIMULI CATALOG STRUCTURE

### `red_team_stimuli_catalog.jsonl` (9.49 MB, ~150,000+ entries)
Format: One JSON object per line
```json
{"dataset_name": "imagenet_val", "image_identifier": "ILSVRC2012_val_00000001.JPEG"}
```

### Available Datasets:
1. **imagenet_val** - ImageNet ILSVRC2012 validation set
   - Identifiers: `ILSVRC2012_val_XXXXXXXX.JPEG` (flat file names, no subdirectories)
   - ~50,000 images
2. **objectnet** - ObjectNet test set
   - Identifiers: `objectnet-1.0/images/{category}/{hash}.png`
   - Categories include: squeeze_bottle, removable_blade, t-shirt, shovel, coin_money, vase, cutting_board, bottle_cap, calendar, detergent, bread_knife, baseball_glove, keyboard, running_shoe, power_cable, stapler, comb, cellphone_case, headphones_over_ear, tv, blanket, dress, butchers_knife, computer_mouse, etc.

### `red_team_stimuli_catalog_1000.jsonl` (95.1 kB, 1000 entries)
This is a **pre-sampled set of 1000 stimuli** mixing both objectnet and imagenet_val.
Likely used as the baseline submission or testing reference.
Mix is roughly ~50/50 objectnet and imagenet_val images.

---

## 7. FIXED SET OF MODELS (Red Team Evaluation)

**CRITICAL: The Red Team's "fixed set of models" is SECRET.**

From `modal_client.py`:
```python
# The red team model registry is always loaded server-side from the
# Modal volume (secret — never sent from the public Space).
```

The code shows:
- `HACKATHON_RED_MODEL_REGISTRY` env var exists but is server-side only
- The backend loads it from the Modal volume, never exposed publicly
- The code references `RED_MODEL_REGISTRY_ENV = "HACKATHON_RED_MODEL_REGISTRY"`

**Best guess:** The Red Team fixed models likely come from the same `one_model_per_family.json` list (all 147 models), or a curated subset. The evaluation contract states "Load the model registry (full evaluation model set)."

---

## 8. SUBMISSION SCRIPTS

### `submit_blue_hf_dataset.py`
- Reads a JSONL file (default: `test_submissions/blue_submission.jsonl`)
- Each line: `{"model_name": "...", "layer_name": "..."}`
- Pushes to HF dataset as private
- Default dataset: `bkhmsi/test-realign-hackathon-blue-team`

### `submit_red_hf_dataset.py`
- Reads a JSONL file (default: `test_submissions/red_submission.jsonl`)
- Each line: `{"dataset_name": "...", "image_identifier": "..."}`
- Pushes to HF dataset as private
- Default dataset: `bkhmsi/test-realign-hackathon-red-team`

### Submission Flow:
1. Create HF dataset with correct columns
2. Paste dataset link in the Gradio app
3. Click "Generate JSON" to parse the dataset into JSON
4. Click "Submit" to score via Modal backend
5. Results appear on leaderboard after ~20 minutes

---

## 9. EVALUATION CONTRACT (`docs/evaluation_contract.md`)

### Blue Team Evaluation:
1. Load the stimuli catalog (full evaluation set) — **SECRET held-out images**
2. For each submitted model, run forward pass on ALL stimuli and extract embeddings
3. Compute mean pairwise linear CKA across submitted models

### Red Team Evaluation:
1. Load the model registry (full evaluation model set) — **SECRET model set**
2. For each model, run forward pass on submitted stimuli and extract embeddings
3. Compute mean pairwise linear CKA across all models
4. Score = 1 - avg CKA

### Embedding Extraction Requirements:
- `model.eval()` and `torch.no_grad()` for all forward passes
- Deterministic settings (seed, disable dropout)
- Embeddings must be 2D arrays: [num_samples, dim]
- If a layer produces spatial features, apply the registry's embedding strategy (flatten)

---

## 10. HIDDEN/SECRET INFORMATION

Things we **cannot** access:
1. **Blue Team held-out images** - The actual images used for evaluation are loaded server-side from Modal volume
2. **Red Team fixed model set** - The models used for Red Team evaluation are loaded server-side
3. **Modal backend code** - Lives in a private `iclr2026-eval-backend` repository
4. **Full blue_team_model_registry.json** - 50.9 kB file with all model specs including layers (failed to fetch directly, but structure is known)

Things we **can** access:
1. Full stimuli catalog (all valid images for Red Team)
2. The one_model_per_family.json (147 model names for Blue Team)
3. Complete CKA computation code
4. Complete scoring logic
5. Validation logic and constraints
6. Leaderboard data

---

## 11. REPO STRUCTURE

```
iclr2026-realign-challenge/
├── app.py                          (23 kB - main Gradio app)
├── README.md                       (4.26 kB)
├── AGENTS.md                       (4.31 kB - MIT cluster guide)
├── Makefile
├── pyproject.toml
├── requirements.txt
├── environment.yml
├── configs/
│   ├── blue_team_model_registry.json   (50.9 kB - full model specs)
│   ├── one_model_per_family.json       (4.13 kB - 147 model names)
│   ├── dataset_roots.example.json      (85 B)
│   ├── red_team_stimuli_catalog.jsonl  (9.49 MB - full catalog)
│   └── red_team_stimuli_catalog_1000.jsonl (95.1 kB - 1000-sample subset)
├── cka-data/
├── docs/
│   ├── evaluation_contract.md
│   └── storage_layout.md
├── hackathon-data/
│   ├── blue_submissions.json
│   └── red_submissions.json
├── scripts/
│   ├── blue_family_smoke_test.py
│   ├── blue_team_submit.py
│   ├── download_test_sets.py        (25.7 kB)
│   ├── generate_blue_registry.py    (6.88 kB)
│   ├── pipeline_smoke_test.py
│   ├── red_team_smoke_test.py
│   ├── run_local.sh
│   ├── smoke_test_registry.py
│   ├── smoke_test_submission.py     (9.49 kB)
│   ├── submit_blue_hf_dataset.py
│   ├── submit_red_hf_dataset.py
│   ├── validate_submission.py
│   └── verify_cka.py
├── src/
│   ├── about.py                     (7.69 kB - UI text)
│   ├── cka/
│   │   ├── __init__.py
│   │   ├── compute.py               (2.21 kB - CKA math)
│   │   ├── embeddings.py            (974 B - dummy embeddings)
│   │   └── storage.py               (1.24 kB)
│   ├── display/
│   │   └── css_html_js.py
│   └── hackathon/
│       ├── __init__.py
│       ├── data.py                  (4.79 kB - dummy data)
│       ├── modal_client.py          (5.51 kB - Modal backend client)
│       ├── scoring.py               (4.84 kB - scoring logic)
│       ├── storage.py               (4.92 kB - submission persistence)
│       └── validation.py            (8.54 kB - input validation)
└── test_submissions/
```

---

## 12. STRATEGIC OBSERVATIONS

### For Blue Team:
- You need to find 20 models from the 147 available that are most similar
- The baseline achieves 0.5557 mean pairwise CKA
- Strategy: Group models by architectural family and pick models within the same family
- The dummy data code reveals family groupings matter (ViT family, ResNet family, etc.)
- Models from the same architectural family (e.g., all ViT variants) should have higher CKA

### For Red Team:
- You need to find 1000 images that maximize disagreement between models
- The baseline achieves 0.479 (1 - avg CKA)
- Available images: imagenet_val (~50K) and objectnet
- ObjectNet images are photographed in unusual contexts/orientations - likely cause more model disagreement
- Strategy: Find images where different model architectures "see" very different things
- The 1000-sample catalog mixes objectnet and imagenet_val roughly 50/50

### Key Technical Insight:
- The evaluation uses BIASED Linear CKA (not unbiased)
- Features are NOT mean-centered in the kernel-space computation (though they are in the feature-space equivalent)
- Embeddings are extracted from specific layers (pool/norm layers, not final classifier)
- All models are from timm with ImageNet pretraining

---

## 13. CONTRIBUTORS / ORGANIZERS

- **kushinm** (Kushin Mukherjee) - recent commits, polish instructions
- **siddsuresh97** (Siddharth Suresh) - initial commit
- **bkhmsi** - test submission datasets
- Workshop: ICLR 2026 Re-Align Workshop on Representational Alignment (3rd edition)
