#!/usr/bin/env python3
"""
Generate Real Plot Data for ICLR 2026 Re-Align Hackathon Reports

This script runs on a GPU-enabled cluster to extract actual embeddings from 
real vision models using the `timm` library, computes the actual Gram and 
CKA matrices, and generates the exact plots used in the challenge reports.

Dependencies:
    pip install torch torchvision timm datasets numpy matplotlib seaborn
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import timm
from datasets import load_dataset

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def get_images_from_hf(num_dogs=150, num_random=150):
    """Streams a small sample of images from HuggingFace to avoid downloading 150GB."""
    print("Loading ImageNet-1k validation split from HuggingFace (streaming)...")
    ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True, trust_remote_code=True)
    
    # ImageNet-1K class indices for dogs: 151-275
    dog_labels = set(range(151, 276))
    
    dog_images = []
    random_images = []
    
    # Note: Because we are streaming, we iterate until we find enough of both.
    # To get block-diagonal structure for random images, we'll sort them by label later
    random_images_with_labels = []
    
    for row in ds:
        label = row["label"]
        img = row["image"]
        if not hasattr(img, "convert"):
            continue
        img = img.convert("RGB")
        
        if label in dog_labels and len(dog_images) < num_dogs:
            dog_images.append(img)
            
        if len(random_images_with_labels) < num_random:
            random_images_with_labels.append((label, img))
            
        if len(dog_images) >= num_dogs and len(random_images_with_labels) >= num_random:
            break
            
    # Sort random images by label to ensure block-diagonal structure in the Gram matrix
    random_images_with_labels.sort(key=lambda x: x[0])
    random_images = [img for _, img in random_images_with_labels]
            
    print(f"Loaded {len(dog_images)} dog images and {len(random_images)} multi-class images.")
    return dog_images, random_images

def extract_features(model_name, images, device):
    """Extracts features (embeddings) using a timm model."""
    print(f"Extracting features using {model_name}...")
    model = timm.create_model(model_name, pretrained=True, num_classes=0).to(device)
    model.eval()
    
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    
    features = []
    with torch.no_grad():
        for img in images:
            tensor = transform(img).unsqueeze(0).to(device)
            feat = model(tensor).squeeze(0).cpu().numpy()
            features.append(feat)
            
    return np.array(features)

def compute_cka(X, Y):
    """Computes Linear CKA between two feature matrices."""
    K = X @ X.T
    L = Y @ Y.T
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H
    
    numerator = np.trace(Kc @ Lc)
    denominator = np.sqrt(np.trace(Kc @ Kc) * np.trace(Lc @ Lc))
    return numerator / denominator

def plot_real_red_team_insight(dogs, randoms):
    print("\n--- Generating Real Red Team Gram Matrix Plot ---")
    
    # Use ResNet as a proxy to visualize the texture-biased Gram matrices
    model_name = "resnet50.a1_in1k"
    
    features_dogs = extract_features(model_name, dogs, device)
    features_randoms = extract_features(model_name, randoms, device)
    
    # Compute Gram matrices
    K_multi = features_randoms @ features_randoms.T
    K_single = features_dogs @ features_dogs.T
    
    # Normalize uncentered Gram matrices for plotting aesthetics
    # We apply row/col normalization (cosine similarity) to make diagonal 1
    norm_multi = np.linalg.norm(features_randoms, axis=1, keepdims=True)
    K_multi = K_multi / (norm_multi @ norm_multi.T)
    
    norm_single = np.linalg.norm(features_dogs, axis=1, keepdims=True)
    K_single = K_single / (norm_single @ norm_single.T)
    
    # Centering operator
    n_samples = len(dogs)
    H = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
    
    Kc_multi = H @ K_multi @ H
    Kc_single = H @ K_single @ H
    
    # Save the data
    np.savez('red_team_real_matrices.npz', 
             K_multi=K_multi, Kc_multi=Kc_multi, 
             K_single=K_single, Kc_single=Kc_single)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    cmap = "viridis"
    
    sns.heatmap(K_multi, ax=axes[0,0], cmap=cmap, cbar=False, xticklabels=False, yticklabels=False)
    axes[0,0].set_title("Multi-Class Gram Matrix ($K$)\nReal ResNet50 embeddings (Sorted by class)", pad=15)
    
    sns.heatmap(Kc_multi, ax=axes[0,1], cmap=cmap, cbar=False, xticklabels=False, yticklabels=False)
    axes[0,1].set_title("Centered Multi-Class ($K_c = HKH$)\nBetween-class variance dominates", pad=15)
    
    sns.heatmap(K_single, ax=axes[1,0], cmap=cmap, cbar=False, xticklabels=False, yticklabels=False)
    axes[1,0].set_title("Single-Class Gram Matrix ($K$)\nReal ResNet50 embeddings (Dog Breeds)", pad=15)
    
    sns.heatmap(Kc_single, ax=axes[1,1], cmap=cmap, cbar=False, xticklabels=False, yticklabels=False)
    axes[1,1].set_title("Centered Single-Class ($K_c = HKH$)\nReveals fine-grained within-class variance", pad=15)
    
    plt.tight_layout()
    plot_path = 'red_team_real_gram_insight.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved real plot to {plot_path}")

def plot_real_blue_team_insight(images):
    print("\n--- Generating Real Blue Team CKA Matrix Plot ---")
    
    # Select a diverse subset of models to compute CKA quickly
    vits = [
        "deit3_base_patch16_224.fb_in1k",
        "beit_base_patch16_224.in22k_ft_in22k",
        "xcit_large_24_p16_224.fb_dist_in1k",
        "vit_base_patch16_224.augreg_in21k_ft_in1k",
        "cait_m36_384.fb_dist_in1k",
        "convit_base.fb_in1k"
    ]
    resnets = [
        "resnet50.a1_in1k",
        "resnet101.a1_in1k",
        "resnext50_32x4d.a1h_in1k",
        "seresnet152d.ra2_in1k",
        "densenet121.ra_in1k",
        "cspresnet50.ra_in1k"
    ]
    hybrids = [
        "mixer_b16_224.goog_in21k",
        "resmlp_12_224.fb_dino",
        "mambaout_base.in1k",
        "convmixer_768_32.in1k",
        "edgenext_base.in21k_ft_in1k",
        "poolformer_m36.sail_in1k"
    ]
    
    models = vits + resnets + hybrids
    all_features = []
    
    for m in models:
        feat = extract_features(m, images, device)
        all_features.append(feat)
        
    n_models = len(models)
    CKA = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            if i <= j:
                cka_val = compute_cka(all_features[i], all_features[j])
                CKA[i, j] = cka_val
                CKA[j, i] = cka_val
                
    # Save the data
    np.savez('blue_team_real_cka.npz', CKA=CKA, models=models)
                
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(CKA, cmap="magma", vmin=0.4, vmax=1.0, 
                cbar_kws={'label': 'Pairwise CKA'}, 
                xticklabels=False, yticklabels=False)
    
    n_vit = len(vits)
    n_resnet = len(resnets)
    n_other = len(hybrids)
    
    # Add delineating lines
    ax.axhline(n_vit, color='white', lw=2)
    ax.axvline(n_vit, color='white', lw=2)
    
    ax.axhline(n_vit+n_resnet, color='white', lw=2)
    ax.axvline(n_vit+n_resnet, color='white', lw=2)
    
    # Add text labels for the blocks
    ax.text(n_vit/2, n_vit/2, 'Vision Transformers\n(Dense Clique)', 
            color='white', ha='center', va='center', fontweight='bold', fontsize=12)
    ax.text(n_vit + n_resnet/2, n_vit + n_resnet/2, 'ResNets / CNNs', 
            color='white', ha='center', va='center', fontweight='bold', fontsize=12)
    ax.text(n_vit + n_resnet + n_other/2, n_vit + n_resnet + n_other/2, 'Hybrids / MLPs', 
            color='white', ha='center', va='center', fontweight='bold', fontsize=12)
    
    plt.title("Real CKA Similarity: Architecture as Primary Determinant", pad=20, fontsize=16)
    plt.tight_layout()
    plot_path = 'blue_team_real_cka_matrix.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved real plot to {plot_path}")

if __name__ == "__main__":
    print("Starting generation of real plot data...")
    # Get 150 dogs and 150 mixed-class images
    dogs, randoms = get_images_from_hf(num_dogs=150, num_random=150)
    
    plot_real_red_team_insight(dogs, randoms)
    
    # Re-use the mixed images to compute Blue Team CKA matrix
    # (CKA on mixed images shows the true architectural family clustering)
    plot_real_blue_team_insight(randoms)
    
    print("\nComplete! To include these in your report, replace the synthetic filenames with:")
    print("  - red_team_real_gram_insight.png")
    print("  - blue_team_real_cka_matrix.png")
