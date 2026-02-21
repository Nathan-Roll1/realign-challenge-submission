import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

def plot_red_team_insight():
    # Generate synthetic Gram Matrices to illustrate the Red Team core insight
    n_samples = 150
    n_classes_multi = 5
    samples_per_class = n_samples // n_classes_multi
    
    # 1. Multi-class uncentered Gram matrix (Block diagonal)
    K_multi = np.zeros((n_samples, n_samples))
    for i in range(n_classes_multi):
        start = i * samples_per_class
        end = (i + 1) * samples_per_class
        K_multi[start:end, start:end] = 0.8  # Strong between-class signal
    # Add within-class noise
    K_multi += np.random.normal(0, 0.1, (n_samples, n_samples))
    K_multi = (K_multi + K_multi.T) / 2
    np.fill_diagonal(K_multi, 1.0)
    
    # 2. Single-class uncentered Gram matrix (No blocks, just within-class variance)
    K_single = np.random.normal(0.5, 0.15, (n_samples, n_samples))
    K_single = (K_single + K_single.T) / 2
    np.fill_diagonal(K_single, 1.0)
    
    # Centering operator
    H = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
    
    # Center the matrices
    Kc_multi = H @ K_multi @ H
    Kc_single = H @ K_single @ H
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    cmap = "viridis"
    
    sns.heatmap(K_multi, ax=axes[0,0], cmap=cmap, cbar=False, xticklabels=False, yticklabels=False)
    axes[0,0].set_title("Multi-Class Gram Matrix ($K$)\nStrong between-class block structure", pad=15)
    
    sns.heatmap(Kc_multi, ax=axes[0,1], cmap=cmap, cbar=False, xticklabels=False, yticklabels=False)
    axes[0,1].set_title("Centered Multi-Class ($K_c = HKH$)\nBlock structure survives and dominates", pad=15)
    
    sns.heatmap(K_single, ax=axes[1,0], cmap=cmap, cbar=False, xticklabels=False, yticklabels=False)
    axes[1,0].set_title("Single-Class Gram Matrix ($K$)\nUniform semantic category (e.g., Dog Breeds)", pad=15)
    
    sns.heatmap(Kc_single, ax=axes[1,1], cmap=cmap, cbar=False, xticklabels=False, yticklabels=False)
    axes[1,1].set_title("Centered Single-Class ($K_c = HKH$)\nReveals fine-grained architectural divergence", pad=15)
    
    plt.tight_layout()
    plt.savefig('red_team_gram_insight.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_blue_team_insight():
    # Generate synthetic CKA similarity matrix for different model families
    # 3 families: ViT, ResNet, Other (MLP/Hybrid)
    n_vit = 20
    n_resnet = 15
    n_other = 15
    n_total = n_vit + n_resnet + n_other
    
    CKA = np.zeros((n_total, n_total))
    
    # ViT-ViT block (Highly aligned)
    CKA[:n_vit, :n_vit] = np.random.normal(0.85, 0.05, (n_vit, n_vit))
    
    # ResNet-ResNet block (Moderately aligned)
    CKA[n_vit:n_vit+n_resnet, n_vit:n_vit+n_resnet] = np.random.normal(0.70, 0.08, (n_resnet, n_resnet))
    
    # Other-Other block
    CKA[n_vit+n_resnet:, n_vit+n_resnet:] = np.random.normal(0.55, 0.1, (n_other, n_other))
    
    # Cross-family interactions (Low alignment)
    # ViT - ResNet
    cross_vr = np.random.normal(0.40, 0.08, (n_vit, n_resnet))
    CKA[:n_vit, n_vit:n_vit+n_resnet] = cross_vr
    CKA[n_vit:n_vit+n_resnet, :n_vit] = cross_vr.T
    
    # ViT - Other
    cross_vo = np.random.normal(0.35, 0.08, (n_vit, n_other))
    CKA[:n_vit, n_vit+n_resnet:] = cross_vo
    CKA[n_vit+n_resnet:, :n_vit] = cross_vo.T
    
    # ResNet - Other
    cross_ro = np.random.normal(0.45, 0.08, (n_resnet, n_other))
    CKA[n_vit:n_vit+n_resnet, n_vit+n_resnet:] = cross_ro
    CKA[n_vit+n_resnet:, n_vit:n_vit+n_resnet] = cross_ro.T
    
    # Symmetrize and bound
    CKA = (CKA + CKA.T) / 2
    np.fill_diagonal(CKA, 1.0)
    CKA = np.clip(CKA, 0, 1)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(CKA, cmap="magma", vmin=0.2, vmax=1.0, 
                cbar_kws={'label': 'Pairwise CKA'}, 
                xticklabels=False, yticklabels=False)
    
    # Add delineating lines
    ax.axhline(n_vit, color='white', lw=2)
    ax.axvline(n_vit, color='white', lw=2)
    
    ax.axhline(n_vit+n_resnet, color='white', lw=2)
    ax.axvline(n_vit+n_resnet, color='white', lw=2)
    
    # Add text labels for the blocks
    ax.text(n_vit/2, n_vit/2, 'Vision Transformers\n(Dense Clique)', 
            color='white', ha='center', va='center', fontweight='bold', fontsize=12)
    ax.text(n_vit + n_resnet/2, n_vit + n_resnet/2, 'ResNets', 
            color='white', ha='center', va='center', fontweight='bold', fontsize=12)
    ax.text(n_vit + n_resnet + n_other/2, n_vit + n_resnet + n_other/2, 'Hybrids/MLPs', 
            color='white', ha='center', va='center', fontweight='bold', fontsize=12)
    
    plt.title("Architecture as the Primary Determinant of Representational Similarity", pad=20, fontsize=16)
    plt.tight_layout()
    plt.savefig('blue_team_cka_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_red_team_insight()
    plot_blue_team_insight()
    print("Plots generated successfully.")
