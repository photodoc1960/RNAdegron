import matplotlib
matplotlib.use('Agg')  # Set backend to non-interactive (no Tk)
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from Dataset_v7 import RNADataset, variable_length_collate_fn  # CRITICAL FIX: Updated import
from X_Network_v7 import RNADegformer  # CRITICAL FIX: Updated import

def plot_attention_with_bpp(attention_maps, bpp_targets, layer_idx, head_idx, seq_idx, threshold=0.5, save_path="attn_maps"):
    """
    Overlay attention with BPP base pairing structure.

    Parameters:
    - attention_maps: List[Tensor], one per layer, shape: [B, n_heads, L, L]
    - bpp_targets: Tensor, shape: [B, L, L] or [B, C, L, L] where C includes BPP
    - layer_idx: int, which encoder layer to use
    - head_idx: int, which attention head to visualize
    - seq_idx: int, which sequence in batch
    - threshold: float, base pair probability threshold
    - save_path: str, directory to save plots
    """
    os.makedirs(save_path, exist_ok=True)

    # Extract attention map
    attn_map = attention_maps[layer_idx][seq_idx, head_idx].detach().cpu().numpy()  # [L, L]

    # Handle bpp_targets: if [B, C, L, L], extract channel 0
    if bpp_targets.ndim == 4:
        bpp = bpp_targets[seq_idx, 0].detach().cpu().numpy()  # [L, L]
    else:
        bpp = bpp_targets[seq_idx].detach().cpu().numpy()  # [L, L]

    # BPP mask: base pairs with prob > threshold
    bp_mask = (bpp > threshold).astype(float)

    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(attn_map, cmap='Greys', cbar=True)
    plt.contour(bp_mask, levels=[0.5], colors='white', linewidths=0.75)
    plt.title(f"Attention L{layer_idx+1} H{head_idx+1} with BPP Overlay")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"attn_bpp_L{layer_idx+1}_H{head_idx+1}_S{seq_idx+1}.png"))
    plt.close()

def plot_all_heads_in_layer_with_bpp(
    attention_maps,
    bpp_targets,
    layer_idx,
    seq_idx=0,
    threshold=0.5,
    save_path="attn_bpp_grid"
):
    """
    Generate a grid of attention maps for all heads in one layer with BPP overlay.

    Parameters:
    - attention_maps: List[Tensor], one per layer, shape: [B, n_heads, L, L]
    - bpp_targets: Tensor, shape: [B, L, L] or [B, C, L, L] (channel 0 = BPP)
    - layer_idx: int, which transformer layer to use
    - seq_idx: int, index of sequence in batch
    - threshold: float, threshold for BPP pairing
    - save_path: str, where to save the figure
    """
    os.makedirs(save_path, exist_ok=True)

    attn_layer = attention_maps[layer_idx]  # [B, n_heads, L, L]
    n_heads = attn_layer.shape[1]
    L = attn_layer.shape[-1]

    # Get attention maps and BPP matrix
    bpp = bpp_targets[seq_idx]
    if bpp.ndim == 3:
        bpp = bpp[0]  # Channel 0 = BPP
    bpp = bpp.detach().cpu().numpy()
    bp_mask = (bpp > threshold).astype(float)

    fig, axs = plt.subplots(
        nrows=int(np.ceil(n_heads / 4)),
        ncols=4,
        figsize=(16, 3 * int(np.ceil(n_heads / 4)))
    )

    axs = axs.flatten()

    for head_idx in range(n_heads):
        attn = attn_layer[seq_idx, head_idx].detach().cpu().numpy()
        ax = axs[head_idx]
        sns.heatmap(attn, cmap="Greys", ax=ax, cbar=True)
        ax.contour(bp_mask, levels=[0.5], colors='red', linewidths=0.75)
        ax.set_title(f"Head {head_idx + 1}")
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")

    # Hide unused subplots if n_heads < grid slots
    for j in range(n_heads, len(axs)):
        axs[j].axis('off')

    plt.suptitle(f"Layer {layer_idx + 1} — All Heads", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_file = os.path.join(save_path, f"attn_grid_L{layer_idx + 1}_S{seq_idx + 1}.png")
    plt.savefig(out_file)
    plt.close()

def visualize_best_model(
        model_path='pretrain_weights/best_model.ckpt',
        data_path='./data/kaggle/test.json',
        save_path='visualizations_best_model',
        device=None,
        batch_size=1,
        mask_ratio=0.15,
        layer_idx=0,
        head_idx=0,
        seq_idx=0,
):
    """
    CRITICAL FIX: Updated model instantiation for simplified architecture compatibility.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    json_data = pd.read_json(data_path, lines=True)
    ids = json_data.id.to_list()

    dataset = RNADataset(json_data.sequence.to_list(), np.zeros(len(json_data)), ids,
                         np.arange(len(json_data)), bpp_path='./data/kaggle/new_sequences_bpps/', pad=True, k=5)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=variable_length_collate_fn)

    # CRITICAL FIX: Use simplified architecture without rinalmo_weights_path
    model = RNADegformer(
        ntoken=21, nclass=5, ninp=512, nhead=8, nhid=2048, nlayers=6,
        kmer_aggregation=False, kmers=[1], stride=1, dropout=0.1,
        pretrain=True, rinalmo_weights_path=None  # SIMPLIFIED ARCHITECTURE
    ).to(device)

    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    os.makedirs(save_path, exist_ok=True)

    # Single batch
    data = next(iter(dataloader))
    embeddings = data['embedding'].to(device)
    bpp = data['bpp'].to(device)
    src_mask = data['src_mask'].to(device)

    batch_size, seq_len, _ = embeddings.shape
    mask_positions = torch.rand(batch_size, seq_len, device=device) < mask_ratio
    masked_embeddings = embeddings.clone()
    masked_embeddings[mask_positions] = 0.0

    with torch.no_grad():
        # Extract structural features for forward pass
        deltaG = data['deltaG'].to(device)
        graph_dist = data['graph_dist'].to(device)
        nearest_p = data['nearest_p'].to(device)
        nearest_up = data['nearest_up'].to(device)
        
        # CRITICAL FIX: Updated forward pass signature for simplified architecture
        outputs, attention_maps, _ = model(
            masked_embeddings, bpp, src_mask, deltaG, graph_dist, nearest_p, nearest_up
        )

        plot_all_heads_in_layer_with_bpp(
            attention_maps=attention_maps,
            bpp_targets=bpp[:, 0],
            layer_idx=layer_idx,
            seq_idx=seq_idx,
            threshold=0.5,
            save_path=save_path
        )

        plot_attention_with_bpp(
            attention_maps=attention_maps,
            bpp_targets=bpp[:, 0],
            layer_idx=layer_idx,
            head_idx=head_idx,
            seq_idx=seq_idx,
            threshold=0.5,
            save_path=save_path
        )

    print(f"[Visualization] Saved attention maps to {save_path}")

def plot_loss_curve(
    csv_log_path='logs/pretrain.csv',
    save_path='visualizations_best_model/loss_curve.png'
):
    """
    Plot train and validation loss curves from CSV log.

    Args:
        csv_log_path (str): Path to the CSV training log.
        save_path (str): Path to save the loss curve figure.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load training log
    df = pd.read_csv(csv_log_path)

    epochs = df['epoch']
    train_loss = df['train_loss']
    val_loss = df['val_loss']

    # Plot
    plt.figure(figsize=(8,6))
    plt.plot(epochs, train_loss, label='Train Loss', linewidth=2)
    plt.plot(epochs, val_loss, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Pretraining Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"[Visualization] Saved loss curve to {save_path}")

def plot_all_heads_in_layer(
    attention_maps,
    layer_idx,
    seq_idx=0,
    save_path="attn_grid"
):
    """
    Plot attention maps for all heads in a given layer (no BPP overlay).

    Parameters:
        attention_maps (List[Tensor]): One per layer, shape: [B, n_heads, L, L]
        layer_idx (int): Which layer to visualize
        seq_idx (int): Which sequence in batch
        save_path (str): Directory to save the output plot
    """
    os.makedirs(save_path, exist_ok=True)
    attn_layer = attention_maps[layer_idx]  # [B, n_heads, L, L]
    n_heads = attn_layer.shape[1]

    fig, axs = plt.subplots(
        nrows=int(np.ceil(n_heads / 4)),
        ncols=4,
        figsize=(16, 3 * int(np.ceil(n_heads / 4)))
    )
    axs = axs.flatten()

    for head_idx in range(n_heads):
        attn = attn_layer[seq_idx, head_idx].detach().cpu().numpy()
        ax = axs[head_idx]
        sns.heatmap(attn, cmap='Greys', ax=ax, cbar=True)
        ax.set_title(f"Head {head_idx + 1}")
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")

    for j in range(n_heads, len(axs)):
        axs[j].axis('off')

    plt.suptitle(f"Layer {layer_idx + 1} — All Attention Heads", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_file = os.path.join(save_path, f"attn_only_L{layer_idx + 1}_S{seq_idx + 1}.png")
    plt.savefig(out_file)
    plt.close()

def plot_bpp_matrix(
    bpp_targets,
    seq_idx=0,
    threshold=0.5,
    save_path="bpp_only"
):
    """
    Plot the BPP matrix alone for a given sequence.

    Parameters:
        bpp_targets (Tensor): Shape [B, L, L] or [B, C, L, L] (channel 0 = BPP)
        seq_idx (int): Index of sequence in batch
        threshold (float): Threshold to visualize binary mask if desired
        save_path (str): Output directory
    """
    os.makedirs(save_path, exist_ok=True)

    if bpp_targets.ndim == 4:
        bpp = bpp_targets[seq_idx, 0].detach().cpu().numpy()
    else:
        bpp = bpp_targets[seq_idx].detach().cpu().numpy()

    plt.figure(figsize=(6, 5))
    sns.heatmap(bpp, cmap='Greys', cbar=True)
    plt.title(f"BPP Matrix for Seq {seq_idx + 1}")
    plt.xlabel("Position")
    plt.ylabel("Position")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"bpp_only_S{seq_idx + 1}.png"))
    plt.close()

def plot_attention_with_structure_overlay(
    attention_maps,
    loop_labels,
    layer_idx,
    head_idx,
    seq_idx=0,
    save_path="attn_struct",
    structure_colors=None
):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import to_rgb
    import numpy as np
    import os

    os.makedirs(save_path, exist_ok=True)

    attn = attention_maps[layer_idx][seq_idx, head_idx].detach().cpu().numpy()

    if loop_labels.ndim == 3:
        loop_data = loop_labels[seq_idx, :, 2]
    else:
        loop_data = loop_labels[seq_idx]
    loop_data = loop_data.detach().cpu().numpy()
    L = attn.shape[0]

    default_colors = {
        0: "#e6194b",  # B
        1: "#aaaaaa",  # E
        2: "#4363d8",  # H
        3: "#f58231",  # I
        4: "#911eb4",  # M
        5: "#000000",  # S
        6: "#46f0f0"   # X — will be ignored
    }
    colors = structure_colors or default_colors

    fig, ax = plt.subplots(figsize=(10, 10))

    sns.heatmap(attn, cmap="Greys", cbar=True, ax=ax, xticklabels=10, yticklabels=10)

    for i in range(L):
        for j in range(L):
            s_id = int(loop_data[i])
            if s_id == 6 or s_id != int(loop_data[j]):
                continue
            color = to_rgb(colors[s_id])
            ax.add_patch(plt.Rectangle(
                (j, i), 1, 1,
                facecolor=color,
                edgecolor='none',
                alpha=0.25,
                zorder=5
            ))

    ax.set_title(f"Attention L{layer_idx+1} H{head_idx+1} with Structure-Matched Overlays")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")

    handles = [
        mpatches.Patch(color=clr, label=lbl)
        for lbl, clr in {
            "B": colors[0], "E": colors[1], "H": colors[2], "I": colors[3],
            "M": colors[4], "S": colors[5]
        }.items()
    ]
    fig.legend(handles=handles, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.04), frameon=False, title="Structure Type")

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    base = f"attn_struct_L{layer_idx+1}_H{head_idx+1}_S{seq_idx+1}"
    plt.savefig(os.path.join(save_path, base + ".png"), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(save_path, base + ".pdf"), bbox_inches='tight')
    plt.close()
