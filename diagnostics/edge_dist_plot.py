"""
Edge Distribution Plot Generator

Analyzes edge distributions across all batch_with_fits.csv files.
Computes edge = probability - market_yes_price for three probability estimates:
  - p_model_fit: Fitted logistic model probability
  - p_model_cal: Calibrated model probability
  - p_real_mc: Raw Monte Carlo probability

Usage:
    python edge_dist_plot.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from datetime import datetime

# Hardcoded paths
BASE_DIR = Path(r"C:\Users\Kieran Trythall\Documents\Trading\Prediction Market Contract Pricing\V2 BTC Contract Pricing")
FITTED_DIR = BASE_DIR / "backtested_probabilities" / "fitted"
PLOTS_DIR = BASE_DIR / "plots"

def load_all_batches() -> pd.DataFrame:
    """Load and concatenate all batch_with_fits.csv files."""
    pattern = str(FITTED_DIR / "*" / "batch_with_fits.csv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No batch_with_fits.csv files found in {FITTED_DIR}")
        return pd.DataFrame()
    
    print(f"Found {len(files)} batch files")
    
    all_data = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["source_file"] = Path(f).parent.name
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(combined)} total rows")
    return combined


def compute_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Compute edge for each probability column."""
    price_col = None
    for col in ["market_price", "market_yes_price", "Polymarket_Price", "polymarket_price"]:
        if col in df.columns:
            price_col = col
            break
    
    if price_col is None:
        raise ValueError(f"No market price column found. Available: {df.columns.tolist()}")
    
    print(f"Using price column: {price_col}")
    
    edges = pd.DataFrame()
    prob_columns = {
        "p_model_fit": "Model Fit",
        "p_model_cal": "Model Calibrated", 
        "p_real_mc": "Raw MC",
    }
    
    for col, label in prob_columns.items():
        if col in df.columns:
            edge = pd.to_numeric(df[col], errors="coerce") - pd.to_numeric(df[price_col], errors="coerce")
            edges[label] = edge
            valid_count = edge.dropna().shape[0]
            print(f"  {label} ({col}): {valid_count} valid edges")
        else:
            print(f"  {label} ({col}): NOT FOUND")
    
    return edges


def plot_edge_distributions(edges: pd.DataFrame, output_path: Path):
    """Create histogram and KDE plots of edge distributions."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    colors = {
        "Model Fit": "#2ecc71",
        "Model Calibrated": "#3498db",
        "Raw MC": "#e74c3c",
    }
    
    ax1 = axes[0]
    for col in edges.columns:
        data = edges[col].dropna()
        if len(data) > 0:
            ax1.hist(data, bins=100, alpha=0.5, label=f"{col} (n={len(data):,})", 
                    color=colors.get(col, "gray"), density=True)
    
    ax1.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax1.set_xlabel("Edge (Model Prob - Market Price)")
    ax1.set_ylabel("Density")
    ax1.set_title("Edge Distribution by Probability Source")
    ax1.legend()
    ax1.set_xlim(-0.5, 0.5)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    edge_data = [edges[col].dropna() for col in edges.columns]
    bp = ax2.boxplot(edge_data, tick_labels=edges.columns, patch_artist=True)
    
    for patch, col in zip(bp['boxes'], edges.columns):
        patch.set_facecolor(colors.get(col, "gray"))
        patch.set_alpha(0.6)
    
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax2.set_ylabel("Edge")
    ax2.set_title("Edge Distribution Box Plots")
    ax2.grid(True, alpha=0.3, axis='y')
    
    stats_text = "Summary Statistics:\n"
    for col in edges.columns:
        data = edges[col].dropna()
        if len(data) > 0:
            stats_text += f"\n{col}:\n"
            stats_text += f"  Mean: {data.mean():.4f}, Median: {data.median():.4f}\n"
            stats_text += f"  Std: {data.std():.4f}, IQR: {data.quantile(0.75) - data.quantile(0.25):.4f}\n"
            stats_text += f"  % Positive: {(data > 0).mean()*100:.1f}%\n"
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=8, family='monospace',
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {output_path}")
    plt.close()


def plot_edge_by_dte(df: pd.DataFrame, edges: pd.DataFrame, output_path: Path):
    """Create edge distribution by DTE bucket."""
    dte_col = None
    for col in ["T_days", "t_days", "dte_days", "days_to_expiry"]:
        if col in df.columns:
            dte_col = col
            break
    
    if dte_col is None:
        print("No DTE column found, skipping DTE plot")
        return
    
    df = df.copy()
    df["dte"] = pd.to_numeric(df[dte_col], errors="coerce")
    
    for col in edges.columns:
        df[f"edge_{col}"] = edges[col].values
    
    buckets = [(0, 2, "0-2d"), (2, 4, "2-4d"), (4, 6, "4-6d"), (6, 10, "6-10d")]
    
    fig, axes = plt.subplots(len(buckets), 1, figsize=(12, 3*len(buckets)))
    if len(buckets) == 1:
        axes = [axes]
    
    colors = {
        "Model Fit": "#2ecc71",
        "Model Calibrated": "#3498db",
        "Raw MC": "#e74c3c",
    }
    
    for ax, (min_dte, max_dte, label) in zip(axes, buckets):
        bucket_df = df[(df["dte"] >= min_dte) & (df["dte"] < max_dte)]
        
        for edge_col in edges.columns:
            data = bucket_df[f"edge_{edge_col}"].dropna()
            if len(data) > 0:
                ax.hist(data, bins=50, alpha=0.5, label=f"{edge_col} (n={len(data):,})",
                       color=colors.get(edge_col, "gray"), density=True)
        
        ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
        ax.set_xlim(-0.4, 0.4)
        ax.set_title(f"DTE: {label}")
        ax.set_xlabel("Edge")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved DTE plot to: {output_path}")
    plt.close()


def main():
    print("=" * 60)
    print("Edge Distribution Analysis")
    print("=" * 60)
    
    df = load_all_batches()
    if df.empty:
        return
    
    print(f"\nColumns available: {df.columns.tolist()}")
    
    print("\nComputing edges...")
    edges = compute_edges(df)
    
    if edges.empty:
        print("No edges computed!")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    main_plot_path = PLOTS_DIR / f"edge_distribution_{timestamp}.png"
    plot_edge_distributions(edges, main_plot_path)
    
    dte_plot_path = PLOTS_DIR / f"edge_by_dte_{timestamp}.png"
    plot_edge_by_dte(df, edges, dte_plot_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
