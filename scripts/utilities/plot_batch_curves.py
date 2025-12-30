#!/usr/bin/env python
"""
plot_batch_curves.py

Reads the augmented `batch_with_fits.csv` and emits a suite of diagnostics
per expiry/bucket:

1) Probabilities vs strike (scatter + fitted sigmoids).
2) Logistic residuals vs strike.
3) Edge vs strike (model vs market/RN).
4) Real-world vs RN overlay.
5) Implied-volatility scatter (per expiry) when IV columns are present.

Each plot type is stored in its own `plots/<subcategory>/` subfolder.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sanitize_label(label: str) -> str:
    return str(label).replace(" ", "_").replace(":", "-").replace("/", "-")


def main():
    input_csv = "batch_with_fits.csv"
    out_dir = "plots"

    subdirs = {
        "prob": os.path.join(out_dir, "prob_vs_strike"),
        "resid": os.path.join(out_dir, "residuals"),
        "edge": os.path.join(out_dir, "edges"),
        "overlay": os.path.join(out_dir, "rn_overlay"),
        "iv": os.path.join(out_dir, "iv_surface"),
    }
    for path in subdirs.values():
        ensure_dir(path)

    df = pd.read_csv(input_csv)

    # Determine grouping columns
    if "expiry_date" in df.columns:
        exp_col = "expiry_date"
    else:
        exp_col = None

    if "t_days" in df.columns:
        t_col = "t_days"
    elif "T_days" in df.columns:
        t_col = "T_days"
    else:
        t_col = None

    if "market_pr" in df.columns:
        mkt_col = "market_pr"
    elif "market_price" in df.columns:
        mkt_col = "market_price"
    else:
        raise ValueError("No market price column found (expected 'market_pr' or 'market_price').")

    for col in ["strike", "p_real_mc", "p_model_fit", "p_rn_fit"]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {input_csv}.")

    if exp_col is not None:
        groups = df.groupby(exp_col, observed=True)
    else:
        if t_col is None:
            raise ValueError("No expiry_date or t_days / T_days column to group on.")
        df["t_bucket"] = df[t_col].round(3)
        groups = df.groupby("t_bucket", observed=True)

    for key, g in groups:
        expiry_label = str(key) if exp_col is not None else f"T={key}d"
        safe_label = sanitize_label(expiry_label)
        g = g.sort_values("strike")
        K = g["strike"].values
        p_mc = g["p_real_mc"].values
        p_model_fit = g["p_model_fit"].values
        p_mkt = g[mkt_col].values
        p_rn_fit = g["p_rn_fit"].values

        # 1) Probabilities vs strike (scatter + fitted curves)
        plt.figure(figsize=(8, 5))
        plt.scatter(K, p_mc, label="MC p_real_mc", alpha=0.5)
        plt.plot(K, p_model_fit, label="Model fit", linewidth=2)
        plt.scatter(K, p_mkt, label="Market price", alpha=0.5)
        plt.plot(K, p_rn_fit, label="RN/market fit", linewidth=2)
        plt.xlabel("Strike K")
        plt.ylabel("Probability / Price")
        plt.title(f"Probabilities vs Strike | {expiry_label}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(subdirs["prob"], f"prob_vs_strike_{safe_label}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved {out_path}")

        # 2) Residuals: MC - logistic fit
        if np.isfinite(p_model_fit).any():
            resid = p_mc - p_model_fit
            plt.figure(figsize=(8, 4))
            plt.axhline(0.0, color="black", linewidth=1, linestyle="--")
            plt.scatter(K, resid, alpha=0.6)
            plt.xlabel("Strike K")
            plt.ylabel("MC - model_fit")
            plt.title(f"Residuals vs Strike | {expiry_label}")
            plt.grid(True)
            plt.tight_layout()
            out_path = os.path.join(subdirs["resid"], f"residuals_{safe_label}.png")
            plt.savefig(out_path)
            plt.close()
            print(f"Saved {out_path}")

        # 3) Edge plot (model - market / RN)
        edge_market = p_mc - p_mkt if "edge_vs_market" not in g.columns else g["edge_vs_market"].values
        edge_market_fit = (
            p_model_fit - p_mkt if "edge_vs_market_fit" not in g.columns else g["edge_vs_market_fit"].values
        )
        edge_rn = (
            p_mc - p_rn_fit if "edge_vs_rn_fit" not in g.columns else g["edge_vs_rn_fit"].values
        )
        plt.figure(figsize=(8, 4))
        plt.axhline(0.0, color="black", linewidth=1, linestyle="--")
        plt.scatter(K, edge_market, label="Edge vs market (MC)", alpha=0.5)
        plt.plot(K, edge_market_fit, label="Edge vs market (fit)", linewidth=2)
        plt.scatter(K, edge_rn, label="Edge vs RN", alpha=0.5)
        plt.xlabel("Strike K")
        plt.ylabel("Edge (model - price)")
        plt.title(f"Edge vs Strike | {expiry_label}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        out_path = os.path.join(subdirs["edge"], f"edges_{safe_label}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved {out_path}")

        # 4) Real-world vs RN overlay
        plt.figure(figsize=(8, 5))
        plt.plot(K, p_model_fit, label="Model fit", linewidth=2)
        plt.plot(K, p_rn_fit, label="RN fit", linewidth=2)
        plt.scatter(K, p_mc, label="MC p_real_mc", alpha=0.4)
        plt.scatter(K, p_mkt, label="Market price", alpha=0.4)
        plt.xlabel("Strike K")
        plt.ylabel("Probability / Price")
        plt.title(f"Model vs RN Overlay | {expiry_label}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(subdirs["overlay"], f"overlay_{safe_label}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved {out_path}")

    # 5) IV surface (if IV columns available)
    iv_cols = [c for c in df.columns if c.lower() in {"iv", "iv_percent", "mark_iv"}]
    if iv_cols:
        iv_col = iv_cols[0]
        plt.figure(figsize=(8, 5))
        if exp_col and exp_col in df.columns:
            for key, g in df.groupby(exp_col, observed=True):
                plt.scatter(g["strike"], g[iv_col], label=str(key), alpha=0.6)
        else:
            plt.scatter(df["strike"], df[iv_col], alpha=0.6, label="IV")
        plt.xlabel("Strike K")
        plt.ylabel("IV (%)" if "percent" in iv_col.lower() else "IV")
        plt.title("Implied Volatility Surface (Deribit)")
        plt.grid(True)
        if exp_col and exp_col in df.columns:
            plt.legend()
        plt.tight_layout()
        out_path = os.path.join(subdirs["iv"], "iv_surface.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved {out_path}")
    else:
        print("Warning: No IV column found (iv/iv_percent/mark_iv); skipping IV surface plot.")


if __name__ == "__main__":
    main()
