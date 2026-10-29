"""
visualisation.py
Plotting utilities for model results and feature importance.
Paper: Ben-Koko & Warribo (2026), WorldS4 2026, Springer LNNS, London
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_model_comparison(results_df, save_path=None):
    """Bar chart comparing AUC-ROC across all models."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colours = ['#2E75B6' if m == 'Random Forest' else '#BDBDBD' for m in results_df['Model']]
    bars = ax.barh(results_df['Model'], results_df['AUC-ROC'], color=colours)
    ax.set_xlabel('AUC-ROC', fontsize=12)
    ax.set_title('Model Performance — Neonatal Mortality Prediction (Nigeria)', fontsize=13)
    ax.set_xlim(0.5, 1.0)
    ax.axvline(x=0.7, color='red', linestyle='--', alpha=0.5, label='AUC = 0.7 threshold')
    for bar, val in zip(bars, results_df['AUC-ROC']):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=10)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_feature_importance(importance_df, top_n=9, save_path=None):
    """Horizontal bar chart of feature importance scores."""
    top = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    colours = ['#1F3864' if i == 0 else '#2E75B6' if i < 3 else '#BDBDBD' for i in range(len(top))]
    bars = ax.barh(top['Feature'][::-1], top['Importance'][::-1], color=colours[::-1])
    ax.set_xlabel('Feature Importance (Mean Decrease in Impurity)', fontsize=11)
    ax.set_title('Random Forest Feature Importance — Neonatal Mortality Predictors', fontsize=13)
    for bar, val in zip(bars, top['Importance'][::-1]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
