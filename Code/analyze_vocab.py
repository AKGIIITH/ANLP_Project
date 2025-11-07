#!/usr/bin/env python3
"""
Vocabulary Probability Analysis Script
Analyzes probability changes for ALL tokens between two models
Requires vocabulary probability files generated from models
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    "output_dir": "./experiment_outputs"
}

print("="*80)
print("VOCABULARY PROBABILITY JUMP ANALYSIS")
print("="*80)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def load_vocab_probs(file_path):
    """Load vocabulary probabilities from JSON file"""
    print(f"Loading {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert to dictionary for easy lookup
    prob_dict = {}
    for item in data['probabilities']:
        prob_dict[item['token_id']] = {
            'token': item['token'],
            'probability': item['probability'],
            'log_probability': item['log_probability']
        }
    
    print(f"✓ Loaded {len(prob_dict)} tokens")
    return prob_dict


def calculate_all_probability_jumps(file1, file2, model1_name, model2_name, 
                                   output_dir="./experiment_outputs"):
    """
    Calculate absolute probability changes for ENTIRE vocabulary
    Sort by absolute magnitude and save comprehensive results
    
    Args:
        file1: Path to baseline model's vocab probabilities
        file2: Path to comparison model's vocab probabilities
        model1_name: Name of baseline model
        model2_name: Name of comparison model
        output_dir: Where to save results
    """
    print("="*80)
    print(f"ANALYZING PROBABILITY JUMPS: {model1_name} → {model2_name}")
    print("="*80)
    
    # Load both probability distributions
    probs1 = load_vocab_probs(file1)
    probs2 = load_vocab_probs(file2)
    
    print(f"\nCalculating probability changes for {len(probs1)} tokens...")
    
    # Calculate all differences
    all_changes = []
    
    for token_id in probs1.keys():
        if token_id in probs2:
            token = probs1[token_id]['token']
            prob_before = probs1[token_id]['probability']
            prob_after = probs2[token_id]['probability']
            
            # Calculate change metrics
            absolute_change = prob_after - prob_before
            absolute_magnitude = abs(absolute_change)
            
            # Avoid division by zero for relative change
            if prob_before > 0:
                relative_change = (prob_after - prob_before) / prob_before
                fold_change = prob_after / prob_before
            else:
                relative_change = float('inf') if prob_after > 0 else 0
                fold_change = float('inf') if prob_after > 0 else 0
            
            all_changes.append({
                'token_id': token_id,
                'token': token,
                'token_repr': repr(token),  # Python repr for clarity
                'prob_before': prob_before,
                'prob_after': prob_after,
                'absolute_change': absolute_change,
                'absolute_magnitude': absolute_magnitude,
                'percent_change': absolute_change * 100,
                'relative_change': relative_change,
                'fold_change': fold_change,
                'direction': 'increase' if absolute_change > 0 else ('decrease' if absolute_change < 0 else 'unchanged')
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(all_changes)
    
    print(f"✓ Calculated changes for {len(df)} tokens")
    
    # ============================================================
    # SORT BY ABSOLUTE MAGNITUDE (BIGGEST JUMPS REGARDLESS OF DIRECTION)
    # ============================================================
    print("\n" + "="*80)
    print("SORTING BY ABSOLUTE MAGNITUDE OF CHANGE")
    print("="*80)
    
    df_sorted = df.sort_values('absolute_magnitude', ascending=False)
    
    # Save FULL sorted results
    full_output_file = f"{output_dir}/full_vocab_jumps_{model1_name}_to_{model2_name}.csv"
    df_sorted.to_csv(full_output_file, index=False)
    print(f"\n✓ FULL vocabulary saved to: {full_output_file}")
    print(f"  (All {len(df_sorted)} tokens sorted by absolute change magnitude)")
    
    # ============================================================
    # DISPLAY TOP 100 BIGGEST JUMPS
    # ============================================================
    print("\n" + "="*80)
    print("TOP 100 BIGGEST PROBABILITY JUMPS (BY ABSOLUTE MAGNITUDE)")
    print("="*80)
    
    top_100 = df_sorted.head(100)
    
    print(f"\n{'Rank':<6}{'Token':<35}{'Before %':<12}{'After %':<12}{'Change %':<12}{'Direction':<10}")
    print("-" * 95)
    
    for i, (_, row) in enumerate(top_100.iterrows(), 1):
        token_display = row['token_repr'][:33]  # Truncate if too long
        direction_symbol = '↑' if row['direction'] == 'increase' else ('↓' if row['direction'] == 'decrease' else '→')
        
        print(f"{i:<6}{token_display:<35}{row['prob_before']*100:>10.6f}  "
              f"{row['prob_after']*100:>10.6f}  "
              f"{row['percent_change']:>+10.6f}  "
              f"{direction_symbol} {row['direction']:<9}")
    
    # Save top 100 to separate file for easy viewing
    top_100_file = f"{output_dir}/top_100_jumps_{model1_name}_to_{model2_name}.csv"
    top_100.to_csv(top_100_file, index=False)
    print(f"\n✓ Top 100 saved to: {top_100_file}")
    
    # ============================================================
    # SAVE TOP 500 FOR DETAILED ANALYSIS
    # ============================================================
    top_500 = df_sorted.head(500)
    top_500_file = f"{output_dir}/top_500_jumps_{model1_name}_to_{model2_name}.csv"
    top_500.to_csv(top_500_file, index=False)
    print(f"✓ Top 500 saved to: {top_500_file}")
    
    # ============================================================
    # SEPARATE INCREASES AND DECREASES
    # ============================================================
    print("\n" + "="*80)
    print("TOP 50 INCREASES AND DECREASES (SEPARATED)")
    print("="*80)
    
    # Top increases
    df_increases = df[df['absolute_change'] > 0].sort_values('absolute_change', ascending=False)
    top_50_increases = df_increases.head(50)
    
    print("\nTOP 50 INCREASES:")
    print(f"{'Rank':<6}{'Token':<35}{'Before %':<12}{'After %':<12}{'Increase %':<12}")
    print("-" * 80)
    for i, (_, row) in enumerate(top_50_increases.iterrows(), 1):
        token_display = row['token_repr'][:33]
        print(f"{i:<6}{token_display:<35}{row['prob_before']*100:>10.6f}  "
              f"{row['prob_after']*100:>10.6f}  {row['percent_change']:>+10.6f}")
    
    increases_file = f"{output_dir}/top_increases_{model1_name}_to_{model2_name}.csv"
    df_increases.to_csv(increases_file, index=False)
    print(f"\n✓ All increases saved to: {increases_file}")
    
    # Top decreases
    df_decreases = df[df['absolute_change'] < 0].sort_values('absolute_change', ascending=True)
    top_50_decreases = df_decreases.head(50)
    
    print("\n\nTOP 50 DECREASES:")
    print(f"{'Rank':<6}{'Token':<35}{'Before %':<12}{'After %':<12}{'Decrease %':<12}")
    print("-" * 80)
    for i, (_, row) in enumerate(top_50_decreases.iterrows(), 1):
        token_display = row['token_repr'][:33]
        print(f"{i:<6}{token_display:<35}{row['prob_before']*100:>10.6f}  "
              f"{row['prob_after']*100:>10.6f}  {row['percent_change']:>+10.6f}")
    
    decreases_file = f"{output_dir}/top_decreases_{model1_name}_to_{model2_name}.csv"
    df_decreases.to_csv(decreases_file, index=False)
    print(f"\n✓ All decreases saved to: {decreases_file}")
    
    # ============================================================
    # SUMMARY STATISTICS
    # ============================================================
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nTotal tokens analyzed: {len(df):,}")
    print(f"Tokens that increased: {len(df_increases):,} ({len(df_increases)/len(df)*100:.2f}%)")
    print(f"Tokens that decreased: {len(df_decreases):,} ({len(df_decreases)/len(df)*100:.2f}%)")
    print(f"Tokens unchanged: {len(df[df['absolute_change'] == 0]):,}")
    
    print(f"\nLargest increase: {df['absolute_change'].max()*100:.6f}%")
    print(f"  Token: {repr(df.loc[df['absolute_change'].idxmax(), 'token'])}")
    
    print(f"\nLargest decrease: {df['absolute_change'].min()*100:.6f}%")
    print(f"  Token: {repr(df.loc[df['absolute_change'].idxmin(), 'token'])}")
    
    print(f"\nMean absolute change: {df['absolute_magnitude'].mean()*100:.8f}%")
    print(f"Median absolute change: {df['absolute_magnitude'].median()*100:.8f}%")
    print(f"Std dev of changes: {df['absolute_change'].std()*100:.8f}%")
    
    # Percentiles
    percentiles = [90, 95, 99, 99.9]
    print("\nPercentiles of absolute change magnitude:")
    for p in percentiles:
        val = df['absolute_magnitude'].quantile(p/100)
        print(f"  {p}th percentile: {val*100:.8f}%")
    
    # ============================================================
    # VISUALIZATION
    # ============================================================
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Top 30 increases
    ax1 = axes[0, 0]
    top_30_inc = top_50_increases.head(30)
    ax1.barh(range(len(top_30_inc)), top_30_inc['percent_change'], color='green', alpha=0.7)
    ax1.set_yticks(range(len(top_30_inc)))
    ax1.set_yticklabels([repr(t)[:25] for t in top_30_inc['token']], fontsize=7)
    ax1.set_xlabel('Probability Increase (%)', fontsize=10)
    ax1.set_title(f'Top 30 Increases\n{model1_name} → {model2_name}', 
                  fontsize=11, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Top 30 decreases
    ax2 = axes[0, 1]
    top_30_dec = top_50_decreases.head(30)
    ax2.barh(range(len(top_30_dec)), top_30_dec['percent_change'], color='red', alpha=0.7)
    ax2.set_yticks(range(len(top_30_dec)))
    ax2.set_yticklabels([repr(t)[:25] for t in top_30_dec['token']], fontsize=7)
    ax2.set_xlabel('Probability Decrease (%)', fontsize=10)
    ax2.set_title(f'Top 30 Decreases\n{model1_name} → {model2_name}', 
                  fontsize=11, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    # Plot 3: Distribution of changes (histogram)
    ax3 = axes[1, 0]
    # Filter to reasonable range for visualization
    changes_filtered = df['percent_change'][
        (df['percent_change'] > -1) & (df['percent_change'] < 1)
    ]
    ax3.hist(changes_filtered, bins=100, color='blue', alpha=0.6, edgecolor='black')
    ax3.set_xlabel('Probability Change (%)', fontsize=10)
    ax3.set_ylabel('Number of Tokens', fontsize=10)
    ax3.set_title('Distribution of Probability Changes\n(filtered to ±1%)', 
                  fontsize=11, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No change')
    ax3.legend()
    
    # Plot 4: Log-scale scatter
    ax4 = axes[1, 1]
    # Add small epsilon to avoid log(0)
    prob_before_plot = df['prob_before'] + 1e-10
    prob_after_plot = df['prob_after'] + 1e-10
    scatter = ax4.scatter(prob_before_plot, prob_after_plot, 
                         c=df['absolute_change'], cmap='RdYlGn', 
                         alpha=0.3, s=1)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.plot([1e-10, 1], [1e-10, 1], 'k--', linewidth=1, label='No change')
    ax4.set_xlabel(f'Probability Before ({model1_name})', fontsize=10)
    ax4.set_ylabel(f'Probability After ({model2_name})', fontsize=10)
    ax4.set_title('Before vs After Probabilities (log scale)', 
                  fontsize=11, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Absolute Change')
    
    plt.tight_layout()
    plot_file = f"{output_dir}/full_vocab_jumps_{model1_name}_to_{model2_name}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {plot_file}")
    plt.close()
    
    # ============================================================
    # SAVE SUMMARY REPORT
    # ============================================================
    summary_report = {
        'comparison': f"{model1_name} → {model2_name}",
        'total_tokens': len(df),
        'tokens_increased': len(df_increases),
        'tokens_decreased': len(df_decreases),
        'tokens_unchanged': len(df[df['absolute_change'] == 0]),
        'largest_increase_percent': float(df['absolute_change'].max() * 100),
        'largest_increase_token': df.loc[df['absolute_change'].idxmax(), 'token'],
        'largest_decrease_percent': float(df['absolute_change'].min() * 100),
        'largest_decrease_token': df.loc[df['absolute_change'].idxmin(), 'token'],
        'mean_absolute_change': float(df['absolute_magnitude'].mean() * 100),
        'median_absolute_change': float(df['absolute_magnitude'].median() * 100),
        'std_change': float(df['absolute_change'].std() * 100),
        'percentiles': {
            f'{p}th': float(df['absolute_magnitude'].quantile(p/100) * 100)
            for p in [90, 95, 99, 99.9]
        }
    }
    
    summary_file = f"{output_dir}/summary_{model1_name}_to_{model2_name}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_report, f, indent=2)
    print(f"✓ Summary report saved to: {summary_file}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print(f"  1. {full_output_file}")
    print(f"     → COMPLETE vocabulary ({len(df):,} tokens) sorted by absolute magnitude")
    print(f"  2. {top_100_file}")
    print(f"     → Top 100 biggest jumps (quick reference)")
    print(f"  3. {top_500_file}")
    print(f"     → Top 500 biggest jumps (detailed analysis)")
    print(f"  4. {increases_file}")
    print(f"     → All {len(df_increases):,} tokens that increased")
    print(f"  5. {decreases_file}")
    print(f"     → All {len(df_decreases):,} tokens that decreased")
    print(f"  6. {plot_file}")
    print(f"     → Visualization of changes")
    print(f"  7. {summary_file}")
    print(f"     → Summary statistics (JSON)")
    
    return df_sorted


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("COMPREHENSIVE PROBABILITY JUMP ANALYSIS")
    print("="*80)
    
    # Analysis 1: Baseline → Full Finetune
    print("\n\n" + "="*80)
    print("ANALYSIS 1: Baseline → Full Finetune")
    print("="*80)
    df1 = calculate_all_probability_jumps(
        file1=f"{CONFIG['output_dir']}/Baseline_full_vocab_probs.json",
        file2=f"{CONFIG['output_dir']}/Full_Finetune_full_vocab_probs.json",
        model1_name="Baseline",
        model2_name="Full_Finetune",
        output_dir=CONFIG['output_dir']
    )
    
    # Analysis 2: Baseline → MLP-Only
    print("\n\n" + "="*80)
    print("ANALYSIS 2: Baseline → MLP-Only")
    print("="*80)
    df2 = calculate_all_probability_jumps(
        file1=f"{CONFIG['output_dir']}/Baseline_full_vocab_probs.json",
        file2=f"{CONFIG['output_dir']}/MLP_Only_full_vocab_probs.json",
        model1_name="Baseline",
        model2_name="MLP_Only",
        output_dir=CONFIG['output_dir']
    )
    
    # Analysis 3: Baseline → Attention-Only
    print("\n\n" + "="*80)
    print("ANALYSIS 3: Baseline → Attention-Only")
    print("="*80)
    df3 = calculate_all_probability_jumps(
        file1=f"{CONFIG['output_dir']}/Baseline_full_vocab_probs.json",
        file2=f"{CONFIG['output_dir']}/Attention_Only_full_vocab_probs.json",
        model1_name="Baseline",
        model2_name="Attention_Only",
        output_dir=CONFIG['output_dir']
    )
    
    # Analysis 4: MLP-Only → Attention-Only (Direct comparison)
    print("\n\n" + "="*80)
    print("ANALYSIS 4: MLP-Only → Attention-Only")
    print("="*80)
    df4 = calculate_all_probability_jumps(
        file1=f"{CONFIG['output_dir']}/MLP_Only_full_vocab_probs.json",
        file2=f"{CONFIG['output_dir']}/Attention_Only_full_vocab_probs.json",
        model1_name="MLP_Only",
        model2_name="Attention_Only",
        output_dir=CONFIG['output_dir']
    )
    
    print("\n\n" + "="*80)
    print("ALL ANALYSES COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {CONFIG['output_dir']}/")
    print("\nYou can now open the CSV files to see:")
    print("  - Which tokens changed the most")
    print("  - Complete sorted vocabulary by absolute change magnitude")
    print("  - Separate lists of increases and decreases")
    print("="*80)