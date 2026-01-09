import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
FILE_PATH = "experiment_results_final.csv"
sns.set_theme(style="whitegrid")

# Branding Colors
TECH_COLOR = "#1f77b4" # Blue
EMP_COLOR = "#2ca02c"  # Green
TIE_COLOR = "#ffc107"  # Yellow

def calculate_hierarchical_winner(row):
    """Priority: Safety > Empathy. Independent method."""
    if row['Tech_Safety'] > row['Emp_Safety']: return 'Tech'
    if row['Emp_Safety'] > row['Tech_Safety']: return 'Emp'
    if row['Tech_Empathy'] > row['Emp_Empathy']: return 'Tech'
    if row['Emp_Empathy'] > row['Tech_Empathy']: return 'Emp'
    return 'Tie'

def load_data():
    try:
        df = pd.read_csv(FILE_PATH)
        df['Winner'] = df['Winner'].replace({'A': 'Tech', 'B': 'Emp'})
        if 'Risk' not in df.columns:
            half = len(df) // 2
            df['Risk'] = np.where(df['ID'] <= half, "Low Risk", "High Risk")
        # Ensure the hierarchical method result is present
        df['Score_Winner'] = df.apply(calculate_hierarchical_winner, axis=1)
        return df
    except Exception as e:
        print(f"❌ Error: {e}"); exit()

def save_count_plot(df, col, title, filename):
    plt.figure(figsize=(10, 6))
    # 1. Keep 'col' here so it finds 'Score_Winner' in your data
    sns.countplot(data=df, x=col, palette={"Tech": TECH_COLOR, "Emp": EMP_COLOR, "Tie": TIE_COLOR}, order=['Tech', 'Emp', 'Tie'])
    # 2. ADD THIS LINE to remove the "Score_Winner" text from the center
    plt.xlabel('') 
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def run_all_plots():
    df = load_data()
    
    # --- CHART 1: COGNITIVE TRADEOFF (JITTERED DOTS) ---
    # High jitter (0.18) to show the volume of dots at each score point
    plt.figure(figsize=(10, 8))
    j = 0.18 
    plt.scatter(df['Tech_Safety'] + np.random.uniform(-j, j, len(df)), 
                df['Tech_Empathy'] + np.random.uniform(-j, j, len(df)), 
                color=TECH_COLOR, label='Tech Agent', alpha=0.6, s=100, edgecolors='w')
    plt.scatter(df['Emp_Safety'] + np.random.uniform(-j, j, len(df)), 
                df['Emp_Empathy'] + np.random.uniform(-j, j, len(df)), 
                color=EMP_COLOR, label='Emp Agent', alpha=0.6, s=100, edgecolors='w')
    plt.axvline(3, ls='--', color='gray', alpha=0.4); plt.axhline(3, ls='--', color='gray', alpha=0.4)
    plt.title("Chart 1: Tech vs Emp - Safety & Empathy Distribution", fontsize=14)
    plt.xlabel("Safety Score"); plt.ylabel("Empathy Score"); plt.legend()
    plt.savefig('chart_1_cognitive_tradeoff.png', dpi=300); plt.close()
    print("📸 Saved chart_1_cognitive_tradeoff.png")

    # --- CHART 2: PERFORMANCE VARIANCE (BOXPLOTS) ---
    plt.figure(figsize=(10, 6))
    melted = df[['Tech_Safety', 'Emp_Safety', 'Tech_Empathy', 'Emp_Empathy']].melt(var_name='Var', value_name='Score')
    melted['Agent'] = np.where(melted['Var'].str.contains('Tech'), 'Tech', 'Emp')
    melted['Metric'] = np.where(melted['Var'].str.contains('Safety'), 'Safety', 'Empathy')
    sns.boxplot(data=melted, x='Metric', y='Score', hue='Agent', palette={"Tech": TECH_COLOR, "Emp": EMP_COLOR})
    plt.title("Chart 2: Score Variance (Tech vs Emp)", fontsize=14)
    plt.savefig('chart_2_variance_boxplot.png', dpi=300); plt.close()
    print("📸 Saved chart_2_variance_boxplot.png")

    # --- CHART 3 & 4: OVERALL SUMMARIES BY METHOD ---
    # Results based on the Pairwise method
    save_count_plot(df, 'Winner', "Chart 3: Overall Results (Pairwise Method)", 'chart_3_pairwise_overall.png')
    # Results based on the Hierarchical method
    save_count_plot(df, 'Score_Winner', "Chart 4: Overall Results (Hierarchical Method)", 'chart_4_hierarchical_overall.png')

   # --- CHART 5-8: RISK-BASED SPLITS BY METHOD ---
    for risk in ["Low Risk", "High Risk"]:
        sub = df[df['Risk'] == risk]
        sfx = risk.lower().replace(" ", "_")
        
        # Determine chart numbers based on risk
        p_num = "5" if risk == "Low Risk" else "7"
        h_num = "6" if risk == "Low Risk" else "8"
        
        # Pairwise method per risk (Uses 'Winner' column)
        save_count_plot(sub, 'Winner', f"Chart {p_num}: {risk} Results (Pairwise Method)", f'chart_{p_num}_pairwise_{sfx}.png')
        save_count_plot(sub, 'Score_Winner', f"Chart {h_num}: {risk} Results (Hierarchical Method)", f'chart_{h_num}_hierarchical_{sfx}.png')

if __name__ == "__main__":
    run_all_plots()
    print("\n✅ All 8 charts generated with correct naming and density.")