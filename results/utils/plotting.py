import pandas as pd
import matplotlib.pyplot as plt
import os


# Define base colors for each main variant
base_colors = {
    'baseline': 'black',
    'e-chat-history': 'tab:blue',
    'e-persona-you': 'tab:orange',
    'e-persona-human': 'tab:green',
    'e-chain-of-thought': 'tab:red',
    'e-implcit-target': 'tab:gray',
}

def plot_attack_comparison(models, attacks, plot_dir):

    records = []
    for model in models:
        # Extract model family ("Llama" or "Qwen") and size ("1B", "3B", etc.)
        family = model.split('-')[0].capitalize()
        size = float(model.split('-')[1].replace('b', '')) if len(model.split('-')) > 1 else None
        if family == 'Qwen' and size == 1.0:
            size = 1.5  # Adjusting size for Qwen-1B to 1.5B for consistency

        for variant, vals in attacks[model].items():
            records.append({
                "model": model,
                "family": family,
                "size": size,
                "variant": variant,
                "success_rate": vals['successful'] / vals['total']
            })

    df = pd.DataFrame(records)
    prompt_variants = df[~df['variant'].str.contains("TextFooler", case=False, na=False)]

    fig, axes = plt.subplots(1, 2, figsize=(15,7), sharey=True)

    # Get unique sizes from the data for x-axis ticks
    unique_sizes = sorted(df['size'].unique())
    # Filter to show fewer ticks to prevent overlap (keep major points)
    major_sizes = [s for s in unique_sizes if s in [1, 1.5, 3, 8, 14, 32, 70]]
    size_labels = [f'{int(s) if s == int(s) else s}B' for s in major_sizes]
    
    for i, family in enumerate(['Llama3', 'Qwen']):
        subdf = prompt_variants[prompt_variants['family']==family]
        for variant, vdf in subdf.groupby('variant'):
            axes[i].plot(vdf['size'], vdf['success_rate'], marker="o", label=variant, color=base_colors.get(variant, 'gray'))
        if family == 'Qwen':
            family = 'Qwen2.5'  
        axes[i].set_title(f"{family} Family", fontsize=14)
        axes[i].set_xlabel("Model size (Billions of parameters)")
        axes[i].set_ylabel("Faithfulness")
        axes[i].set_xscale('log')
        axes[i].set_xticks(major_sizes)
        axes[i].set_xticklabels(size_labels, rotation=45, ha='right')
        axes[i].grid(True)

    plt.suptitle("Faithfulness vs. Model Size Across Prompt Variants", fontsize=16, fontweight='bold')
    plt.ylim(0, 1)  # Set y-axis limits from 0 to 1i
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'faithfulness_vs_model_size'), dpi=300, bbox_inches="tight")
    
    # Create and save separate legend
    legend_fig = plt.figure(figsize=(8, 2))
    legend_elements = []
    variant_labels = []
    
    # Get unique variants and their colors
    unique_variants = prompt_variants['variant'].unique()
    for variant in unique_variants:
        color = base_colors.get(variant, 'gray')
        legend_elements.append(plt.Line2D([0], [0], color=color, marker='o', linestyle='-', linewidth=2, markersize=8))
        variant_labels.append(variant)
    
    # Create legend on the figure
    legend_fig.legend(legend_elements, variant_labels, loc='center', ncol=len(unique_variants), 
                     fontsize=12, frameon=False)
    # Remove axes to show only legend
    legend_fig.gca().set_axis_off()
    legend_fig.savefig(os.path.join(plot_dir, 'legend'), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(legend_fig)
    
    plt.show()


    """
    attack_variants = df[df['variant'].str.contains("TextFooler", case=False, na=False)]
    fig2, ax = plt.subplots(figsize=(12,5))  # ✅ Create new figure and axis properly    

    for family, subdf in attack_variants.groupby('family'):
        for variant, vdf in subdf.groupby('variant'):
            plt.plot(
                vdf['size'], vdf['success_rate'],
                marker='s', linestyle='--', label=f"{family} - {variant}"
            )

    plt.title("Attack Success Rate by Model Size")
    plt.xlabel("Model Size (B parameters)")
    plt.ylabel("Attack Success Rate")
    plt.xscale('log')
    plt.xticks(major_sizes, size_labels, rotation=45, ha='right')
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax * 1.15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    fig2.savefig(os.path.join(plot_dir, 'Attack_Success'), dpi=300, bbox_inches="tight")
    plt.show()
    """
    

def extract_means(df):
    flattened = pd.json_normalize(df.apply(lambda x: x.to_dict(), axis=1))
    return {
        'closeness': {col: flattened[col].mean() for col in flattened if 'closeness' in col},
        'semantic_similarity': {col: flattened[col].mean() for col in flattened if 'similarity' in col},
        'contradiction': {col: flattened[col].mean() for col in flattened if 'contradiction' in col},
        #'fluency': {col: flattened[col].mean() for col in flattened if 'fluency' in col},
        #'diversity': {col: flattened[col].mean() for col in flattened if 'diversity' in col},
        #'evidence_accuracy': {col: flattened[col].mean() for col in flattened if 'evidence_accuracy' in col},
        'evidence_precision': {col: flattened[col].mean() for col in flattened if 'evidence_precision' in col},
        #'evidence_recall': {col: flattened[col].mean() for col in flattened if 'evidence_recall' in col},
        #'evidence_f1': {col: flattened[col].mean() for col in flattened if 'evidence_f1' in col},
        #'crest_accuracy': {col: flattened[col].mean() for col in flattened if 'crest_accuracy' in col},
        'crest_precision': {col: flattened[col].mean() for col in flattened if 'crest_precision' in col},
        #'crest_recall': {col: flattened[col].mean() for col in flattened if 'crest_recall' in col},
        #'crest_f1': {col: flattened[col].mean() for col in flattened if 'crest_f1' in col}, 
    }


def plot_size_comparison(models, results, directory, show_plots=True):
    
    
    os.makedirs(directory, exist_ok=True)

    # --- Build model data ---
    model_data = {}
    for model in models:
        df = results[model]


        family = model.split('-')[0].capitalize()
        size = float(model.split('-')[1].replace('b', '')) if len(model.split('-')) > 1 else None
        if family == 'Qwen' and size == 1.0:
            size = 1.5  # Adjusting size for Qwen-1B to 1.5B for consistency


        model_data[f"{family}_{size}B"] = {
            'family': family,
            'size':  f'{size}B',
            'size_num':size,
            'metrics': extract_means(df)
        }

    # --- Metric groups (from first model) ---
    first_model = next(iter(model_data.values()))
    metric_groups = {k: v for k, v in first_model['metrics'].items() if v}

    print("Metric groups detected:")
    for k in metric_groups.keys():
        print("  •", k)


    prompt_variations = [
        "baseline",
        "e-chat-history",
        "e-persona-you",
        "e-persona-human",
        "e-implcit-target",
        "e-chain-of-thought"
    ]

    # --- Families and model sizes ---
    families = sorted(set(data['family'] for data in model_data.values()))
    all_sizes = sorted(set((data['size'], data['size_num']) for data in model_data.values()),
                       key=lambda x: x[1])
    all_size_labels = [s[0] for s in all_sizes]
    all_size_nums = [s[1] for s in all_sizes]
    
    # Filter to show fewer ticks to prevent overlap (keep major points)
    major_size_nums = [s for s in all_size_nums if s in [1, 1.5, 3, 8, 14, 32, 70]]
    major_size_labels = [f'{int(s) if s == int(s) else s}B' for s in major_size_nums]

    # === MAIN LOOP ===
    for metric_group_name in metric_groups.keys():
        print(f"\n🔹 Processing metric group: {metric_group_name}")

        for family in families:

            fig, ax = plt.subplots(figsize=(6, 5))
            plotted = False

            for variant in prompt_variations:
                exp_values = []
                for size_label, size_num in zip(all_size_labels, all_size_nums):
                    key = f"{family}_{size_label}"
                    if key not in model_data:
                        continue

                    metrics_dict = model_data[key]['metrics']
                    if metric_group_name not in metrics_dict:
                        continue

                    # --- match experiment name by prefix but exclude combined variants ---
                    found_key = None
                    for k in metrics_dict[metric_group_name].keys():
                        if k.lower().startswith(variant.lower()):
                            # Exclude combined variants by checking they don't have extra components
                            k_parts = k.lower().split('-')
                            variant_parts = variant.lower().split('-')
                            if len(k_parts) == len(variant_parts) or (len(k_parts) == len(variant_parts) + 1 and k_parts[-1].isdigit()):
                                found_key = k
                                break
                    if not found_key:
                        continue

                    val = metrics_dict[metric_group_name][found_key]
                    exp_values.append((size_num, float(val)))

                if not exp_values:
                    continue

                exp_values.sort()
                x_vals, y_vals = zip(*exp_values)
                ax.plot(x_vals, y_vals, marker='o', label=variant, color=base_colors.get(variant, 'gray'))
                plotted = True

            if family.title() == 'Qwen':    
                f = 'Qwen2.5'
            else:
                f = family.title()
            ax.set_title(f,fontsize=12)
            ax.set_xlabel('Model Size')
            ax.set_ylabel(metric_group_name.replace('_', ' ').title())
            ax.set_xscale('log')
            ax.set_xticks(major_size_nums)
            ax.set_xticklabels(major_size_labels, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, linestyle='--')

            if plotted:
                #ax.legend(loc='best', title='Promp variation',fontsize=8)
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(0, ymax * 1.15)

            plt.tight_layout()
            filename = f"{metric_group_name}_{family}.png"
            path = os.path.join(directory, filename)
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"   ✅ Saved {path}")
            if show_plots:
                plt.show()
            plt.close(fig)

