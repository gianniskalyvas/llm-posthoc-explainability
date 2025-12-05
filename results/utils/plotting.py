import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re
import os
import seaborn as sns


# Define base colors for each main variant
base_colors = {
    'baseline': 'tab:gray',
    'e-persona-you': 'tab:orange',
    'e-persona-human': 'tab:green'
}

# Function to make color darker
def darken_color(color, amount=0.6):
    c = mcolors.to_rgb(color)
    return tuple([amount*x for x in c])

def get_variant_color(variant):
    if 'e-implcit-target' in variant:
        # Map the corresponding base variant and darken it
        if 'e-persona-you' in variant:
            return darken_color(base_colors['e-persona-you'])
        elif 'e-persona-human' in variant:
            return darken_color(base_colors['e-persona-human'])
        else:  # default for baseline-related e-implcit-target
            return darken_color(base_colors['baseline'])
    else:
        return base_colors.get(variant, 'gray')  # fallback color

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

    # Split prompt variants from attacks
    prompt_variants = df[~df['variant'].str.contains("TextFooler", case=False, na=False)]
    attack_variants = df[df['variant'].str.contains("TextFooler", case=False, na=False)]

    # ------------------------------
    # 📈 1. Prompt variation success by model size
    # ------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(15,7), sharey=True)

    for i, family in enumerate(['Llama3', 'Qwen']):
        subdf = prompt_variants[prompt_variants['family']==family]
        for variant, vdf in subdf.groupby('variant'):
            axes[i].plot(vdf['size'], vdf['success_rate'], marker="o", label=variant, color=get_variant_color(variant))

        axes[i].set_title(f"{family} Models")
        axes[i].set_xlabel("Model Size (B parameters)")
        axes[i].set_ylabel("Success Rate")
        axes[i].grid(True)
        axes[i].legend(fontsize=8)

    plt.suptitle("Faithful Counterfactuals by Model Size & Prompt Variant")
    plt.ylim(0, 1)  # Set y-axis limits from 0 to 1
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'Introspection_Success'), dpi=300, bbox_inches="tight")
    plt.show()

    # ------------------------------
    # 💥 2. Attack success by model size
    # ------------------------------
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
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax * 1.15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    fig2.savefig(os.path.join(plot_dir, 'Attack_Success'), dpi=300, bbox_inches="tight")
    plt.show()
    
    

def extract_means(df):
    flattened = pd.json_normalize(df.apply(lambda x: x.to_dict(), axis=1))
    return {
        'closeness': {col: flattened[col].mean() for col in flattened if 'closeness' in col},
        'semantic_similarity': {col: flattened[col].mean() for col in flattened if 'similarity' in col},
        'contradiction': {col: flattened[col].mean() for col in flattened if 'contradiction' in col},
        'fluency': {col: flattened[col].mean() for col in flattened if 'fluency' in col},
        'diversity': {col: flattened[col].mean() for col in flattened if 'diversity' in col},
        'evidence_accuracy': {col: flattened[col].mean() for col in flattened if 'evidence_accuracy' in col},
        'evidence_precision': {col: flattened[col].mean() for col in flattened if 'evidence_precision' in col},
        'evidence_recall': {col: flattened[col].mean() for col in flattened if 'evidence_recall' in col},
        'evidence_f1': {col: flattened[col].mean() for col in flattened if 'evidence_f1' in col},
        'crest_accuracy': {col: flattened[col].mean() for col in flattened if 'crest_accuracy' in col},
        'crest_precision': {col: flattened[col].mean() for col in flattened if 'crest_precision' in col},
        'crest_recall': {col: flattened[col].mean() for col in flattened if 'crest_recall' in col},
        'crest_f1': {col: flattened[col].mean() for col in flattened if 'crest_f1' in col}, 
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

    # --- Experiment group sets ---
    experiment_sets = {
        "Introspection": [
            "baseline",
            "e-persona-you",
            "e-persona-human",
            "e-implcit-target",
            "e-implcit-target-e-persona-you",
            "e-implcit-target-e-persona-human"
        ],
        "TextFooler": [
            "TextFoolerJin2019"
        ]
    }

    # --- Families and model sizes ---
    families = sorted(set(data['family'] for data in model_data.values()))
    all_sizes = sorted(set((data['size'], data['size_num']) for data in model_data.values()),
                       key=lambda x: x[1])
    all_size_labels = [s[0] for s in all_sizes]
    all_size_nums = [s[1] for s in all_sizes]

    # === MAIN LOOP ===
    for metric_group_name in metric_groups.keys():
        print(f"\n🔹 Processing metric group: {metric_group_name}")

        for group_label, experiments in experiment_sets.items():

            # --- CASE 1: Baseline group → separate plots per family ---
            if group_label == "Introspection":
                for family in families:
                    print(f"   📊 Plotting {group_label} for {family}")
                    fig, ax = plt.subplots(figsize=(6, 5))
                    plotted = False

                    for variant in experiments:
                        exp_values = []
                        for size_label, size_num in zip(all_size_labels, all_size_nums):
                            key = f"{family}_{size_label}"
                            if key not in model_data:
                                continue

                            metrics_dict = model_data[key]['metrics']
                            if metric_group_name not in metrics_dict:
                                continue

                            # --- match experiment name by prefix ---
                            found_key = None
                            for k in metrics_dict[metric_group_name].keys():
                                if k.lower().startswith(variant.lower()):
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
                        ax.plot(x_vals, y_vals, marker='o', label=variant, color=get_variant_color(variant))
                        plotted = True

                    ax.set_title(f"{metric_group_name.replace('_', ' ').title()} – {group_label.replace('_', ' ')} ({family.title()})",
                                 fontsize=12, fontweight='bold')
                    ax.set_xlabel('Model Size', fontweight='bold')
                    ax.set_ylabel('Score', fontweight='bold')
                    ax.set_xticks(all_size_nums)
                    ax.set_xticklabels(all_size_labels)
                    ax.grid(True, alpha=0.3, linestyle='--')

                    if plotted:
                        ax.legend(loc='best', title='Promp variation',fontsize=8)
                        ymin, ymax = ax.get_ylim()
                        ax.set_ylim(0, ymax * 1.15)

                    plt.tight_layout()
                    filename = f"{metric_group_name}_{group_label}_{family}.png"
                    path = os.path.join(directory, filename)
                    fig.savefig(path, dpi=300, bbox_inches="tight")
                    print(f"   ✅ Saved {path}")
                    if show_plots:
                        plt.show()
                    plt.close(fig)

            # --- CASE 2: TextFooler group → single plot with both families ---
            elif group_label == "TextFooler":
                print(f"   📊 Plotting {group_label} (both families combined)")

                fig, ax = plt.subplots(figsize=(6, 5))
                plotted = False

                for exp_name in experiments:
                    for fam_idx, family in enumerate(["Llama3", "Qwen"]):
                        exp_values = []
                        for size_label, size_num in zip(all_size_labels, all_size_nums):
                            key = f"{family}_{size_label}"
                            if key not in model_data:
                                continue

                            metrics_dict = model_data[key]['metrics']
                            if metric_group_name not in metrics_dict:
                                continue

                            # --- match experiment name by prefix ---
                            found_key = None
                            for k in metrics_dict[metric_group_name].keys():
                                if k.lower().startswith(exp_name.lower()):
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
                        ax.plot(
                            x_vals, y_vals,
                            marker='s', linestyle='--',
                            label=family.title(),
                        )

                        plotted = True

                ax.set_title(f"{metric_group_name.replace('_', ' ').title()} – {group_label.replace('_', ' ')}",
                             fontsize=12, fontweight='bold')
                ax.set_xlabel('Model Size', fontweight='bold')
                ax.set_ylabel('Score', fontweight='bold')
                ax.set_xticks(all_size_nums)
                ax.set_xticklabels(all_size_labels)
                ax.grid(True, alpha=0.3, linestyle='--')

                if plotted:
                    ax.legend(loc='best', title='Model Family',fontsize=8)
                    ax.set_ylim(0, 1)

                plt.tight_layout()
                filename = f"{metric_group_name}_{group_label}.png"
                path = os.path.join(directory, filename)
                fig.savefig(path, dpi=300, bbox_inches="tight")
                print(f"   ✅ Saved {path}")
                if show_plots:
                    plt.show()
                plt.close(fig)