# How Model Scale Influences Post-Hoc Explainability in Large Language Models

As large language models (LLMs) grow in size to enhance performance, there remains a limited understanding of how this expansion affects their explainability.
In this project, we use counterfactuals as our primary approach to post-hoc explainability.

We evaluate multiple models across different scales by generating two types of counterfactual explanations:
 - Introspections: Self-generated counterfactuals, where the model attempts to produce an alternative input that flips its own prediction.
 - Adversarial attacks, representing minimal perturbations that change the model’s output.

The code for Self-generated counterfactuals is derived from the paper Are Self-Explanations from Large Language Models Faithful?, but we modified and extended it to suit the specific needs of our study.
The Adversarial attacks are generated via TextAttack's TextFoolerJin2019 algorithm. 

These edits are treated as explanations, and we assess their quality along two dimensions:
 - **Faithfulness** — whether the edits reliably flip the model’s prediction.
 - **Minimality** — how close, semantically similar, and contradictory the conterfactual is to the input.

We also examined a central question: Are the parts of the input that models choose to change—the features they rely on for classification—the same parts that humans consider important?
To explore this, we analyzed how model size influences the interpretability of counterfactual explanations using the ERASER benchmark, which provides human-annotated rationale spans.
By measuring how closely the counterfactual edits overlap with human-identified important tokens, we assessed the **degree of alignment between human reasoning and LLM reasoning** across models of different scales.

### Datasets

Evaluating Rationales And Simple English Reasoning (ERASER) benchmark: Standard benchmark datasets augmented with human-provided rationale annotations.

 - Movie Reviews: A collection of movie reviews labeled for sentiment (positive or negative). 
 - e-SNLI: Sentence pairs annotated for entailment, with “yes” or “no” labels.

### Models

- **LLaMA-3:** 1B, 3B, 8B  
- **Qwen-2.5:** 1.5B, 3B, 7B  


# Experiment Pipeline

## 1. Model Classification

Ask the model to classify the instance.
The figure below shows the classifier's accuracy across the models:

<div style="text-align:center;">
  <img src="results/classifiers_accuracy_vs_size.png" alt="Experiment Summary"/>
</div>

## 2. Counterfactual Generation

- Extract the adversarial attack from TextAttack.
- Run the introspect module with different task configurations. We vary two factors:

**Baseline** –  
- No flags set. The model is informed of its prediction and is prompted to generate a counterfactual. No specific perspective is adopted.

**Target Visibility** –  
- **e-implicit-target:**  
  The model is **not informed** of its prediction and must infer the target label itself.

**Perspective / Persona** –  
- **e-persona-you:**  
  The model is prompted to generate the counterfactualto reverse its own decision.

- **e-persona-human:**  
  The model is prompted to generate the counterfactualto as if it is reversing a human's decision.



## 3. Quality Evaluation

<h3 style="text-align:center;">Movie Reviews</h3>
<div style="text-align:center;">
  <img src="results/movie_results/plots/Attack_Success.png" alt="Attack Success" style="display:block; margin:0 auto;"/>
  <div style="height:40px;"></div>
  <img src="results/movie_results/plots/Introspection_Success.png" alt="Introspection Success" style="display:block; margin:0 auto;"/>
</div>

<div style="height:40px;"></div>

<div style="text-align:center;">
  <div style="display:flex; flex-wrap:wrap;margin:0 auto; line-height:0;">
    <!-- Distance Row -->
    <img src="results/movie_results/plots/distance_Introspection_Llama3.png" alt="Distance LLaMA-3" style="width:33.333%; margin:0; padding:0;"/>
    <img src="results/movie_results/plots/distance_Introspection_Qwen.png" alt="Distance Qwen" style="width:33.333%; margin:0; padding:0;"/>
    <img src="results/movie_results/plots/distance_TextFooler.png" alt="Distance TextFooler" style="width:33.333%; margin:0; padding:0;"/>
    <!-- Contradiction Row -->
    <img src="results/movie_results/plots/contradiction_Introspection_Llama3.png" alt="Contradiction LLaMA-3" style="width:33.333%; margin:0; padding:0;"/>
    <img src="results/movie_results/plots/contradiction_Introspection_Qwen.png" alt="Contradiction Qwen" style="width:33.333%; margin:0; padding:0;"/>
    <img src="results/movie_results/plots/contradiction_TextFooler.png" alt="Contradiction TextFooler" style="width:33.333%; margin:0; padding:0;"/>
    <!-- Semantic Similarity Row -->
    <img src="results/movie_results/plots/semantic_similarity_Introspection_Llama3.png" alt="Semantic Similarity LLaMA-3" style="width:33.333%; margin:0; padding:0;"/>
    <img src="results/movie_results/plots/semantic_similarity_Introspection_Qwen.png" alt="Semantic Similarity Qwen" style="width:33.333%; margin:0; padding:0;"/>
    <img src="results/movie_results/plots/semantic_similarity_TextFooler.png" alt="Semantic Similarity TextFooler" style="width:33.333%; margin:0; padding:0;"/>
  </div>
</div>


<h3 style="text-align:center;">e-SNLI</h3>
<div style="text-align:center;">
  <img src="results/esnli_results/plots/Attack_Success.png" alt="Attack Success" style="display:block; margin:0 auto;"/>
  <div style="height:40px;"></div>
  <img src="results/esnli_results/plots/Introspection_Success.png" alt="Introspection Success" style="display:block; margin:0 auto;"/>
</div>

<div style="height:40px;"></div>

<div style="text-align:center;">
  <div style="display:flex; flex-wrap:wrap; margin:0 auto; line-height:0;">
    <!-- Distance Row -->
    <img src="results/esnli_results/plots/distance_Introspection_Llama3.png" alt="Distance LLaMA-3" style="width:33.333%; margin:0; padding:0;"/>
    <img src="results/esnli_results/plots/distance_Introspection_Qwen.png" alt="Distance Qwen" style="width:33.333%; margin:0; padding:0;"/>
    <img src="results/esnli_results/plots/distance_TextFooler.png" alt="Distance TextFooler" style="width:33.333%; margin:0; padding:0;"/>
    <!-- Contradiction Row -->
    <img src="results/esnli_results/plots/contradiction_Introspection_Llama3.png" alt="Contradiction LLaMA-3" style="width:33.333%; margin:0; padding:0;"/>
    <img src="results/esnli_results/plots/contradiction_Introspection_Qwen.png" alt="Contradiction Qwen" style="width:33.333%; margin:0; padding:0;"/>
    <img src="results/esnli_results/plots/contradiction_TextFooler.png" alt="Contradiction TextFooler" style="width:33.333%; margin:0; padding:0;"/>
    <!-- Semantic Similarity Row -->
    <img src="results/esnli_results/plots/semantic_similarity_Introspection_Llama3.png" alt="Semantic Similarity LLaMA-3" style="width:33.333%; margin:0; padding:0;"/>
    <img src="results/esnli_results/plots/semantic_similarity_Introspection_Qwen.png" alt="Semantic Similarity Qwen" style="width:33.333%; margin:0; padding:0;"/>
    <img src="results/esnli_results/plots/semantic_similarity_TextFooler.png" alt="Semantic Similarity TextFooler" style="width:33.333%; margin:0; padding:0;"/>
  </div>
</div>



## 4. Human and LLM Alignment

<h3 style="text-align:center;">Movie Reviews</h3>
<div style="display:flex; flex-wrap:wrap; margin:0 auto; line-height:0;">

  <!-- Evidence Accuracy Row -->
  <img src="results/movie_results/plots/evidence_accuracy_Introspection_Llama3.png" alt="Evidence Accuracy LLaMA-3" style="width:33.333%; margin:0; padding:0;"/>
  <img src="results/movie_results/plots/evidence_accuracy_Introspection_Qwen.png" alt="Evidence Accuracy Qwen" style="width:33.333%; margin:0; padding:0;"/>
  <img src="results/movie_results/plots/evidence_accuracy_TextFooler.png" alt="Evidence Accuracy TextFooler" style="width:33.333%; margin:0; padding:0;"/>
  
  <!-- Evidence Precision Row -->
  <img src="results/movie_results/plots/evidence_precision_Introspection_Llama3.png" alt="Evidence Precision LLaMA-3" style="width:33.333%; margin:0; padding:0;"/>
  <img src="results/movie_results/plots/evidence_precision_Introspection_Qwen.png" alt="Evidence Precision Qwen" style="width:33.333%; margin:0; padding:0;"/>
  <img src="results/movie_results/plots/evidence_precision_TextFooler.png" alt="Evidence Precision TextFooler" style="width:33.333%; margin:0; padding:0;"/>
  
  <!-- Evidence Recall Row -->
  <img src="results/movie_results/plots/evidence_recall_Introspection_Llama3.png" alt="Evidence Recall LLaMA-3" style="width:33.333%; margin:0; padding:0;"/>
  <img src="results/movie_results/plots/evidence_recall_Introspection_Qwen.png" alt="Evidence Recall Qwen" style="width:33.333%; margin:0; padding:0;"/>
  <img src="results/movie_results/plots/evidence_recall_TextFooler.png" alt="Evidence Recall TextFooler" style="width:33.333%; margin:0; padding:0;"/>
  
  <!-- Evidence F1 Row -->
  <img src="results/movie_results/plots/evidence_f1_Introspection_Llama3.png" alt="Evidence F1 LLaMA-3" style="width:33.333%; margin:0; padding:0;"/>
  <img src="results/movie_results/plots/evidence_f1_Introspection_Qwen.png" alt="Evidence F1 Qwen" style="width:33.333%; margin:0; padding:0;"/>
  <img src="results/movie_results/plots/evidence_f1_TextFooler.png" alt="Evidence F1 TextFooler" style="width:33.333%; margin:0; padding:0;"/>
  
</div>

<h3 style="text-align:center;">e-SNLI</h3>
<div style="display:flex; flex-wrap:wrap; margin:0 auto; line-height:0;">

  <!-- Evidence Accuracy Row -->
  <img src="results/esnli_results/plots/evidence_accuracy_Introspection_Llama3.png" alt="Evidence Accuracy LLaMA-3" style="width:33.333%; margin:0; padding:0;"/>
  <img src="results/esnli_results/plots/evidence_accuracy_Introspection_Qwen.png" alt="Evidence Accuracy Qwen" style="width:33.333%; margin:0; padding:0;"/>
  <img src="results/esnli_results/plots/evidence_accuracy_TextFooler.png" alt="Evidence Accuracy TextFooler" style="width:33.333%; margin:0; padding:0;"/>
  
  <!-- Evidence Precision Row -->
  <img src="results/esnli_results/plots/evidence_precision_Introspection_Llama3.png" alt="Evidence Precision LLaMA-3" style="width:33.333%; margin:0; padding:0;"/>
  <img src="results/esnli_results/plots/evidence_precision_Introspection_Qwen.png" alt="Evidence Precision Qwen" style="width:33.333%; margin:0; padding:0;"/>
  <img src="results/esnli_results/plots/evidence_precision_TextFooler.png" alt="Evidence Precision TextFooler" style="width:33.333%; margin:0; padding:0;"/>
  
  <!-- Evidence Recall Row -->
  <img src="results/esnli_results/plots/evidence_recall_Introspection_Llama3.png" alt="Evidence Recall LLaMA-3" style="width:33.333%; margin:0; padding:0;"/>
  <img src="results/esnli_results/plots/evidence_recall_Introspection_Qwen.png" alt="Evidence Recall Qwen" style="width:33.333%; margin:0; padding:0;"/>
  <img src="results/esnli_results/plots/evidence_recall_TextFooler.png" alt="Evidence Recall TextFooler" style="width:33.333%; margin:0; padding:0;"/>
  
  <!-- Evidence F1 Row -->
  <img src="results/esnli_results/plots/evidence_f1_Introspection_Llama3.png" alt="Evidence F1 LLaMA-3" style="width:33.333%; margin:0; padding:0;"/>
  <img src="results/esnli_results/plots/evidence_f1_Introspection_Qwen.png" alt="Evidence F1 Qwen" style="width:33.333%; margin:0; padding:0;"/>
  <img src="results/esnli_results/plots/evidence_f1_TextFooler.png" alt="Evidence F1 TextFooler" style="width:33.333%; margin:0; padding:0;"/>
  
</div>