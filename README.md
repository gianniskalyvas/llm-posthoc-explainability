# How Model Scale Influences Post-Hoc Explainability in Large Language Models

As large language models (LLMs) grow in size to enhance performance, there remains a limited understanding of how this expansion affects their explainability.
In this project, we use counterfactuals as our primary approach to post-hoc explainability.
We use the following definition of ‘counterfactual explanation’: “A counterfactual explanation is a minimal edit of the input with the words or phrases crucial for classification changed, revealing what should have been different to observe the opposite outcome.”

We evaluate multiple models across different scales by generating two types of counterfactual explanations:
 - Introspections: Self-generated counterfactuals, where the model attempts to produce an alternative input that flips its own prediction.
 - Adversarial attacks, representing minimal perturbations that change the model’s output.

The code for Self-generated counterfactuals is derived from the paper *Are Self-Explanations from Large Language Models Faithful?*, but has been modified and extended to suit the specific needs of our study.
The Adversarial attacks are generated via TextAttack's *TextFoolerJin2019* algorithm. 

These edits are treated as explanations, and we assess their quality along two dimensions:
 - **Faithfulness** — whether the edits reliably flip the model’s prediction.
 - **Minimality** — how close, semantically similar and contradictory the conterfactual is to the input.

We also examined a central question: Are the parts of the input that models choose to change—the features they rely on for classification—the same parts that humans consider important?
To explore this, we analyzed how model size influences the interpretability of counterfactual explanations using the ERASER benchmark, which provides human-annotated rationale spans.
By measuring how closely the counterfactual edits overlap with human-identified important tokens, we assessed the **degree of alignment between human reasoning and LLM reasoning** across models of different scales.

### Datasets

ERASER benchmark: Standard benchmark datasets augmented with human-provided rationale annotations.

 - Movie Reviews: A collection of movie reviews labeled for sentiment (positive or negative). 
 - e-SNLI: Sentence pairs labeled for entailment (yes or no).

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

### A. Extract the adversarial attack from TextAttack.
### B. Run the introspect module with multiple task configurations.

**Baseline** –  This is the standard setup with no special flags, serving as our point of comparison.
 - Flags: None
 - Description: The model is directly informed of its own initial prediction. It is then prompted to generate a minimal text edit that would change this specific prediction. It does not adopt any specific persona.


**Perspective / Persona Manipulation** – As proposed in the original paper, this changes the subject of the explanation—that is, whose decision is being reversed. 
 - Flag: ```e-persona-you```

     - Description: The model is prompted to generate the counterfactual to reverse its own decision. The explanation is framed from the LLM's perspective (e.g., "What would have to change for you to change your mind?").

 - Flag: ```e-persona-human```

     - Description: The model is prompted to generate the counterfactual as if it is reversing a human's decision. The explanation is framed from an external, human-centric perspective (e.g., "What would have to change for a human to change their mind?").


**Target Visibility Manipulation** 
 - Flag: ```e-implicit-target```
 - Description: The model is not explicitly told what its initial prediction was. It must first infer the target label from the context before it can generate a counterfactual to reverse it. This adds a step of reasoning and tests the model's understanding of the decision boundary.

Finally, it is important to note that the ```e-implicit-target``` manipulation, which controls target visibility, is an orthogonal dimension to the persona manipulation. Therefore, the ```e-implicit-target``` flag can be combined with either the ```e-persona-you``` or ```e-persona-human``` flags to create composite experimental conditions that test the interaction of these factors.

## 3. Quality Evaluation

<h1 style="text-align:center;">Movie Reviews</h1>
<div style="text-align:center;">
  <img src="results/movie_results/plots/Attack_Success.png" alt="Attack Success" style="display:block; margin:0 auto;"/>
  <div style="height:40px;"></div>
  <img src="results/movie_results/plots/Introspection_Success.png" alt="Introspection Success" style="display:block; margin:0 auto;"/>
</div>

<div style="height:40px;"></div>

<div style="text-align:center;">
  <div style="display:flex; flex-wrap:wrap;margin:0 auto; line-height:0;">
    <!-- Distance Row -->
    <img src="results/movie_results/plots/distance_Introspection_Llama3.png" alt="Distance LLaMA-3" style="width:32%; margin:0; padding:0;"/>
    <img src="results/movie_results/plots/distance_Introspection_Qwen.png" alt="Distance Qwen" style="width:32%; margin:0; padding:0;"/>
    <img src="results/movie_results/plots/distance_TextFooler.png" alt="Distance TextFooler" style="width:32%; margin:0; padding:0;"/>
    <!-- Contradiction Row -->
    <img src="results/movie_results/plots/contradiction_Introspection_Llama3.png" alt="Contradiction LLaMA-3" style="width:32%; margin:0; padding:0;"/>
    <img src="results/movie_results/plots/contradiction_Introspection_Qwen.png" alt="Contradiction Qwen" style="width:32%; margin:0; padding:0;"/>
    <img src="results/movie_results/plots/contradiction_TextFooler.png" alt="Contradiction TextFooler" style="width:32%; margin:0; padding:0;"/>
    <!-- Semantic Similarity Row -->
    <img src="results/movie_results/plots/semantic_similarity_Introspection_Llama3.png" alt="Semantic Similarity LLaMA-3" style="width:32%; margin:0; padding:0;"/>
    <img src="results/movie_results/plots/semantic_similarity_Introspection_Qwen.png" alt="Semantic Similarity Qwen" style="width:32%; margin:0; padding:0;"/>
    <img src="results/movie_results/plots/semantic_similarity_TextFooler.png" alt="Semantic Similarity TextFooler" style="width:32%; margin:0; padding:0;"/>
  </div>
</div>


<h1 style="text-align:center;">e-SNLI</h1>
<div style="text-align:center;">
  <img src="results/esnli_results/plots/Attack_Success.png" alt="Attack Success" style="display:block; margin:0 auto;"/>
  <div style="height:40px;"></div>
  <img src="results/esnli_results/plots/Introspection_Success.png" alt="Introspection Success" style="display:block; margin:0 auto;"/>
</div>

<div style="height:40px;"></div>

<div style="text-align:center;">
  <div style="display:flex; flex-wrap:wrap; margin:0 auto; line-height:0;">
    <!-- Distance Row -->
    <img src="results/esnli_results/plots/distance_Introspection_Llama3.png" alt="Distance LLaMA-3" style="width:32%; margin:0; padding:0;"/>
    <img src="results/esnli_results/plots/distance_Introspection_Qwen.png" alt="Distance Qwen" style="width:32%; margin:0; padding:0;"/>
    <img src="results/esnli_results/plots/distance_TextFooler.png" alt="Distance TextFooler" style="width:32%; margin:0; padding:0;"/>
    <!-- Contradiction Row -->
    <img src="results/esnli_results/plots/contradiction_Introspection_Llama3.png" alt="Contradiction LLaMA-3" style="width:32%; margin:0; padding:0;"/>
    <img src="results/esnli_results/plots/contradiction_Introspection_Qwen.png" alt="Contradiction Qwen" style="width:32%; margin:0; padding:0;"/>
    <img src="results/esnli_results/plots/contradiction_TextFooler.png" alt="Contradiction TextFooler" style="width:32%; margin:0; padding:0;"/>
    <!-- Semantic Similarity Row -->
    <img src="results/esnli_results/plots/semantic_similarity_Introspection_Llama3.png" alt="Semantic Similarity LLaMA-3" style="width:32%; margin:0; padding:0;"/>
    <img src="results/esnli_results/plots/semantic_similarity_Introspection_Qwen.png" alt="Semantic Similarity Qwen" style="width:32%; margin:0; padding:0;"/>
    <img src="results/esnli_results/plots/semantic_similarity_TextFooler.png" alt="Semantic Similarity TextFooler" style="width:32%; margin:0; padding:0;"/>
  </div>
</div>



## 4. Human and LLM Alignment

<h1 style="text-align:center;">Movie Reviews</h1>
<div style="display:flex; flex-wrap:wrap; margin:0 auto; line-height:0;">

  <!-- Evidence Accuracy Row -->
  <img src="results/movie_results/plots/evidence_accuracy_Introspection_Llama3.png" alt="Evidence Accuracy LLaMA-3" style="width:32%; margin:0; padding:0;"/>
  <img src="results/movie_results/plots/evidence_accuracy_Introspection_Qwen.png" alt="Evidence Accuracy Qwen" style="width:32%; margin:0; padding:0;"/>
  <img src="results/movie_results/plots/evidence_accuracy_TextFooler.png" alt="Evidence Accuracy TextFooler" style="width:32%; margin:0; padding:0;"/>
  
  <!-- Evidence Precision Row -->
  <img src="results/movie_results/plots/evidence_precision_Introspection_Llama3.png" alt="Evidence Precision LLaMA-3" style="width:32%; margin:0; padding:0;"/>
  <img src="results/movie_results/plots/evidence_precision_Introspection_Qwen.png" alt="Evidence Precision Qwen" style="width:32%; margin:0; padding:0;"/>
  <img src="results/movie_results/plots/evidence_precision_TextFooler.png" alt="Evidence Precision TextFooler" style="width:32%; margin:0; padding:0;"/>
  
  <!-- Evidence Recall Row -->
  <img src="results/movie_results/plots/evidence_recall_Introspection_Llama3.png" alt="Evidence Recall LLaMA-3" style="width:32%; margin:0; padding:0;"/>
  <img src="results/movie_results/plots/evidence_recall_Introspection_Qwen.png" alt="Evidence Recall Qwen" style="width:32%; margin:0; padding:0;"/>
  <img src="results/movie_results/plots/evidence_recall_TextFooler.png" alt="Evidence Recall TextFooler" style="width:32%; margin:0; padding:0;"/>
  
  <!-- Evidence F1 Row -->
  <img src="results/movie_results/plots/evidence_f1_Introspection_Llama3.png" alt="Evidence F1 LLaMA-3" style="width:32%; margin:0; padding:0;"/>
  <img src="results/movie_results/plots/evidence_f1_Introspection_Qwen.png" alt="Evidence F1 Qwen" style="width:32%; margin:0; padding:0;"/>
  <img src="results/movie_results/plots/evidence_f1_TextFooler.png" alt="Evidence F1 TextFooler" style="width:32%; margin:0; padding:0;"/>
  
</div>

<h1 style="text-align:center;">e-SNLI</h1>
<div style="display:flex; flex-wrap:wrap; margin:0 auto; line-height:0;">

  <!-- Evidence Accuracy Row -->
  <img src="results/esnli_results/plots/evidence_accuracy_Introspection_Llama3.png" alt="Evidence Accuracy LLaMA-3" style="width:32%; margin:0; padding:0;"/>
  <img src="results/esnli_results/plots/evidence_accuracy_Introspection_Qwen.png" alt="Evidence Accuracy Qwen" style="width:32%; margin:0; padding:0;"/>
  <img src="results/esnli_results/plots/evidence_accuracy_TextFooler.png" alt="Evidence Accuracy TextFooler" style="width:32%; margin:0; padding:0;"/>
  
  <!-- Evidence Precision Row -->
  <img src="results/esnli_results/plots/evidence_precision_Introspection_Llama3.png" alt="Evidence Precision LLaMA-3" style="width:32%; margin:0; padding:0;"/>
  <img src="results/esnli_results/plots/evidence_precision_Introspection_Qwen.png" alt="Evidence Precision Qwen" style="width:32%; margin:0; padding:0;"/>
  <img src="results/esnli_results/plots/evidence_precision_TextFooler.png" alt="Evidence Precision TextFooler" style="width:32%; margin:0; padding:0;"/>
  
  <!-- Evidence Recall Row -->
  <img src="results/esnli_results/plots/evidence_recall_Introspection_Llama3.png" alt="Evidence Recall LLaMA-3" style="width:32%; margin:0; padding:0;"/>
  <img src="results/esnli_results/plots/evidence_recall_Introspection_Qwen.png" alt="Evidence Recall Qwen" style="width:32%; margin:0; padding:0;"/>
  <img src="results/esnli_results/plots/evidence_recall_TextFooler.png" alt="Evidence Recall TextFooler" style="width:32%; margin:0; padding:0;"/>
  
  <!-- Evidence F1 Row -->
  <img src="results/esnli_results/plots/evidence_f1_Introspection_Llama3.png" alt="Evidence F1 LLaMA-3" style="width:32%; margin:0; padding:0;"/>
  <img src="results/esnli_results/plots/evidence_f1_Introspection_Qwen.png" alt="Evidence F1 Qwen" style="width:32%; margin:0; padding:0;"/>
  <img src="results/esnli_results/plots/evidence_f1_TextFooler.png" alt="Evidence F1 TextFooler" style="width:32%; margin:0; padding:0;"/>
  
</div>