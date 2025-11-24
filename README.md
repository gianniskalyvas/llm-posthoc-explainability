# How Model Scale Influences Post-Hoc Explainability in Large Language Models

As large language models (LLMs) grow in size to enhance performance, there remains a limited understanding of how this expansion affects their explainability.
This project adopts counterfactuals as a primary method for post-hoc explainability, using the following definition: 
"A counterfactual explanation is a minimal edit of the input with the words or phrases crucial for classification changed, revealing what should have been different to observe the opposite outcome."

We evaluate multiple models across different scales by generating two types of counterfactual explanations:
 - Introspections: Self-generated counterfactuals, where the model attempts to produce an alternative input that flips its own prediction.
 - Adversarial attacks, representing minimal perturbations that change the model’s output.

The code for Self-generated counterfactuals is derived from the paper *Are Self-Explanations from Large Language Models Faithful?*, but has been modified and extended to suit the specific needs of our study.
The Adversarial attacks are generated via TextAttack's *TextFoolerJin2019* algorithm. 

These edits are treated as explanations, and we assess their quality along two dimensions:
 - **Faithfulness** — whether the edits reliably flip the model's prediction.
 - **Minimality** — how close, semantically similar and contradictory the counterfactual is to the input.

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

---

### Directory Structure
- `attacks/` — Adversarial attack results (TextFooler outputs)
- `introspections/` — Self-generated counterfactual results  
- `results/` — Evaluation metrics and visualization plots
- `eraserbenchmark-master/` — ERASER benchmark datasets with human annotations
- `llm-introspection-main/` — Modified introspection module (adapted from *Are Self-Explanations from Large Language Models Faithful?*)
- `textattack/` — Notebooks for generating adversarial attacks using TextAttack framework
- `wrapper/` — Model wrapper utilities for textattack

### Running Experiments
Refer to `experiment_summary.ipynb` for a complete walkthrough of the experimental pipeline.

---

# Experiment Pipeline

## 1. Model Classification

Ask the model to classify the instance.
The figure below shows the classifier's accuracy across the models:

<div style="text-align:center;">
  <img src="results/classifiers_accuracy_vs_size.png" alt="Experiment Summary" style="width:80%;"/>
</div>

## 2. Counterfactual Generation

### Pipeline Examples

<details>
<summary><b>📖 Movie Review Example </b></summary>

#### Original Review
```
michael robbins ' hardball is quite the cinematic achievement . in about two hours , we get a glancing examination of ghetto life , a funeral with a heartfelt eulogy , speeches about never giving up , a cache of cute kids ( including a fat one with asthma ) , a hard - luck gambler who finds salvation in a good woman and a climactic " big game , " where the underdogs prove to have a bigger bite than anyone ever imagined ...
```
**Classification:** `NEGATIVE` ✓

---

#### Adversarial Attack (TextFooler)
```
michael robbins ' hardball is quite the cinematic achievement . in about two hours , we [[achieved]] a [[chuckled]] examination of ghetto life , a funeral with a heartfelt eulogy , speeches about never giving up , a cache of cute kids ( including a fat one with asthma ) , a hard - luck [[keno]] who finds salvation in a good woman and a [[meteorological]] " big game , " where the underdogs prove to have a bigger bite than anyone ever imagined ...
```
**Result:** `NEGATIVE → POSITIVE`  
**Success:** ✓ Successful

---

#### Self-Generated Counterfactual (Introspection)
```
Michael Robbins' Hardball is a cinematic masterpiece. In about two hours, we get a nuanced exploration of ghetto life, a funeral with a heartfelt eulogy, speeches about perseverance, a cache of endearing kids (including a young one with asthma), a hard-luck gambler who finds redemption in a kind woman and a climactic "big game," where the underdogs prove to have a bigger impact than anyone ever imagined ... 
```
**Result:** `NEGATIVE → POSITIVE`  
**Faithfulness:** ✓ Successfully flipped prediction  
**Key Changes:** Systematically replaced negative descriptors ("quite the cinematic achievement"→"masterpiece", "glancing examination"→"nuanced exploration", "cute kids"→"endearing kids") with positive alternatives while maintaining sentence structure.

</details>

<details>
<summary><b>📖 e-SNLI Example </b></summary>

#### Original Instance
**Hypothesis:** `Several people sitting in a boat.` <br>
**Premise:**  `A group of people traveling in a small wooden boat.`  <br>
**Classification:** `YES (entailment)` ✓

---

#### Adversarial Attack (TextFooler)
**Premise:** `A group of people traveling in a small wooden [[seafaring]].`  
**Result:** `YES → NO`  
**Success:** ✓ Successful
**Key Changes:** Perturbed "boat" with "seafaring" - a semantically related but contextually incoherent substitution that breaks the premise-hypothesis alignment.

---

#### Self-Generated Counterfactual (Introspection)
**Premise:** `A group of people traveling in a small wooden airplane.`  
**Result:** `YES → NO`  
**Faithfulness:** ✓ Successfully flipped prediction  
**Key Changes:** Replaced "boat" with "airplane", changing the mode of transport.

</details>

---

### A. Extract the adversarial attack from TextAttack.
TextAttack's pipeline: 
 1. Identify prediction-critical words
 2. Generate semantically similar substitutions 
 3. Test perturbations until prediction flips while preserving coherence and similarity.

### B. Run the introspect module with multiple task configurations.

**Baseline** –  This is the standard setup with no special flags, serving as our point of comparison.
 - Flags: None
 - Description: The model is directly informed of its own initial prediction. It is then prompted to generate a minimal text edit that would change this specific prediction. It does not adopt any specific persona.


**Perspective / Persona Manipulation** – As proposed in the original paper, this changes the subject of the explanation—that is, whose decision is being reversed. 
 - Flag: ```e-persona-you```

     - Description: The model is prompted to generate the counterfactual to reverse its own decision. The explanation is framed from the LLM's perspective (e.g., *"What would have to change for you to change your mind?"*).

 - Flag: ```e-persona-human```

     - Description: The model is prompted to generate the counterfactual as if it is reversing a human's decision. The explanation is framed from an external, human-centric perspective (e.g.,*"What would have to change for a human to change their mind?"*).


**Target Visibility Manipulation** 
 - Flag: ```e-implicit-target```
 - Description: The model is not explicitly told what its initial prediction was. It must first infer the target label from the context before it can generate a counterfactual to reverse it. This adds a step of reasoning and tests the model's understanding of the decision boundary.

Finally, it is important to note that the ```e-implicit-target``` manipulation, which controls target visibility, is an orthogonal dimension to the persona manipulation. Therefore, the ```e-implicit-target``` flag can be combined with either the ```e-persona-you``` or ```e-persona-human``` flags to create composite experimental conditions that test the interaction of these factors.

## 3. Quality Evaluation

### Faithfulness Metrics

***Attack Success Rate***: Percentage of adversarial attacks that successfully flip the model's prediction. Lower values indicate more robust models.

***Introspection Faithfulness***: Percentage of self-generated counterfactuals that successfully flip the model's prediction.

### Minimality Metrics

***Distance***: Normalized edit distance (1 - SequenceMatcher ratio). Lower values indicate more minimal changes.

***Semantic Similarity***: Cosine similarity of `all-mpnet-base-v2` embeddings. Higher values indicate better semantic preservation.

***Contradiction***: Contradiction probability from `roberta-large-mnli`. Higher values indicate stronger logical contradictions.

---

<h2 style="text-align:left;">Movie Reviews</h2>
<div style="text-align:center;">
  <img src="results/movie_results/plots/Attack_Success.png" alt="Attack Success" style="display:block; margin:0 auto; width:70%;padding-bottom:40px; "/>
  <p style="padding-bottom:40px; "><em>TextFooler achieves high attack success rates across all model sizes, with larger models showing slightly better robustness.</em></p>
  <img src="results/movie_results/plots/Introspection_Success.png" alt="Introspection Success" style="display:block; margin:0 auto; width:96%; padding-bottom:40px; "/>
  <p><em>
    1. Larger model size improves faithfulness.<br>
    2. Prompt variations do not substantially affect the model’s behavior.<br>
    3. Smaller models do not benefit from different task configurations.<br>
    4. Larger models benefit significantly—baseline performs the worst, while the e-implicit-target configuration yields the best results.
  </em></p>
</div>



<div style="height:40px;"></div>

<div style="text-align:center;">
  <div style="display:flex; flex-wrap:wrap; justify-content:center; margin:0 auto; line-height:0; padding-bottom:40px;  ">
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
  <p><em>
    1. Adversarial attack constraints keep quality stable across all metrics. <br>
    2. For distance-based evaluations, mid- and large-sized models behave consistently, whereas smaller models exhibit substantial variability. <br>
    3. The e-implicit-target configuration produces more contradictory counterfactuals. <br>
    4. Semantically, smaller models produce varying results across different setups.
  </em></p>
</div>



<h2 style="text-align:left;">e-SNLI</h2>
<div style="text-align:center;">
  <img src="results/esnli_results/plots/Attack_Success.png" alt="Attack Success" style="display:block; margin:0 auto; width:70%; padding-bottom:40px;"/>
  <p style="padding-bottom:40px; "><em>Mid-sized models appear to be the most robust.<em></p>
  <img src="results/esnli_results/plots/Introspection_Success.png" alt="Introspection Success" style="display:block; margin:0 auto; width:96%; padding-bottom:40px;"/>
  <p><em>
    1. Increasing model size improves faithfulness.<br>
    2. The baseline prompt consistently achieves the best performance on this task.<br>
    3. Introducing a perspective makes the model more inconsistent in this task.<br>
    4. Hiding the target yields the weakest results.  
  </em></p>
</div>

<div style="height:40px;"></div>

<div style="text-align:center;">
  <div style="display:flex; flex-wrap:wrap; justify-content:center; margin:0 auto; line-height:0; padding-bottom:40px;">
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
  <p><em>
    1. Adversarial attack constraints keep quality stable across all metrics.<br>
    2. Llama's behaviour is not affected by the prompt variations, with mid-sized Llama models creating the most minimally modified counterfactuals. <br>
    3. Target visibility affects Qwen models differently: harms small, boosts mid-sized, no effect on large. <br>
    4. Hiding the target produces more contradictory counterfactuals.
  </em></p>
</div>


## 4. Human and LLM Alignment

### Metric Definitions

To measure how well counterfactual edits align with human-annotated evidence spans, we compute token-level accuracy, precision, recall, and f1:

- **True Positive (TP)** — a token that was **modified** *and* lies **inside** a human-annotated evidence span.  
- **False Positive (FP)** — a token that was **modified** but lies **outside** any human-annotated evidence span.  
- **False Negative (FN)** — a token that lies **inside** a human-annotated evidence span but was **not** modified.
- **True Negative (TN)** — a token that was **not modified** *and* lies **outside** any human-annotated evidence span.

**Accuracy**: What percent of all tokens are correctly classified (either modified within evidence or unmodified outside evidence)?

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

**Precision**: What percent of the perturbations in the input are *evidence* supported?

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

**Recall**: What percent of the *human evidence spans* are covered by the perturbations in the input?  

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

**F1 Score**: The harmonic mean of precision and recall, providing a balanced measure of alignment quality.

$$
\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

---

<h2 style="text-align:left;">Movie Reviews</h2>
<div style="text-align:center;">
  <div style="display:flex; flex-wrap:wrap; justify-content:center; margin:0 auto; line-height:0; padding-bottom:40px;">
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
  <p><em>
    1. Attacks show a slight decrease in their preference for perturbing human-annotated tokens as model size increases, across both model families. <br>
    2. Smaller Llama is affected from prompt variations benefitting from them in general. while bigger version where more consistent <br>
  </em></p>  
</div>

<h2 style="text-align:left;">e-SNLI</h2>
<div style="text-align:center;">
  <div style="display:flex; flex-wrap:wrap; justify-content:center; margin:0 auto; line-height:0; padding-bottom:40px;">
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
  <p><em>
    1. In this task it more clear that human-llm alignment grows with sise <br>
    2. Only target visibility manipulation had a minor impact <br>
  </em></p>  
</div>

## References

This project builds upon and extends the methodology from:

**Lanham, T., Chen, A., Radhakrishnan, A., Steiner, B., Denison, C., & Hernandez, D.** (2023). *Are Self-Explanations from Large Language Models Faithful?* [Original introspection framework]

**DeYoung, J., Jain, S., Rajani, N. F., Lehman, E., Xiong, C., Socher, R., & Wallace, B. C.** (2020). *ERASER: A Benchmark to Evaluate Rationalized NLP Models.* ACL 2020. [Human rationale annotations]

**Jin, D., Jin, Z., Zhou, J. T., & Szolovits, P.** (2020). *Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment.* AAAI 2020. [TextFooler adversarial attack algorithm]
