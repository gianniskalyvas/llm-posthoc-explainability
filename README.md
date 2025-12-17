# How Model Scale Influences Post-Hoc Explainability in Large Language Models

As large language models (LLMs) grow in size to enhance performance, there remains a limited understanding of how this expansion affects their explainability.
This project adopts counterfactuals as a primary method for post-hoc explainability, using the following definition: 
"A counterfactual explanation is a minimal edit of the input with the words or phrases crucial for classification changed, revealing what should have been different to observe the opposite outcome."

We evaluate multiple models across different scales by generating counterfactual explanations as in *Are Self-Explanations from Large Language Models Faithful?*.
We extended the framework’s abstract classes to support newer models and our custom datasets, and to better align the framework with the specific requirements of our study.

We analyze the impact of model scale on the quality of post-hoc counterfactual explanations.

### Datasets

ERASER benchmark: Standard benchmark datasets augmented with human-provided rationale annotations.

 - Movie Reviews: A collection of movie reviews labeled for sentiment (positive or negative). 
 - e-SNLI: Sentence pairs labeled for entailment (yes or no).

### Models

- **LLaMA-3:** 1B, 3B, 8B, 70B  
- **Qwen-2.5:** 1.5B, 3B, 7B, 14B, 32B, 72B  
- **t5-small** trained according to CREST

The selected models enable comparisons across scales within the same architecture, with CREST serving as a trained rationalization baseline rather than a generative self-explanation model.

---

### Directory Structure
- `introspections/` — Self-generated counterfactual results  
- `results/` — Evaluation metrics and visualization plots
- `eraserbenchmark-master/` — ERASER benchmark datasets with human annotations
- `llm-introspection-main/` — Modified introspection module (adapted from *Are Self-Explanations from Large Language Models Faithful?*)
- `crest/` — CREST rationalizers


# Experiment Pipeline

Refer to `experiment_summary.ipynb` for a complete walkthrough of the experimental pipeline.

```mermaid
flowchart LR
    A[Instance]

    %% Left branch: Explanations
    A --> B(ERASER)
    B --> C[Human<br/>Annotations]


    A --> F(CREST<br/>Explainer)
    F --> G[Rationales]


    %% Middle: Prediction
    A --> J(Model/<br/>Classifier)
    J --> K[Prediction]

    %% Right branch: Counterfactuals
    A --> M(Model/<br/>Counterfactual<br/>Generator)
    L[Persona] --> M
    K --> W{Visible<br/>Target?}
    W --> |Yes|M

    M --> N[Self-Generated<br/>Counterfactual]
    N --> O(Model/<br/>Classifier)
    O --> P[Counterfactual<br/>Prediction]

    %% Evaluation logic
    K --> Q{Same?}
    P --> Q
    
    A --> T(editdistance)
    N --> T
    A --> IA(mpnet)
    N --> IA
    A --> IC(roberta_mnli)
    N --> IC

    G --> H(Measure<br/>Precision)
    N --> H
    C --> D(Measure<br/>Precision)
    N --> D

    subgraph IB[Evaluation]

      subgraph IE[Intrinsic]
          Q -->|Yes| R[Unfaithful]
          Q -->|No| S[Faithful]
          T --> U[Closeness]
          IA --> V[Semantic<br/>Similarity]
          IC --> ID[Contradiction]
      end

      subgraph IF[Precision]
        H --> I[CREST<br/>Precision]
        D --> E[Human<br/>Precision]
      end

    end

```


## 1. Classifiers

<div style="text-align:center;">
  <img src="results/classifier_accuracy_comparison.png" alt="Experiment Summary" style="width:80%;"/>
</div>

## 2. Counterfactuals

### Examples

<details>
<summary><b>📖 Movie Review Example </b></summary>

#### Original Review
```
michael robbins ' hardball is quite the cinematic achievement . in about two hours , we get a glancing examination of ghetto life , a funeral with a heartfelt eulogy , speeches about never giving up , a cache of cute kids ( including a fat one with asthma ) , a hard - luck gambler who finds salvation in a good woman and a climactic " big game , " where the underdogs prove to have a bigger bite than anyone ever imagined ...
```
**Classification:** `NEGATIVE` ✓

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

#### Self-Generated Counterfactual (Introspection)
**Premise:** `A group of people traveling in a small wooden airplane.`  
**Result:** `YES → NO`  
**Faithfulness:** ✓ Successfully flipped prediction  
**Key Changes:** Replaced "boat" with "airplane", changing the mode of transport.

</details>

### Introspect

**Perspective / Persona Manipulation** – subject of the explanation
 - ```e-persona-you``` (e.g., *"What would have to change for you to change your mind?"*).

 - ```e-persona-human``` (e.g.,*"What would have to change for a human to change their mind?"*).


**Target Visibility Manipulation** 
 - ```e-implicit-target``` (The model is not told what its initial prediction was.)
 

```e-implicit-target``` flag can be combined with ```e-persona-you``` or ```e-persona-human```.


### CREST

We train a CREST rationalizer on each task that produces a token-level mask indicating features relevant to the prediction, allowing us to examine whether counterfactual edits perturb tokens identified as causally important.

## 3. Evaluation

 - *Faithfulness*: Percentage of self-generated counterfactuals that successfully flip the model's prediction.

 - *Closeness*: Normalized edit distance.

 - *Semantic Similarity*: Cosine similarity of `all-mpnet-base-v2` embeddings.

 - *Contradiction*: Contradiction probability from `roberta-large-mnli`.

<h2 style="text-align:left;">Movie Reviews</h2>
<div style="text-align:center;">
  <img src="results/movie_results/plots/Introspection_Success.png" alt="Introspection Success" style="display:block; margin:0 auto; width:96%; padding-bottom:40px; "/>
</div>



<div style="height:40px;"></div>

<div style="text-align:center;">
  <div style="display:flex; flex-wrap:wrap; justify-content:center; margin:0 auto; line-height:0; padding-bottom:40px;  ">
    <!-- Distance Row -->
    <img src="results/movie_results/plots/closeness_Introspection_Llama3.png" alt="Distance LLaMA-3" style="width:40%; margin:0; padding:0;"/>
    <img src="results/movie_results/plots/closeness_Introspection_Qwen.png" alt="Distance Qwen" style="width:40%; margin:0; padding:0;"/>
    <!-- Contradiction Row -->
    <img src="results/movie_results/plots/contradiction_Introspection_Llama3.png" alt="Contradiction LLaMA-3" style="width:40%; margin:0; padding:0;"/>
    <img src="results/movie_results/plots/contradiction_Introspection_Qwen.png" alt="Contradiction Qwen" style="width:40%; margin:0; padding:0;"/>
    <!-- Semantic Similarity Row -->
    <img src="results/movie_results/plots/semantic_similarity_Introspection_Llama3.png" alt="Semantic Similarity LLaMA-3" style="width:40%; margin:0; padding:0;"/>
    <img src="results/movie_results/plots/semantic_similarity_Introspection_Qwen.png" alt="Semantic Similarity Qwen" style="width:40%; margin:0; padding:0;"/>
  </div>
</div>



<h2 style="text-align:left;">e-SNLI</h2>
<div style="text-align:center;">
  <img src="results/esnli_results/plots/Introspection_Success.png" alt="Introspection Success" style="display:block; margin:0 auto; width:96%; padding-bottom:40px;"/>
</div>

<div style="height:40px;"></div>

<div style="text-align:center;">
  <div style="display:flex; flex-wrap:wrap; justify-content:center; margin:0 auto; line-height:0; padding-bottom:40px;">
    <!-- Distance Row -->
    <img src="results/esnli_results/plots/closeness_Introspection_Llama3.png" alt="Distance LLaMA-3" style="width:40%; margin:0; padding:0;"/>
    <img src="results/esnli_results/plots/closeness_Introspection_Qwen.png" alt="Distance Qwen" style="width:40%; margin:0; padding:0;"/>
    <!-- Contradiction Row -->
    <img src="results/esnli_results/plots/contradiction_Introspection_Llama3.png" alt="Contradiction LLaMA-3" style="width:40%; margin:0; padding:0;"/>
    <img src="results/esnli_results/plots/contradiction_Introspection_Qwen.png" alt="Contradiction Qwen" style="width:40%; margin:0; padding:0;"/>
    <!-- Semantic Similarity Row -->
    <img src="results/esnli_results/plots/semantic_similarity_Introspection_Llama3.png" alt="Semantic Similarity LLaMA-3" style="width:40%; margin:0; padding:0;"/>
    <img src="results/esnli_results/plots/semantic_similarity_Introspection_Qwen.png" alt="Semantic Similarity Qwen" style="width:40%; margin:0; padding:0;"/>
  </div>
</div>


## 4. Precision

### Metric Definitions

To measure how well counterfactual edits align with evidence spans, we compute word-level precision:

- **True Positive (TP)** — a word that was **modified** *and* lies **inside** a  evidence span.  
- **False Positive (FP)** — a word that was **modified** but lies **outside** any evidence span.  

Precision = TP / (TP + FP): What percent of the perturbations in the input are evidence supported?


We consider two types of evidence spans.
 - Human-annotation provided by ERASER
 - Rationales generated by trained CREST Explainer


<h2 style="text-align:left;">e-Movies</h2>
<div style="text-align:center;">
  <div style="display:flex; flex-wrap:wrap; justify-content:center; margin:0 auto; line-height:0; padding-bottom:40px;">
    <!-- Evidence Accuracy Row -->
    <img src="results/movie_results/plots/evidence_precision_Introspection_Llama3.png" alt="Evidence Precision LLaMA-3" style="width:40%; margin:0; padding:0;"/>
    <img src="results/movie_results/plots/evidence_precision_Introspection_Qwen.png" alt="Evidence Precision Qwen" style="width:40%; margin:0; padding:0;"/>
    <!-- Evidence Recall Row -->
    <img src="results/movie_results/plots/crest_precision_Introspection_Llama3.png" alt="Crest F1 LLaMA-3" style="width:40%; margin:0; padding:0;"/>
    <img src="results/movie_results/plots/crest_precision_Introspection_Qwen.png" alt="Crest F1 Qwen" style="width:40%; margin:0; padding:0;"/>
  </div>

</div>

<h2 style="text-align:left;">e-SNLI</h2>
<div style="text-align:center;">
  <div style="display:flex; flex-wrap:wrap; justify-content:center; margin:0 auto; line-height:0; padding-bottom:40px;">
    <!-- Evidence Precision Row -->
    <img src="results/esnli_results/plots/evidence_precision_Introspection_Llama3.png" alt="Evidence Precision LLaMA-3" style="width:40%; margin:0; padding:0;"/>
    <img src="results/esnli_results/plots/evidence_precision_Introspection_Qwen.png" alt="Evidence Precision Qwen" style="width:40%; margin:0; padding:0;"/>
    <!-- Evidence F1 Row -->
    <img src="results/esnli_results/plots/crest_precision_Introspection_Llama3.png" alt="Crest Precision LLaMA-3" style="width:40%; margin:0; padding:0;"/>
    <img src="results/esnli_results/plots/crest_precision_Introspection_Qwen.png" alt="Crest Precision Qwen" style="width:40%; margin:0; padding:0;"/>
  </div>
</div>


## References

This project builds upon and extends the methodology from:

Lanham, T., Chen, A., Radhakrishnan, A., Steiner, B., Denison, C., & Hernandez, D. (2023). *Are Self-Explanations from Large Language Models Faithful?* [Original introspection framework]

DeYoung, J., Jain, S., Rajani, N. F., Lehman, E., Xiong, C., Socher, R., & Wallace, B. C. (2020). *ERASER: A Benchmark to Evaluate Rationalized NLP Models.* ACL 2020. [Human rationale annotations]

Treviso, M., Ross, A., Guerreiro, N. M., & Martins, A. F. T. (2023). *CREST: A Joint Framework for Rationalization and Counterfactual Text Generation.* 

