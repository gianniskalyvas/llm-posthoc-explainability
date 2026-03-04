# Post-Hoc Explainability in LLMs

As large language models (LLMs) grow in size to enhance performance, there remains limited understanding of how this scaling affects their explainability.

This project studies **counterfactual explanations** as a method for post-hoc explainability. We adopt the following definition:

> A counterfactual explanation is a minimal edit of the input in which words or phrases crucial for classification are changed, revealing what would need to differ to obtain the opposite prediction.

Our study analyzes how **model scale and prompting strategies** influence the quality of generated counterfactual explanations.

Faithful counterfactual reasoning appears to emerge only beyond a certain model scale. Prompting can guide how explanations are expressed, but it cannot create reasoning abilities that the model does not already possess. Consequently, counterfactual explanations reflect an interaction between internal model capacity and external prompt design rather than providing a transparent view into the model’s reasoning.



# Setup

### Datasets

ERASER benchmark: Standard benchmark datasets augmented with human-provided rationale annotations.

 - Movie Reviews: A collection of movie reviews labeled for sentiment (positive or negative). 
 - e-SNLI: Sentence pairs labeled for entailment (yes or no).

### LLMs 

- **LLaMA-3:** 1B, 3B, 8B, 70B  
- **Qwen-2.5:** 1.5B, 3B, 7B, 14B, 32B, 72B  



# Pipeline


<div style="text-align:center;">
  <img src="experimental-pipeline.png" alt="pipeline" style="width:100%;"/>
</div>



## Counterfactual Generation

Generation is organized into three stages: original prediction,counterfactual generation, and counterfactual evaluation. All stages are implemented using a single large
language model (LLM), which assumes different roles through distinct prompting configurations.

### Prompt Configurations

**Baseline** -  serves as a point of referance. 

The model is explicitly provided with the target label—defined as the opposite of its prediction in Stage I—and is instructed to modify the original instance accordingly. The prompt further includes a formal definition of a counterfactual to encourage adherence to minimality and label-flipping principles.

**Persona Manipulation** – subject of the explanation
 - ```e-persona-you``` (e.g., *"What would have to change for you to change your mind?"*).

 - ```e-persona-human``` (e.g.,*"What would have to change for a human to change their mind?"*).


**Target Visibility** 
 - ```e-implicit-target``` (The model is not told what its initial prediction was.)
 
**Chat history** 

the initial classification and the counterfactual generation are performed within the same chat history to eximen wether context influences behavior.

**Chain of thought Prompting** 

The model is first instructed to identify the decision-relevant features and then to generate a counterfactual restricting its changes in the identified features.


## Preliminary Evaluation

Before analyzing counterfactual explanations, we perform a preliminary evaluation of the models on the underlying classification tasks (e-Movies and e-SNLI). Establishing reliable baseline accuracy is important to ensure that the generated rationales and counterfactual explanations are meaningful. Overall, classification performance improves with model size, while the CREST rationalizer achieves competitive accuracy despite operating under constraints.

<div style="text-align:center;">
  <img src="results/classifiers/classifier_accuracy_comparison.png" alt="Experiment Summary" style="width:100%;"/>
</div>



## Main Evaluation Protocol

### Faithfulness

Percentage of self-generated counterfactuals that successfully flip the model's prediction.


### Minimality 

We evaluate the extent to which a counterfactual explanation remains close to the original input. We consider multiple complementary instantiations of such a distance:

 - *Closeness*: Normalized edit distance.

 - *Semantic Similarity*: Cosine similarity of `all-mpnet-base-v2` embeddings.

 - *Contradiction*: Contradiction probability from `roberta-large-mnli`.

### Evidence-Supported Modification Precision (ESMP)


We examine to what extent LLM modify only decision-relevant parts of the input.

Do this end we leverage human annotations as provided by ERASER.

To this end we also train a rationalizer to extract a rationale sufficient for prediction. 
Its high accuracy although its constrains suggest it sucessfully identifies decision-relevant features.




<h1 style="text-align:left;">Faithfulness</h2>

<h2 style="text-align:left;">Movie Reviews</h2>
<div style="text-align:center;">
  <img src="results/movie_results/plots/faithfulness_vs_model_size.png" alt="Faithfulness" style="display:block; margin:0 auto; width:100%; padding-bottom:40px; "/>
  <img src="results/movie_results/plots/legend.png" >
</div>

<h2 style="text-align:left;">e-SNLI</h2>
<div style="text-align:center;">
  <img src="results/esnli_results/plots/faithfulness_vs_model_size.png" alt="Faithfulness" style="display:block; margin:0 auto; width:100%; padding-bottom:40px; "/>
  <img src="results/movie_results/plots/legend.png">
</div>



<h1 style="text-align:left;">Minimality</h2>

<h2>Movie Reviews</h2>

<table>
<tr>
<th>LLaMA-3</th>
<th>Qwen</th>
</tr>

<tr>
<td><img src="results/movie_results/plots/Closeness_Llama3.png" width="100%"></td>
<td><img src="results/movie_results/plots/Closeness_Qwen.png" width="100%"></td>
</tr>

<tr>
<td><img src="results/movie_results/plots/Semantic Similarity_Llama3.png" width="100%"></td>
<td><img src="results/movie_results/plots/Semantic Similarity_Qwen.png" width="100%"></td>
</tr>

<tr>
<td><img src="results/movie_results/plots/Contradiction_Llama3.png" width="100%"></td>
<td><img src="results/movie_results/plots/Contradiction_Qwen.png" width="100%"></td>
</tr>
</table>
<img src="results/movie_results/plots/legend.png">


<h2>e-SNLI</h2>

<table>
<tr>
<th>LLaMA-3</th>
<th>Qwen</th>
</tr>

<tr>
<td><img src="results/esnli_results/plots/Closeness_Llama3.png" width="100%"></td>
<td><img src="results/esnli_results/plots/Closeness_Qwen.png" width="100%"></td>
</tr>

<tr>
<td><img src="results/esnli_results/plots/Semantic Similarity_Llama3.png" width="100%"></td>
<td><img src="results/esnli_results/plots/Semantic Similarity_Qwen.png" width="100%"></td>
</tr>

<tr>
<td><img src="results/esnli_results/plots/Contradiction_Llama3.png" width="100%"></td>
<td><img src="results/esnli_results/plots/Contradiction_Qwen.png" width="100%"></td>
</tr>
</table>
<img src="results/movie_results/plots/legend.png">



## 4. Evidence-Supported Modification Precision

To measure how well counterfactual edits align with evidence spans, we compute word-level precision:

- **True Positive (TP)** — a word that was **modified** *and* lies **inside** a  evidence span.  
- **False Positive (FP)** — a word that was **modified** but lies **outside** any evidence span.  

Precision = TP / (TP + FP): What percent of the perturbations in the input are evidence supported?


We consider two types of evidence spans.
 - Human-annotation provided by ERASER (H-ESMP)
 - Rationales generated by trained CREST Explainer (R-ESMP)


<h2>ESMP on Movie Reviews</h2>

<table>
<tr>
<th>LLaMA-3</th>
<th>Qwen</th>
</tr>

<tr>
<td><img src="results/movie_results/plots/H-ESMP_Llama3.png" width="100%"></td>
<td><img src="results/movie_results/plots/H-ESMP_Qwen.png" width="100%"></td>
</tr>

<tr>
<td><img src="results/movie_results/plots/R-ESMP_Llama3.png" width="100%"></td>
<td><img src="results/movie_results/plots/R-ESMP_Qwen.png" width="100%"></td>
</tr>
</table>

<img src="results/movie_results/plots/legend.png">


<h2>ESMP on e-SNLI</h2>
<table>
<tr>
<th>LLaMA-3</th>
<th>Qwen</th>
</tr>

<tr>
<td><img src="results/esnli_results/plots/H-ESMP_Llama3.png" width="100%"></td>
<td><img src="results/esnli_results/plots/H-ESMP_Qwen.png" width="100%"></td>
</tr>

<tr>
<td><img src="results/esnli_results/plots/R-ESMP_Llama3.png" width="100%"></td>
<td><img src="results/esnli_results/plots/R-ESMP_Qwen.png" width="100%"></td>
</tr>
</table>

<img src="results/movie_results/plots/legend.png">


---

## Key Findings

**Capacity threshold.** Faithful counterfactual reasoning emerges only beyond a certain model scale. Smaller models struggle to produce consistent and meaningful counterfactual explanations.

**Limits of prompting.** Prompting alone cannot create reasoning abilities; it primarily steers capabilities that already exist within the model.

**Interaction between model and prompt.** The quality of counterfactual explanations reflects an interaction between the model’s internal capabilities and the external guidance provided by prompts.

## Implications for Explainable AI

Counterfactual self-explanations generated by LLMs can provide useful explanatory signals, but their reliability should not be assumed a priori.

Importantly, even invalid counterfactual explanations can still be informative. Implausible edits or inconsistencies may indicate unstable reasoning or limited model capacity, providing additional signals about the reliability of a model’s prediction.

Overall, counterfactual explanations should be interpreted as **interaction-dependent artifacts** that emerge from the combination of model behavior and prompting, rather than as transparent windows into a stable internal reasoning process.

---

<h3>Directory Structure</h3>

<ul>
  <li>
    <code>introspections/</code> — Self-generated counterfactual outputs and introspective explanations produced by the evaluated LLMs.
  </li>

  <li>
    <code>results/</code> — Experiment outputs including evaluation metrics, logs, and visualization plots.
  </li>

  <li>
    <code>eraserbenchmark-master/</code> — ERASER benchmark datasets containing human-annotated evidence spans used for evaluating explanation faithfulness.
  </li>

  <li>
    <code>llm-introspection-main/</code> — Adapted implementation of the framework from 
    <em>Are Self-Explanations from Large Language Models Faithful?</em> (Madsen et al.). 
    The original codebase was extended to support our experimental setup, including:
    <ul>
      <li>Support for newer LLM architectures (<strong>LLaMA-3</strong> and <strong>Qwen-2.5</strong>)</li>
      <li>Dataset wrapper classes for integration with our evaluation pipeline</li>
      <li>Compatibility with multiple inference providers (<strong>Together AI</strong>, <strong>Featherless</strong>)</li>
      <li>A configurable <strong>chat-history mechanism</strong> for controlled context experiments</li>
      <li>Adjusted prompt designs for sentiment and entailment tasks</li>
    </ul>
  </li>

  <li>
    <code>crest/</code> — Implementation of the <strong>CREST rationalization framework</strong>, used to train rationalizers that produce token-level explanations for comparison with LLM-generated introspections.
  </li>
</ul>


## References

This project builds upon and extends prior works:

1. Lanham, T., Chen, A., Radhakrishnan, A., Steiner, B., Denison, C., & Hernandez, D. (2023).  
   <em>Are Self-Explanations from Large Language Models Faithful?</em>  
   arXiv preprint arXiv:2301.03625.  
   https://arxiv.org/abs/2301.03625

2. DeYoung, J., Jain, S., Rajani, N. F., Lehman, E., Xiong, C., Socher, R., & Wallace, B. C. (2020).  
   <em>ERASER: A Benchmark to Evaluate Rationalized NLP Models.</em>  
   Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020).  
   https://arxiv.org/abs/1911.03429

3. Treviso, M., Ross, A., Guerreiro, N. M., & Martins, A. F. T. (2023).  
   <em>CREST: A Joint Framework for Rationalization and Counterfactual Text Generation.</em>  
   arXiv preprint arXiv:2305.00641.  
   https://arxiv.org/abs/2305.00641
