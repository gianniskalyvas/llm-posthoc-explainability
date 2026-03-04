# How Model Scale Influences Post-Hoc Explainability in Large Language Models

As large language models (LLMs) grow in size to enhance performance, there remains a limited understanding of how this expansion affects their explainability.
This project adopts counterfactuals as a primary method for post-hoc explainability, using the following definition: 
"A counterfactual explanation is a minimal edit of the input with the words or phrases crucial for classification changed, revealing what should have been different to observe the opposite outcome."

We evaluate multiple models across different scales by generating counterfactual explanations as in *Are Self-Explanations from Large Language Models Faithful?*.
We extended the framework’s abstract classes to support newer models and our custom datasets, and to better align the framework with the specific requirements of our study.

We analyze the impact of model scale but also prompt configuration on the quality of post-hoc counterfactual explanations.

### Datasets

ERASER benchmark: Standard benchmark datasets augmented with human-provided rationale annotations.

 - Movie Reviews: A collection of movie reviews labeled for sentiment (positive or negative). 
 - e-SNLI: Sentence pairs labeled for entailment (yes or no).

### LLMs 

- **LLaMA-3:** 1B, 3B, 8B, 70B  
- **Qwen-2.5:** 1.5B, 3B, 7B, 14B, 32B, 72B  



# Experiment Pipeline


<div style="text-align:center;">
  <img src="experimental-pipeline.png" alt="pipeline" style="width:100%;"/>
</div>

## 1. Classifiers

<div style="text-align:center;">
  <img src="results/classifiers/classifier_accuracy_comparison.png" alt="Experiment Summary" style="width:100%;"/>
</div>


## 2. Counterfactual Generation

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



## 3. Evaluation

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
  <img src="results/movie_results/plots/legend.png">
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