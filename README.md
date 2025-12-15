# Re-ranking-Agent-via-RL

This repository contains an implementation that builds the re-ranking reasoning agents via RL algorithms for modern information retrieval and LLM-based ranking Agents. It is comprised of instructions, tools, grading logic, and pass-rate expectations.

---

# **README — RE-RANK Listwise NDCG RL Task for LLM Training**

## Overview

This repository contains a custom **Reinforcement Learning (RL) task for LLM training** assessment**. 

The goal of the task is to teach an LLM a practical, research-relevant ML skill for Reasoning Re-ranking LLM Agent:

### ** Performing listwise document reranking using relevance labels and computing NDCG@5**

This skill is fundamental in information retrieval (IR) systems, recommender models, and modern LLM reranking reasoning pipelines: (https://arxiv.org/pdf/2505.20046).

The task is fully self-contained and runs using the agent loop provided in the `hello-py` the provided starter repository.

---

## Task Description

The LLM is given:

* A **query**
* Five passages
* A **baseline BM25 ranking**
* Relevance labels (`0–3`, where `3` is highly relevant)

The model will:

1. **Re-rank the five passages** by descending relevance
2. **Compute DCG@5** using
   [
   DCG = \sum_{i=1}^{5} \frac{rel_i}{\log_2(i+1)}
   ]
3. **Compute the ideal DCG (IDCG)**
4. **Compute NDCG@5 = DCG / IDCG**
5. **Submit the final ranking + NDCG** using the required JSON format:

```json
{
  "ranking": "[1] > [2] > [5] > [4] > [3]",
  "ndcg": 1.0
}


##  Tools Provided to the Model

### ** python**

The logic implemented are for:

* Sorting
* Computing DCG
* Logging intermediate values
* Validating NDCG calculations


## Grading Logic

The grading logic (in `tasks/rearank_grader.py`) checks:

### Correct ranking

Must be exactly the five passage IDs sorted by relevance:
`[1] > [2] > [5] > [4] > [3]`
(IDs 2 and 5 tie, so ordering follows original ID order.)

### NDCG within tolerance

The student's NDCG must be within **±0.02** of the true value.

### Parsing & formatting

Ranking must follow the string format:
`"[i] > [j] > [k] > ..."`

### Valid set

Model must return exactly the IDs `{1,2,3,4,5}`.

This ensures the model is being evaluated on **understanding**, not memorization or string matching.

---


Key takeaways from this task:

* **Real ML engineer skills**
  Ranking systems, evaluation metrics, IR techniques
* **Multiple reasoning steps**
  Sorting, computing logs, understanding DCG
* **Multiple solution strategies**
  Python tool, reasoning, manual math
* **Non-trivial**
  Models frequently fail due to ranking logic, formatting, or math
* **Pass rate 10–40%**
  Perfect target for RL fine-tuning

---

## Repository Structure

```
hello-py/
 ├── main.py                     # Agent loop using Anthropic API
 ├── tasks/
 │     ├── rearank_task.py       # Prompt provided to the LLM
 │     ├── rearank_grader.py     # Automatic grading logic
 ├── README.md                   # This file
 ├── pyproject.toml
 ├── ...
```

---

## How to Run

### **1. Install dependencies**

```
pip install anthropic
```

Optional (with PyTorch):

```
pip install torch
```

---

### **2. Setting API key**

PowerShell:

```powershell
$env:ANTHROPIC_API_KEY="api_key"
```

or inside the script:

```python
import os
os.environ["ANTHROPIC_API_KEY"] = "api_key"
```

---

### **Run the task**

```
python main.py
```


---

## Pass Rate Expectation

The typical pass rate is **20–30%**.

---

## Inspiration

This task is inspired by the **RE-RANK** research paper:

> *RERANK: Reasoning-Enhanced LLM Reranking via Reinforcement Learning*
> https://arxiv.org/pdf/2505.20046






