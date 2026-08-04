# Empirical Benchmark Evaluation Results

Automated evaluation generated over **3 ground-truth labeled documents** and **8 question-answer evaluation pairs**.

---

## 1. Retrieval Layer Metrics

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Hit Rate @ 1** | `100.00%` | Percentage of queries where top-1 retrieved chunk contains answer keywords. |
| **Hit Rate @ 3** | `100.00%` | Percentage of queries where top-3 retrieved chunks contain answer keywords. |
| **Hit Rate @ 5** | `100.00%` | Percentage of queries where top-5 retrieved chunks contain answer keywords. |
| **Mean Reciprocal Rank (MRR)** | `1.0000` | Reciprocal rank average of first relevant chunk. |

---

## 2. Structured Extraction & Classification Metrics

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Doc Classification Accuracy** | `100.00%` | Accuracy predicting Contract, Resume, Financial Report, Invoice, etc. |
| **Entity Extraction Precision** | `57.14%` | Ratio of correctly identified entities vs total predicted. |
| **Entity Extraction Recall** | `25.00%` | Ratio of correctly identified entities vs total true entities. |
| **Entity Extraction F1-Score** | `0.3478` | Harmonic mean of entity precision and recall. |

---

## 3. Evaluation Environment Details
- **Vector Engine**: Chroma Persistent Local DB (`cosine` metric space)
- **Embedding Transformer**: `all-MiniLM-L6-v2` (384-dimensional dense vectors)
- **Chunking Strategy**: 300 character sliding window, 30 character overlap


<!-- Last computed: automated benchmark run via eval/evaluate.py -->