"""Evaluation Module.

Computes empirical benchmark metrics (Retrieval Hit Rate @ k, MRR, Extraction Precision/Recall/F1) over a labeled test set and outputs eval/results.md.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import logging

from ingestion.chunker import TextChunker
from embeddings.vector_store import ChromaVectorStore
from rag.pipeline import RAGPipeline
from extraction.pipeline import ExtractionPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_evaluation(
    eval_set_path: str = "eval/eval_set.json",
    results_output_path: str = "eval/results.md"
) -> Dict[str, Any]:
    """Runs end-to-end evaluation pipeline and writes results.md."""

    eval_file = Path(eval_set_path)
    if not eval_file.exists():
        raise FileNotFoundError(f"Evaluation set not found at {eval_set_path}")

    with open(eval_file, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    # Setup isolated test vector store
    store = ChromaVectorStore(persist_dir=".chroma_db_eval", collection_name="eval_collection")
    chunker = TextChunker(chunk_size=300, chunk_overlap=30)
    rag = RAGPipeline(vector_store=store)
    extractor = ExtractionPipeline()

    # Index test documents
    total_indexed_chunks = 0
    for doc in eval_data:
        parsed_pages = [{"page": 1, "text": doc["text"], "source": doc["source"]}]
        chunks = chunker.chunk_document(parsed_pages, doc_id=doc["doc_id"])
        store.add_chunks(chunks)
        total_indexed_chunks += len(chunks)

    logger.info(f"Indexed {total_indexed_chunks} test chunks across {len(eval_data)} documents.")

    # 1. Retrieval Evaluation
    total_queries = 0
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks = []

    for doc in eval_data:
        doc_id = doc["doc_id"]
        for qa in doc["qa_pairs"]:
            total_queries += 1
            query = qa["question"]
            expected_kws = [kw.lower() for kw in qa["expected_answer_keywords"]]

            # Retrieve top 5
            retrieved = store.search(query=query, top_k=5, doc_id=doc_id)

            rank = 0
            for i, chunk in enumerate(retrieved, start=1):
                chunk_text_lower = chunk["text"].lower()
                if any(kw in chunk_text_lower for kw in expected_kws):
                    rank = i
                    break

            if rank > 0:
                if rank <= 1:
                    hits_at_1 += 1
                if rank <= 3:
                    hits_at_3 += 1
                if rank <= 5:
                    hits_at_5 += 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

    hit_rate_1 = (hits_at_1 / total_queries) if total_queries else 0.0
    hit_rate_3 = (hits_at_3 / total_queries) if total_queries else 0.0
    hit_rate_5 = (hits_at_5 / total_queries) if total_queries else 0.0
    mrr = (sum(reciprocal_ranks) / len(reciprocal_ranks)) if reciprocal_ranks else 0.0

    # 2. Structured Extraction & Classification Evaluation
    total_docs = len(eval_data)
    correct_classifications = 0
    total_pred_entities = 0
    total_true_entities = 0
    true_positives = 0

    for doc in eval_data:
        res = extractor.extract_from_text(doc["text"], doc_name=doc["source"])
        pred_type = res["document_type"]

        if pred_type.lower() == doc["doc_type"].lower():
            correct_classifications += 1

        pred_ents = res["entities"]
        all_pred_set = set()
        for key in ["organizations", "persons", "dates", "amounts"]:
            if isinstance(pred_ents.get(key), list):
                for item in pred_ents[key]:
                    val = item["text"] if isinstance(item, dict) else item
                    all_pred_set.add(val.lower())

        expected_ents = doc.get("expected_entities", {})
        all_true_set = set()
        for key in ["organizations", "persons", "dates", "amounts"]:
            for item in expected_ents.get(key, []):
                all_true_set.add(item.lower())

        total_pred_entities += len(all_pred_set)
        total_true_entities += len(all_true_set)
        true_positives += len(all_pred_set.intersection(all_true_set))

    cls_accuracy = (correct_classifications / total_docs) if total_docs else 0.0
    precision = (true_positives / total_pred_entities) if total_pred_entities else 0.0
    recall = (true_positives / total_true_entities) if total_true_entities else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    # Summary Metrics Object
    metrics_summary = {
        "retrieval": {
            "total_queries": total_queries,
            "hit_rate_1": round(hit_rate_1, 4),
            "hit_rate_3": round(hit_rate_3, 4),
            "hit_rate_5": round(hit_rate_5, 5),
            "mrr": round(mrr, 4)
        },
        "extraction": {
            "total_test_documents": total_docs,
            "classification_accuracy": round(cls_accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4)
        }
    }

    # Write Markdown Results Report
    markdown_content = f"""# Empirical Benchmark Evaluation Results

Automated evaluation generated over **{total_docs} ground-truth labeled documents** and **{total_queries} question-answer evaluation pairs**.

---

## 1. Retrieval Layer Metrics

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Hit Rate @ 1** | `{hit_rate_1:.2%}` | Percentage of queries where top-1 retrieved chunk contains answer keywords. |
| **Hit Rate @ 3** | `{hit_rate_3:.2%}` | Percentage of queries where top-3 retrieved chunks contain answer keywords. |
| **Hit Rate @ 5** | `{hit_rate_5:.2%}` | Percentage of queries where top-5 retrieved chunks contain answer keywords. |
| **Mean Reciprocal Rank (MRR)** | `{mrr:.4f}` | Reciprocal rank average of first relevant chunk. |

---

## 2. Structured Extraction & Classification Metrics

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Doc Classification Accuracy** | `{cls_accuracy:.2%}` | Accuracy predicting Contract, Resume, Financial Report, Invoice, etc. |
| **Entity Extraction Precision** | `{precision:.2%}` | Ratio of correctly identified entities vs total predicted. |
| **Entity Extraction Recall** | `{recall:.2%}` | Ratio of correctly identified entities vs total true entities. |
| **Entity Extraction F1-Score** | `{f1_score:.4f}` | Harmonic mean of entity precision and recall. |

---

## 3. Evaluation Environment Details
- **Vector Engine**: Chroma Persistent Local DB (`cosine` metric space)
- **Embedding Transformer**: `all-MiniLM-L6-v2` (384-dimensional dense vectors)
- **Chunking Strategy**: 300 character sliding window, 30 character overlap
"""

    out_file = Path(results_output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    logger.info(f"Evaluation report written to {results_output_path}")
    return metrics_summary


if __name__ == "__main__":
    results = run_evaluation()
    print("Evaluation Results Summary:\n", json.dumps(results, indent=2))


# Outputs eval/results.md with markdown tables for easy portfolio embedding