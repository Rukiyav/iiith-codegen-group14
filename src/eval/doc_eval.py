"""Documentation generation evaluation: ROUGE-L primary, BLEU/BERTScore secondary."""

from __future__ import annotations

from typing import Dict, List, Optional

from rouge_score import rouge_scorer

from src.generation.doc_postprocess import fallback_documentation


def compute_rouge_l(predictions: List[str], references: List[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        if not pred.strip() or not ref.strip():
            scores.append(0.0)
            continue
        result = scorer.score(ref, pred)
        scores.append(result["rougeL"].fmeasure)
    return sum(scores) / len(scores) if scores else 0.0


def compute_bleu(predictions: List[str], references: List[str]) -> float:
    try:
        import sacrebleu
    except ImportError:
        return 0.0
    if not predictions:
        return 0.0
    return sacrebleu.corpus_bleu(predictions, [references]).score / 100.0


def compute_bertscore(predictions: List[str], references: List[str]) -> float:
    try:
        from bert_score import score as bert_score
    except ImportError:
        return 0.0
    if not predictions:
        return 0.0
    _, _, f1 = bert_score(predictions, references, lang="en", verbose=False)
    return float(f1.mean())


def evaluate_doc_batch(
    examples: List[Dict],
    generate_fn,
    max_samples: Optional[int] = None,
    include_bertscore: bool = False,
) -> Dict:
    subset = examples[:max_samples] if max_samples else examples
    predictions: List[str] = []
    references: List[str] = []
    details = []
    for ex in subset:
        code = ex.get("code", "")
        reference = ex.get("documentation", "")
        try:
            pred = generate_fn(code)
        except Exception:
            pred = fallback_documentation(code)
        predictions.append(pred)
        references.append(reference)
        details.append(
            {
                "id": ex.get("id"),
                "code": code,
                "reference": reference,
                "prediction": pred,
            }
        )

    metrics = {
        "rouge_l": compute_rouge_l(predictions, references),
        "bleu": compute_bleu(predictions, references),
    }
    if include_bertscore:
        metrics["bertscore_f1"] = compute_bertscore(predictions, references)

    return {
        "n_samples": len(subset),
        "metrics": metrics,
        "details": details,
    }
