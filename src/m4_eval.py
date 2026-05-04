"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, re
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    per_question = []
    import math

    def tok_set(s: str):
        return {w.lower() for w in re.findall(r"\w+", s)}

    faith_list = []
    ansrel_list = []
    cprec_list = []
    crec_list = []

    for q, a, ctxs, gt in zip(questions, answers, contexts, ground_truths):
        a_toks = tok_set(a)
        gt_toks = tok_set(gt)
        # answer relevancy: overlap between answer and ground truth
        ansrel = (len(a_toks & gt_toks) / max(len(gt_toks), 1)) if gt_toks else 0.0

        # context precision: proportion of provided contexts that contain any gt token
        ctx_hits = 0
        covered = set()
        for c in ctxs:
            c_toks = tok_set(c)
            if gt_toks & c_toks:
                ctx_hits += 1
                covered |= (gt_toks & c_toks)
        cprec = ctx_hits / max(len(ctxs), 1)
        crec = (len(covered) / max(len(gt_toks), 1)) if gt_toks else 0.0

        # faithfulness: fraction of answer tokens that are supported by contexts
        support = 0
        total = max(len(a_toks), 1)
        for token in a_toks:
            for c in ctxs:
                if token in c.lower():
                    support += 1
                    break
        faith = support / total

        per_question.append(EvalResult(question=q, answer=a, contexts=ctxs, ground_truth=gt,
                                       faithfulness=faith, answer_relevancy=ansrel,
                                       context_precision=cprec, context_recall=crec))
        faith_list.append(faith)
        ansrel_list.append(ansrel)
        cprec_list.append(cprec)
        crec_list.append(crec)

    def mean_or_zero(lst):
        return float(sum(lst) / len(lst)) if lst else 0.0

    return {
        "faithfulness": mean_or_zero(faith_list),
        "answer_relevancy": mean_or_zero(ansrel_list),
        "context_precision": mean_or_zero(cprec_list),
        "context_recall": mean_or_zero(crec_list),
        "per_question": per_question,
    }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    scored = []
    for r in eval_results:
        avg = (r.faithfulness + r.answer_relevancy + r.context_precision + r.context_recall) / 4.0
        scored.append((avg, r))
    scored.sort(key=lambda x: x[0])
    out = []
    for avg, r in scored[:bottom_n]:
        metrics = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }
        worst_metric = min(metrics, key=lambda k: metrics[k])
        score = metrics[worst_metric]
        if worst_metric == "faithfulness":
            diagnosis = "LLM hallucinating"
            fix = "Tighten prompt, lower temperature, or increase grounding"
        elif worst_metric == "context_recall":
            diagnosis = "Missing relevant chunks"
            fix = "Improve chunking or expand retrieval (BM25/dense)"
        elif worst_metric == "context_precision":
            diagnosis = "Too many irrelevant chunks"
            fix = "Add reranking or metadata filters"
        else:
            diagnosis = "Answer mismatch"
            fix = "Improve prompt template or provide clearer context"
        out.append({"question": r.question, "worst_metric": worst_metric, "score": float(score), "diagnosis": diagnosis, "suggested_fix": fix})
    return out


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
