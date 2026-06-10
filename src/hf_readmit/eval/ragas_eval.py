"""RAGAS retrieval evaluation for the guideline RAG pipeline.

Builds a 10-item question / answer / contexts dataset (contexts come from the
live :class:`HybridRetriever`) and scores it with RAGAS:
context_precision, context_recall, faithfulness, answer_relevancy.

COST NOTE: ``run_ragas_eval`` makes real OpenAI calls (faithfulness and
answer_relevancy are LLM-judged; embeddings are used for retrieval/relevancy).
It is mocked in tests and is only invoked by the full eval suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# (question, reference answer) pairs grounded in the guideline corpus content.
QA_PAIRS: list[tuple[str, str]] = [
    (
        "What is the Class I recommendation for follow-up timing after HF discharge?",
        "An early follow-up visit within 7-14 days (and telephone follow-up within ~3 days) "
        "after discharge is recommended to reduce readmissions.",
    ),
    (
        "Which beta-blockers have mortality benefit in HFrEF?",
        "Carvedilol, metoprolol succinate (extended-release), and bisoprolol are the "
        "evidence-based beta-blockers with mortality benefit in HFrEF.",
    ),
    (
        "What is the recommended starting dose of carvedilol for HFrEF?",
        "Carvedilol is typically started at 3.125 mg twice daily and uptitrated as tolerated.",
    ),
    (
        "When should an ICD be considered in HF patients?",
        "An ICD is recommended for primary prevention in HFrEF with LVEF <=35% despite >=3 months "
        "of guideline-directed medical therapy, with expected survival >1 year.",
    ),
    (
        "What sodium restriction is recommended for HF patients?",
        "Dietary sodium restriction (commonly <2-3 g/day) is advised for patients with symptomatic HF "
        "to reduce congestive symptoms.",
    ),
    (
        "What are the BOOST toolkit's key components for care transitions?",
        "BOOST emphasizes risk assessment, medication reconciliation, patient/caregiver education with "
        "teach-back, a clear discharge plan, and timely follow-up communication.",
    ),
    (
        "How should diuretics be adjusted at discharge?",
        "Diuretic doses should be adjusted to achieve and maintain euvolemia, with a discharge weight "
        "target and instructions for self-adjustment and monitoring.",
    ),
    (
        "What patient education topics are essential before HF discharge?",
        "Education should cover daily weights, sodium and fluid guidance, medication adherence, symptom "
        "recognition, and when to seek care.",
    ),
    (
        "What are the signs of worsening HF patients should monitor?",
        "Patients should watch for rapid weight gain, increasing dyspnea or orthopnea, worsening edema, "
        "and reduced exercise tolerance.",
    ),
    (
        "What is the recommended ACE inhibitor approach for HFrEF?",
        "ACE inhibitors (or ARBs if intolerant, with ARNI preferred where appropriate) are recommended "
        "for HFrEF and uptitrated to target doses as tolerated.",
    ),
]


def _retrieve_contexts(top_k: int = 5) -> list[list[str]]:
    """Retrieve guideline contexts for each question via the HybridRetriever."""
    from hf_readmit.agent import tools

    retriever = tools._get_retriever()
    contexts: list[list[str]] = []
    for question, _ in QA_PAIRS:
        hits = retriever.retrieve(question, top_k=top_k)
        contexts.append([h.get("document") or h.get("text") or "" for h in hits])
    return contexts


def _evaluate_ragas(questions: list[str], answers: list[str], contexts: list[list[str]]) -> dict[str, float]:
    """Run RAGAS over the assembled samples and return canonical metric scores.

    Isolated so tests can mock the (expensive, network-bound) RAGAS call.
    """
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    samples = [
        SingleTurnSample(
            user_input=q,
            response=a,
            retrieved_contexts=ctx or [""],
            reference=a,
        )
        for q, a, ctx in zip(questions, answers, contexts)
    ]
    dataset = EvaluationDataset(samples=samples)

    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-large"))

    # Map our canonical keys to RAGAS metric instances.
    metric_map = {
        "context_precision": LLMContextPrecisionWithReference(),
        "context_recall": LLMContextRecall(),
        "faithfulness": Faithfulness(),
        "answer_relevancy": ResponseRelevancy(),
    }
    result = evaluate(
        dataset=dataset,
        metrics=list(metric_map.values()),
        llm=llm,
        embeddings=embeddings,
    )

    df = result.to_pandas()
    scores: dict[str, float] = {}
    for key, metric in metric_map.items():
        col = metric.name
        scores[key] = float(df[col].mean()) if col in df.columns else None
    return scores


def run_ragas_eval(output_path: Path) -> dict[str, Any]:
    """Run the RAGAS retrieval eval and write results to ``output_path``.

    Args:
        output_path: Where to write the metrics JSON.

    Returns:
        Dict with keys ``context_precision``, ``context_recall``,
        ``faithfulness``, ``answer_relevancy`` plus metadata.
    """
    questions = [q for q, _ in QA_PAIRS]
    answers = [a for _, a in QA_PAIRS]
    contexts = _retrieve_contexts()

    scores = _evaluate_ragas(questions, answers, contexts)

    output = {"n_questions": len(QA_PAIRS), **scores}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    return output
