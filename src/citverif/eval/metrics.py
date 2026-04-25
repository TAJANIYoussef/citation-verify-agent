from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from citverif.schema import Verdict

ALL_VERDICTS: list[Verdict] = [
    "supported", "partially_supported", "unsupported", "misleading", "unverifiable"
]


@dataclass
class VerdictMetrics:
    precision: float
    recall: float
    f1: float
    support: int   # number of ground-truth instances of this class


@dataclass
class EvalMetrics:
    per_class: dict[str, VerdictMetrics] = field(default_factory=dict)
    macro_f1: float = 0.0
    # % of cases where expected != "supported" but model returned "supported"
    hallucination_rate: float = 0.0
    # % of entries where a paper was resolved (pdf or abstract)
    resolution_rate: float = 0.0
    mean_latency_s: float = 0.0
    total: int = 0
    correct: int = 0
    accuracy: float = 0.0


def compute_metrics(
    expected: list[Verdict],
    predicted: list[Verdict],
    latencies_s: list[float] | None = None,
    resolved_flags: list[bool] | None = None,
) -> EvalMetrics:
    """
    Compute macro-F1 (averaged over all 5 verdict classes, including zero-support classes),
    hallucination rate, resolution rate, and mean latency.
    """
    assert len(expected) == len(predicted), "expected and predicted must be same length"
    n = len(expected)
    if n == 0:
        return EvalMetrics()

    # Per-class counts
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    support: dict[str, int] = defaultdict(int)

    for exp, pred in zip(expected, predicted):
        support[exp] += 1
        if exp == pred:
            tp[pred] += 1
        else:
            fp[pred] += 1
            fn[exp] += 1

    per_class: dict[str, VerdictMetrics] = {}
    f1_scores: list[float] = []

    for v in ALL_VERDICTS:
        p = tp[v] / (tp[v] + fp[v]) if (tp[v] + fp[v]) > 0 else 0.0
        r = tp[v] / (tp[v] + fn[v]) if (tp[v] + fn[v]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class[v] = VerdictMetrics(precision=p, recall=r, f1=f1, support=support[v])
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores)

    # Hallucination rate: model says "supported" when expected is NOT "supported"
    hallucinations = sum(
        1 for exp, pred in zip(expected, predicted)
        if pred == "supported" and exp != "supported"
    )
    hallucination_rate = hallucinations / n

    # Resolution rate
    if resolved_flags is not None:
        resolution_rate = sum(resolved_flags) / len(resolved_flags)
    else:
        resolution_rate = 0.0

    # Latency
    mean_latency = sum(latencies_s) / len(latencies_s) if latencies_s else 0.0

    correct = sum(1 for e, p in zip(expected, predicted) if e == p)

    return EvalMetrics(
        per_class=per_class,
        macro_f1=macro_f1,
        hallucination_rate=hallucination_rate,
        resolution_rate=resolution_rate,
        mean_latency_s=mean_latency,
        total=n,
        correct=correct,
        accuracy=correct / n,
    )


def format_metrics_table(metrics: EvalMetrics, mode: str = "") -> str:
    """Return a plain-text table suitable for logging."""
    header = f"{'Verdict':<22} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>5}"
    rows = [f"\n=== Eval results {mode} ===", header, "-" * len(header)]
    for v in ALL_VERDICTS:
        m = metrics.per_class.get(v)
        if m is None:
            continue
        rows.append(
            f"{v:<22} {m.precision:>6.3f} {m.recall:>6.3f} {m.f1:>6.3f} {m.support:>5}"
        )
    rows.append("-" * len(header))
    rows.append(f"{'Macro-F1':<22} {'':>6} {'':>6} {metrics.macro_f1:>6.3f} {metrics.total:>5}")
    rows.append(f"Accuracy:          {metrics.accuracy:.1%}  ({metrics.correct}/{metrics.total})")
    rows.append(f"Hallucination rate:{metrics.hallucination_rate:.1%}")
    rows.append(f"Resolution rate:   {metrics.resolution_rate:.1%}")
    rows.append(f"Mean latency:      {metrics.mean_latency_s:.1f}s / citation")
    return "\n".join(rows)
