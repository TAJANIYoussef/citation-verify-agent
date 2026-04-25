Run the benchmark on the labeled evaluation set and print metrics.

Steps:
1. Verify `data/eval/ground_truth.jsonl` exists and has at least 10 entries:
   `wc -l data/eval/ground_truth.jsonl`

2. Run both loop modes and compare:
   `citverif-eval --ground-truth data/eval/ground_truth.jsonl --loops react,react+reflexion --out data/eval/results`

   Or for a quick smoke-test on the first 10 entries:
   `citverif-eval --limit 10`

3. After completion, print the side-by-side macro-F1 comparison:
   `cat data/eval/results/metrics_react.json data/eval/results/metrics_react+reflexion.json | python -c "
import json, sys
data = [json.loads(l) for l in sys.stdin if l.strip()]
for d in data:
    print(f\"{d['mode']:20} macro-F1={d['macro_f1']:.3f}  halluc={d['hallucination_rate']:.1%}  latency={d['mean_latency_s']:.1f}s\")
"`

4. Flag any verdict class with F1 < 0.5 as a priority regression.
5. Flag hallucination_rate > 0.1 as critical.
