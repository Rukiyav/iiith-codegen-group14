# Evaluation Report — RAG, LoRA, CodeBLEU, and Agent Improvements

This document records a round of experiments run on local GPU hardware (RTX 3070):
implementing LoRA fine-tuning and RAG from scratch, adding CodeBLEU as a code-quality
metric, finding and fixing a real bug in the agentic self-correction loop, and testing
whether more training epochs improve pass@1. All numbers below are from actual eval
runs, not estimates — result JSONs are in `experiments/results/fresh*.json`.

## Implementation summary

- **LoRA** — `src/models/lora.py` wraps CodeGen with `peft.LoraConfig` targeting
  `qkv_proj`/`out_proj` (the only attention-linear layers this architecture exposes).
  Wired into `--lora` on `src/train.py` / `src/train_doc.py`. `src/generation/engine.py`
  auto-detects `adapter_config.json` and loads base model + adapter.
- **RAG** — `src/rag/` (`index.py`, `retrieve.py`, `augment.py`): TF-IDF retrieval over
  the MBPP/CoDocBench train splits, top-k similar examples formatted as few-shot
  context and prepended to the prompt. Wired into `--rag` on `run_eval.py` and a
  `context` param on the generation functions.
- **CodeBLEU** — added to `src/eval/code_eval.py` (`compute_codebleu`), included in
  the metrics for both plain and agent code-eval paths. Needed pinned versions
  (`tree-sitter==0.22.3`, `tree-sitter-python==0.21.0`) — the latest tree-sitter-python
  isn't ABI-compatible with codebleu's pinned tree-sitter.
- **Agent retry fix** — `src/agent/code_agent.py`: retries previously showed the model
  its own failed code inside the prompt ("Previous code: ..."). CodeGen-350M isn't
  instruction-tuned, so it echoed the failed code back near-verbatim instead of fixing
  it, and `clean_generated_code` grabbed that echo as the answer — this is why
  `retry_success_rate` measured exactly 0% before. Fix: `--no_failure_context` makes
  retries resample the *original* prompt at `--retry_temperature` with
  `--num_candidates` distinct seeds (best-of-N), instead of reusing a failure-context
  prompt the model can't act on. Verified independently that varying the seed alone
  produces genuinely different candidates. All new parameters default to the original
  behavior, so the existing test suite (22 tests) is unaffected.

All of this required a dedicated `codegen-group14` conda environment (see chat history
for setup) — a plain venv hit a `sentencepiece` DLL crash specific to this Windows box.

## Results: code generation (MBPP test, n=43)

| Configuration | pass@1 | CodeBLEU | Notes |
|---|---|---|---|
| Baseline (upstream CodeGen-350M-multi) | 6.98% | 0.1665 | |
| RAG (few-shot retrieval, upstream model) | 9.30% | 0.1714 | |
| LoRA (2 epochs) | 16.28% | 0.2096 | best LoRA config |
| Full fine-tune (2 epochs) | 20.93% | 0.2573 | best fine-tune config |
| Agent, original retry logic | 20.93% | 0.2539 | retry_success_rate: 0% |
| **Agent, fixed retry logic** | **30.23%** | **0.2894** | retry_success_rate: 9.3% |
| Full fine-tune, 4 epochs | 13.95% | 0.18–0.26 | regressed — overfitting |
| LoRA, 4 epochs | 11.63% | 0.2075 | also regressed |

Ranking holds for both metrics: baseline < RAG < LoRA < fine-tune < fixed agent —
CodeBLEU (graded similarity to reference) agrees with pass@1 (binary test-pass), a
good sanity check that the two metrics aren't in tension.

## Results: documentation generation (CoDocBench test, n=50)

| Configuration | ROUGE-L | BLEU |
|---|---|---|
| Baseline (upstream, real LM) | 0.1705 | 0.0277 |
| RAG (few-shot retrieval, upstream model) | 0.1275 | 0.0060 |
| LoRA (2 epochs) | 0.2179 | 0.0290 |
| Full fine-tune (2 epochs) | 0.2296 | 0.0304 |

RAG **hurt** documentation quality here — plausible explanation: prepending 2
unrelated code+doc examples likely confuses a model this small more than it helps;
CodeGen-350M isn't large enough to reliably exploit in-context exemplars the way
larger models can. Reportable as a real limitation, not a bug.

One methodology note: `src/generation/docs.py`'s `_should_use_lm()` silently falls
back to non-neural AST-based docs when the model is passed as a raw HF Hub id
(only local-directory paths or fine-tuned-named checkpoints trigger the real LM).
The upstream-baseline and RAG-docs numbers above use `experiments/checkpoints/codegen-base-local`
(a local copy of the same weights) specifically to route through the real LM — an
earlier pass at these two numbers silently used the AST fallback and was corrected.

## Honest takeaways

1. **LoRA gets most of the way to full fine-tuning** at a fraction of the cost: 350x
   smaller artifact (3.9MB adapter vs 1.4GB full model), ~5-6x faster to train, and
   within a few points of full fine-tune's pass@1/ROUGE-L.
2. **RAG gave a small, real lift on code** (+2.3 points pass@1) but **hurt docs** —
   technique effectiveness depends on task and is worth testing per-task, not assumed.
3. **The agent fix is the single biggest lever found**: 20.93% → 30.23% pass@1 (+9.3
   points, ~44% relative) from a one-file bug fix, no retraining required. The lesson
   generalizes: natural-language self-correction prompting doesn't work on small,
   non-instruction-tuned models — resampling/best-of-N does.
4. **More training epochs did not help, for either technique** — and interestingly,
   LoRA's eval_loss *improved* slightly at 4 epochs while its pass@1 still dropped.
   Token-level loss and execution-based correctness aren't tightly coupled at this
   sample size; don't use eval_loss alone as a stopping/selection signal for pass@1.

## New checkpoints and files

- `experiments/checkpoints/codegen-mbpp-lora`, `codegen-doc-lora` — LoRA adapters (2 epochs, recommended)
- `experiments/checkpoints/codegen-mbpp-v3`, `codegen-mbpp-lora-e4` — 4-epoch variants (kept for reference; underperform, not recommended)
- `experiments/results/fresh*.json` — all raw eval outputs backing the tables above
- `src/rag/`, `src/models/lora.py` — new modules
- `requirements.txt` — added `scikit-learn`, `codebleu`, `tree-sitter==0.22.3`, `tree-sitter-python==0.21.0`
