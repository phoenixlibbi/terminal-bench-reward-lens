# reward-lens

**Same agent runs, two verifiers.** A toolkit for grading terminal-agent trajectories under two reward philosophies and showing what each one hides:

| | Terminal-Bench 2.0 | Long-Horizon-Terminal-Bench |
|---|---|---|
| Verifier output | Binary: 1 only if every test passes | Dense: weighted milestone score R ∈ [0, 1] |
| RL view | Sparse outcome reward | Fine-grained terminal reward with partial credit |
| What it misses | Near-misses look identical to doing nothing | Easy "hygiene" checks can hand out free credit |

The goal is to make reward design measurable: given the same final container states, how do pass rates, model rankings and reward-hacking resistance change when the verifier changes?

## Why

Terminal-Bench ([Merrill et al., 2026](https://arxiv.org/abs/2601.11868)) grades each task as solved / unsolved. Long-Horizon-Terminal-Bench ([Li et al., 2026](https://arxiv.org/abs/2607.08964)) argues that on long tasks this is too sparse, and decomposes each task into weighted subtasks so partial progress becomes visible.

Both verifiers still run **once, at the end of the episode**. The difference is the *resolution* of the reward, not its timing. That leaves an open question the LHTB paper does not measure: a denser reward is also easier to game. `reward-lens` measures both sides.

## Components

| Part | Module | What it does | Status |
|---|---|---|---|
| 1 | `reward_lens.report` | Re-grader. Reads Harbor trial logs (`verifier/ctrf.json`) and computes the binary reward, a naive dense reward and a rubric-weighted milestone reward. Reports pass@1 vs. mean R, near-misses, ranking changes and a reward histogram. | done |
| 2 | `reward_lens.convert` | Converts a Terminal-Bench-style Harbor task into a dense-reward task: adds a `rubric.toml` and a verifier that writes a fractional `reward.txt` plus a per-milestone breakdown. | planned |
| 3 | `reward_lens.floor` | Reward-hacking floor test. Runs adversarial "dummy" solutions (no-op, schema-only, input echo) through each verifier and measures how much free credit they collect. | planned |
| 4 | `reward_lens.harness` | Minimal Terminus-style agent loop (keystrokes → tmux in Docker → screen back to the model) that plugs into Harbor as a custom agent and reports both rewards live. Includes a zero-cost replay agent for recorded trajectories. | planned |

## Three rewards, one run

For every trial, reward-lens computes:

| Reward | Definition | Paper |
|---|---|---|
| Binary | 1 if no test fails, else 0 | Terminal-Bench 2.0 |
| Naive dense | Share of tests passed, all tests equal | The obvious shortcut |
| Milestone R | `R = Σ w_k r_k / Σ w_k` over rubric milestones; 0 if any integrity gate fails | Long-Horizon-Terminal-Bench |

Comparing the second and third rewards isolates the effect of **weighting**. Many verifiers spend most of their tests on cheap checks (files exist, schema, determinism), so an unweighted dense reward can hand a non-solution most of the credit.

## Usage

```bash
pip install -e .

# Draft a rubric from the tests seen in your trials (one milestone per test class), then edit the weights
reward-lens rubric-init path/to/jobs/my-task --name my-task > rubric.toml

# Re-grade every trial under all three rewards
reward-lens regrade path/to/jobs/my-task --rubric rubric.toml --out report.md --json report.json
```

`regrade` accepts any directory that contains Harbor trial folders (`result.json` + `verifier/ctrf.json`). The verifier must write a CTRF report, e.g. `pytest --ctrf /logs/verifier/ctrf.json`.

## Rubric format

See [`examples/rubric.example.toml`](examples/rubric.example.toml). In short:

```toml
[task]
name = "demo-pipeline-repair"
pass_threshold = 0.95

[[gate]]                       # integrity: if any of these fails, R = 0
name = "integrity"
tests = ["test_inputs_not_modified"]

[[milestone]]
name = "deliverables"
weight = 0.05
tests = ["TestOutputFiles::*", "TestSchema::*"]

[[milestone]]
name = "model_matches_reference"
weight = 0.70
tests = ["test_rmse", "test_coverage"]
scoring = "fraction"           # or "all"
```

Every test reported by the verifier must belong to exactly one milestone or gate (or be listed in `ignore`). Uncovered or ambiguous tests are an error, so no check silently drops out of the reward.

## What the report shows

- **Sanity checks:** the oracle run must score R = 1.0, and the recomputed binary reward must match the verifier's own `reward.txt`.
- **Leaderboard:** binary pass rate, naive dense, mean milestone R, pass @ R ≥ threshold, near-misses, no-progress runs, and how each model's rank moves from binary to R.
- **What each reward hides:** how much progress binary grading erases, and how much credit naive dense grants beyond the rubric.
- **Reward distribution:** histogram with the bucket edges of LHTB's Figure 4.
- **Milestone completion:** which stage of the workflow each model actually reaches.
- **How runs ended:** exited vs. timeout vs. error (LHTB's Figure 6 analysis).

## Layout

```
reward_lens/
  ctrf.py           read per-test verifier results
  rubric.py         rubric format, validation, test -> milestone matching
  rewards.py        binary, naive dense and milestone rewards
  harbor.py         discover Harbor trials and how they ended
  report.py         aggregate, rank and render the comparison
  cli.py            reward-lens regrade / rubric-init
examples/           example rubric
tests/              unit tests (pytest)
```

## Status

Work in progress. Built as a hands-on study of the two papers above.

## License

MIT
