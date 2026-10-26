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
| 1 | `reward_lens.regrade` | Reads Harbor trial logs (`verifier/ctrf.json`) and computes both the binary reward and a rubric-weighted milestone reward. Reports pass@1 vs. mean R, near-misses, ranking changes and a reward histogram. | planned |
| 2 | `reward_lens.convert` | Converts a Terminal-Bench-style Harbor task into a dense-reward task: adds a `rubric.toml` and a verifier that writes a fractional `reward.txt` plus a per-milestone breakdown. | planned |
| 3 | `reward_lens.floor` | Reward-hacking floor test. Runs adversarial "dummy" solutions (no-op, schema-only, input echo) through each verifier and measures how much free credit they collect. | planned |
| 4 | `reward_lens.harness` | Minimal Terminus-style agent loop (keystrokes → tmux in Docker → screen back to the model) that plugs into Harbor as a custom agent and reports both rewards live. Includes a zero-cost replay agent for recorded trajectories. | planned |

## Rubric format (draft)

```toml
[task]
name = "demo-pipeline-repair"
pass_threshold = 0.95

[[milestone]]
name = "outputs_and_schema"
weight = 0.10
tests = ["test_outputs_exist", "test_schema"]

[[milestone]]
name = "core_metrics_match_reference"
weight = 0.70
tests = ["test_rmse", "test_coverage"]

[[gate]]          # integrity: failing any of these sets R = 0
name = "anti_cheat"
tests = ["test_inputs_not_modified"]
```

## Layout

```
reward_lens/        library code
examples/           small original demo task + rubric
tests/              unit tests for the toolkit
```

## Status

Work in progress. Built as a hands-on study of the two papers above.

## License

MIT
