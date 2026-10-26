# terminal-bench-reward-lens

> Binary vs. dense rewards for AI agents: an RL / RLHF reward-design and evaluation toolkit for Terminal-Bench, Long-Horizon-Terminal-Bench and any Harbor task.

[![CI](https://github.com/phoenixlibbi/terminal-bench-reward-lens/actions/workflows/ci.yml/badge.svg)](https://github.com/phoenixlibbi/terminal-bench-reward-lens/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Same agent runs, two verifiers.** `reward-lens` (the CLI and Python package of this repo) grades terminal-agent trajectories under the binary reward of [Terminal-Bench 2.0](https://arxiv.org/abs/2601.11868) and the dense milestone reward of [Long-Horizon-Terminal-Bench](https://arxiv.org/abs/2607.08964), and measures what each one hides: progress that binary grading erases, and free credit that dense grading hands out.

It is a working toolkit for anyone who builds RL environments or evaluations on [Harbor](https://github.com/laude-institute/harbor) tasks:

| Command | What it does | Docker / LLM needed |
|---|---|---|
| `reward-lens regrade` | Re-grade recorded Harbor runs under three rewards; leaderboard, near-misses, rank changes, reward histogram, milestone completion | no / no |
| `reward-lens convert` | Turn a binary Terminal-Bench task into a dense-reward task (rubric + in-container grader) | no / no |
| `reward-lens check-rubric` | Statically verify a rubric covers every test exactly once | no / no |
| `reward-lens floor` | Reward-hacking floor test: run adversarial non-solutions through the real verifier | yes / no |
| `reward-lens sensitivity` | Bootstrap CIs and robustness of model rankings to rubric weights | no / no |
| `reward-lens run` | Run an agent (replay, LLM loop, oracle, nop) on a converted task and report every reward | yes / optional |

## Who is this for

| You are... | Use it to... | Commands |
|---|---|---|
| **Training agents with RL / RLHF** on terminal tasks | Get a shaped reward instead of a 0/1 signal, with integrity gates so partial credit can't be farmed; log binary, naive and milestone reward for every rollout | `convert`, Python API |
| **Building RL environments or benchmark tasks** | Check that every test is weighted exactly once, measure how much a cheater earns *before* shipping a task, and gate verifier changes in CI | `check-rubric`, `floor --max-free-credit` |
| **Running evals / leaderboards** | Re-grade thousands of existing Harbor runs without re-running a single agent; get near-misses, rank changes and bootstrap CIs; tell apart models that binary pass rates tie | `regrade`, `sensitivity` |
| **Doing reward-design research** | Compare sparse, naive-dense and weighted rewards on the same final states; study reward hacking and weight sensitivity | `regrade`, `floor`, `sensitivity` |
| **Developing or debugging agents** | Replay a recorded trajectory in a fresh container to reproduce a failure, or run a minimal LLM agent against a local model with no API cost | `run --agent replay`, `run --agent loop` |

### Typical workflows

**"Our pass rate is 8%. Is the model making progress or not?"**
```bash
reward-lens rubric-init jobs/ --name my-task > rubric.toml   # then edit weights
reward-lens regrade jobs/ --rubric rubric.toml --out report.md
reward-lens sensitivity jobs/ --rubric rubric.toml           # are the differences real?
```

**"I want to train on partial credit without teaching the policy to game the tests."**
```bash
reward-lens check-rubric tasks/my-task --rubric rubric.toml
reward-lens floor tasks/my-task --rubric rubric.toml --work runs/floor --max-free-credit 0.1
reward-lens convert tasks/my-task --rubric rubric.toml --out tasks-dense/my-task
```

**"Why did this run fail, and is it reproducible?"**
```bash
reward-lens run tasks/my-task --rubric rubric.toml --agent replay --trajectory jobs/<trial> --work runs/replay
```

### Use in CI

Guard a task repository so that no verifier change silently lets non-solutions earn reward:

```yaml
# .github/workflows/verifiers.yml
on: [pull_request]
jobs:
  reward-floor:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install "terminal-bench-reward-lens[harness] @ git+https://github.com/phoenixlibbi/terminal-bench-reward-lens"
      - run: reward-lens check-rubric tasks/my-task --rubric tasks/my-task/rubric.toml
      - run: reward-lens floor tasks/my-task --rubric tasks/my-task/rubric.toml --work /tmp/floor --max-free-credit 0.1
```

Exit codes: `0` ok, `1` a gate failed (`--strict`, `--max-free-credit`), `2` usage or rubric error.

---

## Background: two papers, one question

**Terminal-Bench 2.0** (Merrill et al., 2026) evaluates agents on 89 hard, realistic tasks inside Docker containers. Each task is an instruction, an environment, a human-written oracle solution and a pytest verifier that checks the **final container state**. The reward is binary: 1 if every test passes, else 0. Its main contribution is trust: three-stage human review, CI checks that the oracle passes and a no-op agent fails, and an *adversarial exploit agent* that tries to cheat the tests.

**Long-Horizon-Terminal-Bench** (Li et al., 2026) keeps the same task format but targets workflows of hundreds of steps (≈239 episodes and ≈89 minutes per run). Its argument: at this horizon a binary reward is too **sparse**. An agent that completes 9 of 10 stages scores the same 0 as one that did nothing. So each task is decomposed into weighted subtasks and graded as

$$R = \frac{\sum_k w_k\, r_k}{\sum_k w_k}, \qquad r_k \in [0, 1],$$

with a run counted as solved at R ≥ 0.95. Public checks are cheap and low-weight; most reward sits in hidden stress tests.

| | Terminal-Bench 2.0 | Long-Horizon-Terminal-Bench |
|---|---|---|
| Tasks | 89, mostly < 20 min | 46, ≈ 90 min, hundreds of steps |
| Verifier output | 0 or 1 | R ∈ [0, 1] from weighted subtasks |
| When the verifier runs | once, at the end | **also once, at the end** |
| RL view | sparse outcome reward | fine-grained terminal reward with partial credit |
| Anti-cheat | human audits + exploit agent | hidden, high-weight stress tests |
| Main failure mode found | execution errors | running out of horizon; "false finishes" |

Two observations motivated this project:

1. **"Dense" here is about resolution, not timing.** Both verifiers score the final state once. LHTB's reward is a graded terminal reward, not a per-step process reward. That makes it a drop-in replacement for a Terminal-Bench verifier, which is what `reward-lens convert` does.
2. **A denser reward is also easier to game.** Terminal-Bench measured exploitability with an adversarial agent; LHTB never measures how much reward a non-solution can collect under partial credit. Real task verifiers spend many tests on cheap properties (file exists, schema, determinism, no input modification). Weighted naively, those tests pay out for outputs that are wrong. `reward-lens floor` measures exactly that.

## The three rewards

For every run, reward-lens computes:

| Reward | Definition | Stands for |
|---|---|---|
| **Binary** | 1 if no test fails, else 0 | Terminal-Bench 2.0 |
| **Naive dense** | share of tests passed, all equal | the obvious shortcut |
| **Milestone R** | Σ w_k r_k / Σ w_k over rubric milestones; 0 if an integrity gate fails | Long-Horizon-Terminal-Bench, with gates |

Comparing binary with milestone R shows what sparse grading hides. Comparing naive dense with milestone R isolates the effect of **weighting**.

## Results

Case study on two production-style Harbor tasks that already had recorded agent runs: 52 runs from two LLMs (anonymized as Model A and Model B) on the Terminus 2 scaffold. Both tasks are multi-stage ML-pipeline tasks graded by pytest verifiers (16 and 26 checks); a rubric of 5–6 weighted milestones plus one integrity gate was written for each.

### 1. Binary grading hides progress, and naive dense grading inflates it

| Task | Model | Runs | Binary pass | Naive dense | **Milestone R** |
|---|---|---|---|---|---|
| A | Model A | 16 | 12% | 0.66 | **0.54** |
| A | Model B | 10 | 0% | 0.36 | **0.16** |
| B | Model A | 16 | 12% | 0.73 | **0.30** |
| B | Model B | 10 | 0% | 0.67 | **0.20** |

- **Binary hides progress.** On Task A, the runs binary grading scores 0 reach a mean milestone R of 0.34. The milestone breakdown shows Model B stalls at an early stage, scoring 0.00 on the three later ones, while Model A reaches every stage partially. A binary score only says "12% vs 0%".
- **Naive dense inflates.** On Task B, failed runs pass on average **68% of the tests but earn R = 0.20**: **+0.48 of free credit per run**. Most of the task's tests check files, schema and determinism, which a wrong model passes easily.

### 2. Dense rewards give statistical power that binary rewards lack

95% bootstrap CIs on Task A:

| Model | Binary pass (95% CI) | Milestone R (95% CI) |
|---|---|---|
| Model A | 12% [0%, 31%] | 0.54 [0.42, 0.67] |
| Model B | 0% [0%, 0%] | 0.16 [0.10, 0.21] |

With 10–16 runs per model the binary intervals overlap; the milestone intervals do not. The ranking is unchanged in **100%** of 1,500 weight resamples (Dirichlet α = 50 and α = 10 around the rubric, and a flat Dirichlet), so the conclusion does not depend on the chosen weights. Absolute R does: under arbitrary weights Model B's mean R on Task B ranges from 0.27 to 0.88, which is why weights have to be justified by a floor test.

### 3. A non-solution collects up to 62% of naive dense reward

`reward-lens floor` on both tasks, in Docker, through the tasks' real verifiers:

| Adversary | Task A: binary / naive / **R** | Task B: binary / naive / **R** |
|---|---|---|
| `nop`: does nothing | 0 / 0.06 / **0.00** | 0 / 0.04 / **0.00** |
| `zeros`: correct files and schema, every number 0 | 0 / 0.38 / **0.20** | 0 / 0.62 / **0.16** |
| `perturb`: correct files and schema, values shuffled across rows | 0 / 0.38 / **0.20** | 0 / 0.62 / **0.16** |
| `oracle`: reference solution | 1 / 1.00 / **1.00** | 1 / 1.00 / **1.00** |

- On Task B, outputs in which **every number is zero pass 62% of the tests**. A policy trained on naive dense reward would be paid most of the reward for learning the output format.
- The weighted rubric cuts this to 0.16–0.20, all of it from deliverables, reproducibility and sanity milestones (weight 0.20–0.25). That residual is the honest price of rewarding format at all; `floor` makes it measurable and `--max-free-credit` makes it enforceable in CI.
- The floor test also caught a flaw in the first version of `perturb`: rescaling left 0/1 labels unchanged (`round(1 × 1.37 + 0.11) = 1`), so label checks still passed. Lists of JSON records are now treated as tables and shuffled column-wise.

### 4. End-to-end checks in Docker

| Check | Result |
|---|---|
| Converted Task A and Task B, oracle solution | binary 1, naive dense 1, milestone 1 on both, recorded by Harbor in `reward.json` |
| Recomputed binary reward vs. each run's original `reward.txt` | 54 / 54 runs agree |
| `ReplayAgent`: recorded passing run, re-typed in a fresh container | reward 1.0, **16 / 16** tests in the same state as the recording |
| `ReplayAgent`: recorded near-miss run (binary 0, R 0.92) | binary 0, R 0.917, **16 / 16** tests in the same state |
| `LoopAgent` with a local `llama3.1:8b` via Ollama | runs the full observe → JSON action → tmux loop through Harbor; valid actions every turn, at zero API cost |

## Install

```bash
git clone https://github.com/phoenixlibbi/terminal-bench-reward-lens && cd terminal-bench-reward-lens
pip install -e .                # re-grade, convert, check, floor reports, sensitivity: zero dependencies
pip install -e ".[harness]"     # + Harbor, for floor runs and the agent harness (Python 3.12+)
```

Python ≥ 3.11 on the host (≥ 3.12 for the `harness` extra, which Harbor requires). The in-container grader and adversary scripts are standard-library only and run on Python ≥ 3.8, so they work in any task image.

## Usage

### 1. Re-grade recorded runs

```bash
reward-lens rubric-init jobs/my-task --name my-task > rubric.toml    # draft: one milestone per test class
reward-lens regrade jobs/my-task --rubric rubric.toml --out report.md --json report.json
```

Any directory containing Harbor trial folders works (`result.json` + `verifier/ctrf.json`). The report contains:

- **Sanity checks:** the oracle must score R = 1.0, and the recomputed binary reward must match each run's own `reward.txt`.
- **Leaderboard:** binary pass rate, naive dense, mean R, pass @ R ≥ threshold, near-misses (0.75 ≤ R < 0.95), no-progress runs (R < 0.05), and each model's rank under binary vs. R.
- **What each reward hides:** mean R of runs binary scored 0, and the credit naive dense grants beyond the rubric.
- **Reward distribution** with the bucket edges of LHTB's Figure 4.
- **Milestone completion:** which stage of the workflow each model actually reaches.
- **How runs ended:** exited / timeout / error, the analysis behind LHTB's Figure 6.

### 2. Convert a task to dense grading

```bash
reward-lens check-rubric path/to/task --rubric rubric.toml
reward-lens convert path/to/task --rubric rubric.toml --out path/to/task-dense
harbor run -p path/to/task-dense -a oracle
```

The source task is only read (a content digest is checked before and after). The converted copy keeps the environment, instruction, solution and test setup, and replaces the 0/1 tail of `tests/test.sh` with:

```bash
pytest --ctrf /logs/verifier/ctrf.json /tests/test_state.py -rA || TEST_FAIL=1

# --- reward-lens: dense milestone reward (generated) ---
PY="$(command -v python3 || command -v python)"
"$PY" /tests/reward_lens_grader.py --ctrf /logs/verifier/ctrf.json --rubric /tests/rubric.json --out-dir /logs/verifier --primary milestone
```

The grader writes `reward.json`, which Harbor records in full:

```json
{"reward": 0.62, "binary": 0.0, "naive_dense": 0.81, "milestone": 0.62}
```

plus `reward_breakdown.json` with each milestone's score. `--primary` chooses which value Harbor treats as *the* reward, so the same task can train or evaluate under either philosophy. The grader always writes a reward, even when pytest crashes before producing a report.

### 3. Reward-hacking floor test

```bash
reward-lens floor path/to/task --rubric rubric.toml --work runs/floor
```

Four adversaries are run through the task's real verifier in Docker, as replacement `solution/solve.sh` scripts executed by Harbor's oracle agent:

| Adversary | Behaviour | A sound verifier gives |
|---|---|---|
| `nop` | does nothing | 0 |
| `zeros` | runs the reference solution, then sets every number in its outputs to 0 | ≈ 0 |
| `perturb` | runs the reference solution, then shuffles numeric columns/lists and rescales scalars | ≈ 0 |
| `oracle` | the reference solution | 1 |

`zeros` and `perturb` keep every file, column, row count and ID of a correct answer and none of its values: the cheapest form of "looks right". The report shows the reward each adversary collects under all three rewards and which milestones pay it.

### 4. Robustness

```bash
reward-lens sensitivity jobs/my-task --rubric rubric.toml
```

- 95% bootstrap CIs for each model's mean R and binary pass rate. With few runs per model, binary pass rates often cannot separate models that milestone R separates clearly.
- Milestone weights resampled from Dirichlet distributions centred on the rubric (α = 50, 10) and from a flat Dirichlet (any weighting): how often the model ranking survives, and how far each model's R moves.

### 5. Run agents: the harness

```bash
# Replay a recorded run in a fresh container (no model cost) and measure replay fidelity
reward-lens run path/to/task --rubric rubric.toml --agent replay --trajectory jobs/my-task/<trial> --work runs/replay

# A minimal Terminus-style LLM agent over any OpenAI-compatible endpoint
reward-lens run path/to/task --rubric rubric.toml --agent loop -m ollama/llama3.1:8b --work runs/loop
reward-lens run path/to/task --rubric rubric.toml --agent loop -m openai/gpt-4o-mini --ak max_turns=40 --work runs/loop

# Any Harbor agent works too, e.g. Claude Code on a subscription token or API key kept in a dotenv file
#   .env:  CLAUDE_CODE_OAUTH_TOKEN=...   (from `claude setup-token`)   and  CLAUDE_FORCE_OAUTH=1
reward-lens run path/to/task --rubric rubric.toml --agent claude-code -m anthropic/claude-sonnet-5-5 --env-file .env --work runs/cc

# Or use the agents directly with Harbor
harbor run -p path/to/task-dense -a reward_lens.harness.agents:LoopAgent -m ollama/llama3.1:8b
```

The harness is under 400 lines, docstrings included, and has no SDK dependencies:

- **`Terminal`** drives a tmux pane through the container's `exec()`. Literal keystrokes travel as base64 → `tmux load-buffer` → `paste-buffer`, so no command can break shell quoting; a keystroke string that is exactly a tmux key name (`C-c`, `Escape`, `Up`) presses that key. After each command it waits the requested duration, then keeps waiting while the screen is still changing.
- **`ReplayAgent`** re-types the keystrokes of a recorded Harbor (ATIF) trajectory and reports, per test, whether the replayed run ends in the same state as the recording.
- **`LoopAgent`** follows the Terminus 2 design: the model sees only the screen, replies with a JSON object (`analysis`, `plan`, `commands[{keystrokes, duration}]`, `task_complete`), must confirm completion once, and works with a sliding context window (system prompt + task + last N exchanges). Malformed replies are fed back as errors instead of crashing the run.

`reward-lens run` converts the task, runs the agent through Harbor and prints every reward plus the per-milestone breakdown.

## Writing a good rubric

See [`examples/rubric.example.toml`](examples/rubric.example.toml).

```toml
[task]
name = "demo-pipeline-repair"
pass_threshold = 0.95
near_miss = 0.75

[[gate]]                        # integrity: if any of these fails or never ran, R = 0
name = "integrity"
tests = ["test_inputs_not_modified", "test_no_label_leakage"]

[[milestone]]
name = "deliverables"
weight = 0.05
tests = ["TestOutputFiles::*", "TestSchema::*"]

[[milestone]]
name = "model_matches_reference"
weight = 0.70
tests = ["test_rmse", "test_mae", "test_coverage"]
scoring = "fraction"            # or "all": 1 only if every test in the milestone passes
```

Patterns are globs; without `::` they match the bare test name, and `[params]` suffixes are optional. Every test the verifier reports must belong to exactly one milestone or gate (or be listed in `ignore`); anything else is an error, so no check silently drops out of the reward.

Guidelines:

1. **Weight by evidence of capability, not by test count.** Files, schemas and determinism are necessary but say almost nothing about whether the task was solved. Keep them at 5–15% combined.
2. **Make integrity a gate, not a milestone.** Modifying inputs, leaking labels or reading the future should zero the reward, as in Terminal-Bench's integrity criterion. Partial credit for a cheating run is a reward-hacking incentive.
3. **One milestone per workflow stage.** Milestones should correspond to states a human would recognize as progress (data cleaned → features built → model fitted → calibrated → reported), so `regrade` can show where agents stall.
4. **Run `floor` before trusting a rubric.** If `zeros` or `perturb` collects meaningful reward, move weight from cheap checks to value checks, or add value checks.
5. **Run `sensitivity` before trusting a ranking.** If the ranking flips under α = 10, the conclusion depends on your weights rather than on the agents.

## Using it as an RL reward

The library has no dependencies and can be called from a training loop:

```python
from reward_lens import load_ctrf, load_rubric, milestone_reward, binary_reward

rubric = load_rubric("rubric.toml")
results = load_ctrf("trial/verifier/ctrf.json")       # {test_id: status}
r = milestone_reward(results, rubric)
print(r.reward, r.failed_gate, [(m.name, m.score) for m in r.milestones])
```

Or convert tasks once and read `reward.json` from Harbor's results. All three rewards are recorded for every rollout, so you can train on one and monitor the others (for example, train on milestone R and alert when it rises while binary stays flat).

## How it maps to the papers

| Paper concept | reward-lens |
|---|---|
| TB task = instruction + Docker env + tests + oracle | Harbor task format, read and written as is |
| TB binary verifier | `binary_reward`, `--primary binary` |
| TB integrity criterion | rubric `[[gate]]` |
| TB "oracle passes, no-op fails" CI check | `floor` adversaries `oracle` and `nop` |
| TB adversarial exploit agent | `floor` adversaries `zeros` / `perturb` |
| TB Terminus 2 neutral scaffold | `LoopAgent` (single terminal tool, JSON actions) |
| TB 95% confidence intervals | `sensitivity` bootstrap CIs |
| LHTB subtask reward R = Σ w r / Σ w | `milestone_reward` |
| LHTB binary / continuous / episode subtasks | `scoring = "all"` / `"fraction"` milestones over test groups |
| LHTB pass @ R ≥ 0.95, near-miss analysis | `regrade` leaderboard |
| LHTB Figure 4 reward histogram | `regrade` reward distribution |
| LHTB Figure 6 termination analysis | `regrade` "How runs ended" |

## Limitations

- Milestone scores are built from pytest outcomes, so a continuous LHTB subtask (e.g. "1.0 if within tolerance, decaying linearly") is approximated by groups of threshold tests. Native continuous checks would need the verifier to emit scores, not just pass/fail.
- The floor adversaries are format-preserving corruptions of the reference outputs. They bound what a format-aware cheater collects; they do not search for exploits the way an LLM exploit agent does.
- Replay fidelity is limited by nondeterminism (timing, network, unseeded randomness) and by commands typed while a previous one was still running.

## Development

```bash
pip install -e ".[dev,harness]"
pytest -q
```

```
reward_lens/
  ctrf.py              per-test verifier results
  rubric.py            rubric format, validation, test -> milestone matching
  rewards.py           binary, naive dense and milestone rewards
  harbor.py            discover Harbor trials and how they ended
  report.py            re-grade report
  convert.py           TB -> dense task converter
  floor.py             reward-hacking floor test
  sensitivity.py       bootstrap CIs, weight sensitivity
  run.py               run agents on converted tasks
  harness/             tmux terminal, trajectory replay, LLM loop, Harbor agents
  templates/           in-container grader and output corruptor (stdlib, Python 3.8+)
tests/                 unit tests, including grader/library parity on random runs
```

## References

- Merrill, Shaw, et al. *Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces.* arXiv:2601.11868, 2026.
- Li, Li, Shi, et al. *Long-Horizon-Terminal-Bench: Testing the Limits of Agents on Long-Horizon Terminal Tasks with Dense Reward-Based Grading.* arXiv:2607.08964, 2026.

## License

MIT
