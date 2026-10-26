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
| `reward-lens floor` | Reward-hacking floor test: run adversarial non-solutions through the real verifier; weak-test report and suggested rubric fixes | yes / no |
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

Exit codes: `0` ok, `1` a gate failed (`--strict`, `--max-free-credit`, or an adversary produced no runs to check), `2` usage or rubric error, `3` Harbor/Docker failed.

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
| `LoopAgent` with a local `llama3.1:8b` via Ollama | full observe → JSON action → tmux loop through Harbor at zero API cost; 7 well-formed turns, then the task's 60-minute limit (recorded as a timeout, R = 0). The run exposed two harness issues, now fixed: the model typed into an unclosed command's `>` prompt, and one uncapped generation stalled |
| Recorded runs replayed into a **hardened** verifier (4 runs) | **26 / 26** original test outcomes reproduced in every replay; the new value tests then graded the rebuilt final states with no model calls |

### 5. Case study: hardening a verifier, step by step

The floor test showed *that* non-solutions collect reward; the weak-test report shows *why*. On Task B, an all-zeros answer passed 16 of 26 tests:

| Group of tests | Passed by the cheater because | Verdict |
|---|---|---|
| Files exist, columns, row count, IDs, metric keys, value types (8) | a wrong answer can be perfectly formatted | format-only |
| "lower ≤ prediction ≤ upper", "lower ≥ 0" (3) | 0 ≤ 0 ≤ 0 is true: the check is vacuous for constant outputs | format-only |
| Determinism (2) | the test read the same output file twice and compared it with itself: it cannot fail | format-only |
| Integrity: inputs untouched, no leakage, no future data (3) | the cheater did not cheat on those | integrity gate (correctly passes) |
| Only the 8 tests that compare values with the reference model failed it | | strong |

Four levels of fixes, measured on the same task. "Real failed runs" are three recorded agent runs (both models) that did not solve the task, and "passing run" a recorded run that did; for L3 all four were replayed into the hardened task so the new tests could grade them.

| Level | What changes | Cheaters (`zeros` / `perturb`) | 3 real failed runs | Passing run / oracle |
|---|---|---|---|---|
| L0: naive dense | every test worth the same | **0.62** / 0.62 | 0.69 | 1.00 |
| L1: weights + integrity gate | format checks 5–20% of the weight | 0.16 / 0.16 | 0.20–0.27 | 1.00 |
| L2: + conditional credit (`requires_any`, suggested by `floor`) | format credit only after some value check passes | **0.00** / 0.00 | **0.00**–0.27 | 1.00 |
| L3: + value tests (`reward_lens.checks`) | strict intervals, non-degenerate outputs, correlation with the reference, metrics recomputed from the predictions | 0.14 / 0.16 | 0.25–0.32 | 1.00 |
| **L2 + L3** | both | **0.00 / 0.00** | **0.25–0.32** | **1.00** |

What the table shows:

- **Reweighting (L1) is necessary but leaves a floor** of 0.16: format credit is still format credit.
- **Conditional credit (L2) removes the floor, but can make the reward sparse again.** On this task two of the three replayed failed runs drop to 0, exactly like the cheaters: through this verifier their partial credit *was* format credit. (On Task A, L2 cost real agents almost nothing: 0.54 → 0.54 and 0.16 → 0.15, because their partial credit came from real value checks.)
- **Value tests (L3) restore the signal.** The failed agents' outputs are honest: the metrics they report follow from the predictions they submitted, and their intervals are non-empty. The cheaters' outputs are not. With L2 + L3, every real failed run keeps 0.25–0.32, every cheater gets 0.00, and the passing run and the oracle keep 1.00.
- **L4 is the loop:** run `floor` after every verifier change and gate it in CI with `--max-free-credit`. While doing exactly that, the floor test caught three bugs in this toolkit itself: a `perturb` adversary that left 0/1 labels unchanged, a fix-suggester that trusted milestones mixing strong and weak tests (its "fix" changed 0.16 to 0.16), and the vacuous determinism tests above.

The lesson for anyone writing verifiers for RL: a test that a wrong answer can pass is not evidence, it is a reward-hacking surface. Measure it, weight it, condition it, and replace it with value checks.

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

Any directory containing Harbor trial folders works (`result.json` plus per-test results in `verifier/`: a CTRF report from `pytest-json-ctrf`, or the JUnit XML that converted tasks write). Tests with the same name in different files are kept apart, and reruns keep the worst result, so a failure can never be hidden. The report contains:

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

The source task is only read (a content digest is checked before and after). The converted copy keeps the environment, instruction and solution unchanged, and keeps the original verifier **byte for byte** as `tests/test_original.sh`. A small wrapper becomes the new `tests/test.sh`:

```bash
# --- reward-lens: dense milestone reward (generated) ---
export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} --junitxml=/logs/verifier/reward_lens_junit.xml"
bash /tests/test_original.sh                       # the task's own verifier, untouched
mv /logs/verifier/reward.txt /logs/verifier/reward_original.txt
python3 /tests/reward_lens_grader.py --results-dir /logs/verifier --rubric /tests/rubric.json --primary milestone ...
```

Per-test results come from pytest's **built-in** JUnit XML report, requested through `PYTEST_ADDOPTS`, so nothing has to be installed and verifiers that run pytest through pip, `uv run`, `uvx` or a venv all work. The original script is never edited, so line continuations, comments or `if pytest ...; then` forms cannot break. `convert` warns when a verifier calls pytest more than once or sets its own `--junitxml`.

The grader (standard library only, Python 3.8+) writes `reward.json`, which Harbor records in full:

```json
{"reward": 0.62, "binary": 0.0, "naive_dense": 0.81, "milestone": 0.62, "original_verifier": 0.0}
```

plus `reward_breakdown.json` with each milestone's score. `--primary` chooses which value Harbor treats as *the* reward, so the same task can train or evaluate under either philosophy; `original_verifier` keeps the task's own 0/1 verdict for comparison. The grader always writes a reward, even when the verifier crashes before producing results, and a rubric problem (e.g. a test added after the rubric was written) only zeroes the milestone reward, never binary or naive dense.

### 3. Reward-hacking floor test

```bash
reward-lens floor path/to/task --rubric rubric.toml --work runs/floor
```

Four adversaries are run through the task's real verifier in Docker, as replacement `solution/solve.sh` scripts executed by Harbor's oracle agent:

| Adversary | Behaviour | A sound verifier gives |
|---|---|---|
| `nop` | does nothing | 0 |
| `zeros` | runs the reference solution, then sets every number in its outputs to 0 | ≈ 0 |
| `perturb` | runs the reference solution, then shuffles numeric fields across rows and changes every other number | ≈ 0 |
| `oracle` | the reference solution | 1 |

Outputs are found by content snapshots taken before and after the reference solution (so copies with preserved timestamps count), and ID-like fields (`id`, `user_id`, `userId`, ...) are left alone. `zeros` and `perturb` keep every file, column, row count and ID of a correct answer and none of its values: the cheapest form of "looks right". The report shows the reward each adversary collects under all three rewards and which milestones pay it.

It then diagnoses the verifier, test by test:

| Verdict | Meaning |
|---|---|
| `vacuous` | passes even when the agent does nothing |
| `format-only` | passes for outputs with wrong values (both `zeros` and `perturb`) |
| `partial` | catches one kind of wrong answer but not the other (e.g. constant outputs but not shuffled ones) |
| `strong` | every non-solution fails it: real evidence of solving the task |
| `untested` | no non-solution result exists for it, so nothing can be concluded |
| `integrity gate` | expected to pass for anything that does not cheat |
| `broken` | the reference solution fails it |

Milestones are then judged by behaviour: a milestone that pays any non-solution is *leaking* (even if it also contains a strong test), and a milestone that pays none is *safe*. Every leaking milestone gets a suggested `requires_any` on the safe ones, with the worst non-solution reward before and after the change, and the suggestion is re-scored on the oracle run so a fix can never penalize the reference solution. `--suggest-rubric fixed.toml` writes the improved rubric, and `--max-free-credit 0.1` turns the whole check into a CI gate.

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
- **`LoopAgent`** follows the Terminus 2 design: the model sees only the screen, replies with a JSON object (`analysis`, `plan`, `commands[{keystrokes, duration}]`, `task_complete`), must confirm completion once, and works with a sliding context window (system prompt + task + last N exchanges). Malformed replies are fed back as errors instead of crashing the run; failed model calls are retried (the run stops after 3 in a row); generation is capped with `max_tokens` and a per-call timeout (`--ak max_tokens=... --ak timeout_sec=...`).

`reward-lens run` converts the task (or uses an already converted one), runs the agent through Harbor and prints every reward plus the per-milestone breakdown.

Replay has a second use: **re-verifying old runs against a new verifier.** Point `--agent replay` at a recorded trial and a task whose tests have since been strengthened; the run is rebuilt in a fresh container and graded by the new tests, with no model calls.

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

[[milestone]]
name = "reproducible"
weight = 0.05
tests = ["test_deterministic"]
requires_any = ["model_matches_reference"]   # only pays once some value check passes
```

`requires = [...]` withholds a milestone's credit unless **every** listed milestone scores above 0; `requires_any = [...]` unless **at least one** does. Conditions use the listed milestones' own test results, so they cannot form loops.

Patterns are globs; without `::` they match the bare test name, and `[params]` suffixes are optional. Every test the verifier reports must belong to exactly one milestone or gate (or be listed in `ignore`); anything else is an error, so no check silently drops out of the reward.

Guidelines:

1. **Weight by evidence of capability, not by test count.** Files, schemas and determinism are necessary but say almost nothing about whether the task was solved. Keep them at 5–15% combined.
2. **Make integrity a gate, not a milestone.** Modifying inputs, leaking labels or reading the future should zero the reward, as in Terminal-Bench's integrity criterion. Partial credit for a cheating run is a reward-hacking incentive.
3. **One milestone per workflow stage.** Milestones should correspond to states a human would recognize as progress (data cleaned → features built → model fitted → calibrated → reported), so `regrade` can show where agents stall.
4. **Make cheap credit conditional.** `requires_any` on format and reproducibility milestones means a run earns them only after showing some real progress.
5. **Run `floor` before trusting a rubric.** If `zeros` or `perturb` collects meaningful reward, follow the weak-test report: reweight, add conditions, or strengthen the tests.
6. **Run `sensitivity` before trusting a ranking.** If the ranking flips under α = 10, the conclusion depends on your weights rather than on the agents.

### Strengthening the tests themselves

Rubrics can only redistribute the evidence a verifier collects. When the weak-test report shows that a whole stage of the task is checked only by format tests, the verifier needs value checks. [`reward_lens.checks`](reward_lens/checks.py) provides them, standard library only and Python 3.8+ so they run in any task image (`reward-lens convert --with-checks` copies them into `tests/reward_lens_checks.py`):

| Helper | Catches |
|---|---|
| `assert_non_degenerate(values, min_unique=10)` | constant outputs ("all zeros") |
| `assert_strict_intervals(lower, pred, upper)` | empty intervals, which make `lower <= pred <= upper` pass vacuously |
| `assert_correlated(pred, reference, min_r=0.9)` | values that exist but sit in the wrong rows (shuffled) |
| `assert_consistent(reported, recomputed)` | reported metrics that do not follow from the submitted predictions |

```python
from reward_lens_checks import assert_consistent, assert_correlated, rmse

def test_predictions_track_reference():
    assert_correlated(pred["duration"], reference["duration"], min_r=0.9)

def test_metrics_follow_from_predictions():
    assert_consistent(json.load(open("/app/output/metrics.json")),
                      {"rmse": rmse(pred["duration"], truth["y"])})
```

The strongest pattern of all is to test the agent's code, not its output files: re-run it on hidden inputs and compare with the reference, as Long-Horizon-Terminal-Bench does with its hidden stress tests.

## Using it as an RL reward

The library has no dependencies and can be called from a training loop:

```python
from reward_lens import load_results, load_rubric, milestone_reward, binary_reward

rubric = load_rubric("rubric.toml")
results = load_results("trial/verifier")              # {test_id: status}, from CTRF or JUnit XML
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
  ctrf.py              per-test verifier results (CTRF JSON, JUnit XML)
  rubric.py            rubric format, validation, test -> milestone matching
  rewards.py           binary, naive dense and milestone rewards
  harbor.py            discover Harbor trials and how they ended
  report.py            re-grade report
  convert.py           TB -> dense task converter
  floor.py             reward-hacking floor test
  sensitivity.py       bootstrap CIs, weight sensitivity
  checks.py            value-check helpers for hardening verifiers
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
