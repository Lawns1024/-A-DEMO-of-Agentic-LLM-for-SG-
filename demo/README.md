# CEVIoT Paper Reproduction Demo

This directory contains an offline, reproducible implementation of the core
stochastic-game pipeline in **Agentic LLM-Driven Formal Modeling and
Equilibrium Computation for Multi-Agent Stochastic Games in IoT Channel
Contention**.

The demo implements the paper's mathematical and verification path:

1. construct the four-agent, two-state, two-action general-sum game;
2. compute conditional collision probabilities using Equation (1);
3. compute per-agent rewards using Equation (2);
4. verify every transition row, reward row, and backoff profile;
5. run Correlated Equilibrium Value Iteration (CEVI) using Equations (3)-(4);
6. verify all correlated-equilibrium incentive constraints;
7. export correlated joint backoff recommendations for an ns-3 adapter; and
8. run a seeded packet/retry reproduction of the paper's 50-second experiment.

## What is and is not reproduced

The CEVI solver, stochastic-game checks, correlated policy, backoff mapping,
offered traffic, and reported performance metrics are executable. A real
Prolog verifier is included in `verify_model.pl`; when SWI-Prolog is not
installed, the same axioms are checked in Python and the omission is reported.

The paper does not publish prompts, LLM responses, numerical transition
probabilities, alpha/beta values, or steady-state transmission probabilities.
Those values therefore cannot be recovered exactly. This demo exposes its
concrete choices in `GameConfig` and writes the complete generated model to
`formal_game_model.json`. It does not claim that these unpublished parameters
are the authors' original values.

The small simulation is an algorithm-level reproduction, not a replacement
for the repository's `ns-3/cevi-iot-mac.cc` system experiment. Its traffic
matches the paper: four sources, 1024-byte payloads, one packet every 1500 us,
and seven MAC retries.

## Install and run

Python 3.10 or newer is recommended.

```bash
cd demo
python -m pip install -r requirements.txt
python demo_cevi.py --seconds 50 --seed 42 --output-dir .
```

To require the supplied Prolog verifier rather than using the Python fallback:

```bash
python demo_cevi.py --prolog required
```

Run the test suite with:

```bash
python -m unittest -v test_demo_cevi.py
```

## Expected seeded result

With `--seconds 50 --seed 42`, the demo converges in roughly 376 CEVI
iterations and produces results at the scale reported in the paper:

- aggregate throughput: approximately 21.8 Mbps;
- attempt-level collision rate: approximately 0.28; and
- efficiency: approximately 17.1 using the paper's stated formula
  `throughput / (1 + collision rate)`.

The paper text reports an average efficiency of 1.28, but that number is not
consistent with its own formula and a throughput of 21.8 Mbps. This demo uses
the printed formula without silently rescaling it.

## Output files

| File | Purpose |
| --- | --- |
| `formal_game_model.json` | Complete formal game tuple and generated tensors |
| `generated_game_facts.pl` | Prolog facts generated from the same model |
| `cevi_policy.csv` | Positive-probability joint policies and CW multipliers |
| `demo_results.csv` | Per-second reproduction metrics |
| `demo_summary.json` | Seed, convergence, and aggregate metrics |
| `demo_results.png` | Paper-style throughput, collision, and efficiency plot |

The policy CSV preserves correlation. For each state it gives a probability
distribution over a full four-node action vector, rather than independently
sampling each node's marginal. The `a_aggr` profile maps to base contention
window bounds `(1 x CWmin, 1 x CWmax)` and `a_cons` maps to
`(2 x CWmin, 4 x CWmax)`, preserving binary exponential backoff.

## Source layout

```text
demo/
|- demo_cevi.py          # model, CEVI solver, simulation, and exporters
|- verify_model.pl       # executable stochastic-axiom verifier
|- test_demo_cevi.py     # deterministic unit and reproduction tests
|- requirements.txt
|- README.md
|- formal_game_model.json
|- generated_game_facts.pl
|- cevi_policy.csv
|- demo_results.csv
|- demo_summary.json
`- demo_results.png
```
