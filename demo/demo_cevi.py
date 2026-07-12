#!/usr/bin/env python3
"""Reproducible CEVI demo for the four-node IoT contention game.

The implementation follows Equations (1)-(4) in the accompanying paper.  It
keeps the model parameters that the paper leaves unspecified explicit, checks
the generated stochastic game, solves a welfare-maximising correlated
equilibrium at every state, and produces a policy table suitable for an ns-3
policy adapter.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog


JointAction = tuple[int, ...]


@dataclass(frozen=True)
class BackoffProfile:
    min_multiplier: int
    max_multiplier: int


@dataclass(frozen=True)
class GameConfig:
    """Concrete, inspectable instantiation of the paper's game tuple."""

    n_agents: int = 4
    states: tuple[str, str] = ("s_idle", "s_congested")
    actions: tuple[str, str] = ("a_aggr", "a_cons")
    gamma: float = 0.95
    alpha: float = 1.0
    beta_by_state: dict[str, float] = field(
        default_factory=lambda: {"s_idle": 0.8, "s_congested": 2.5}
    )
    tau_by_action: dict[str, float] = field(
        default_factory=lambda: {"a_aggr": 0.145, "a_cons": 0.070}
    )
    backoff_profiles: dict[str, BackoffProfile] = field(
        default_factory=lambda: {
            "a_aggr": BackoffProfile(1, 1),
            "a_cons": BackoffProfile(2, 4),
        }
    )
    transition_bias_by_state: dict[str, float] = field(
        default_factory=lambda: {"s_idle": 0.04, "s_congested": 0.24}
    )
    transition_collision_gain: float = 1.75
    collision_threshold: float = 0.24
    payload_bytes: int = 1024
    client_interval_us: int = 1500
    retry_limit: int = 7

    def validate_shape(self) -> None:
        if self.n_agents != 4:
            raise ValueError("The paper experiment requires exactly four agents")
        if len(self.states) != 2 or len(self.actions) != 2:
            raise ValueError("The paper model requires two states and two actions")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must be in [0, 1)")
        for state in self.states:
            if state not in self.beta_by_state or state not in self.transition_bias_by_state:
                raise ValueError(f"Missing parameters for state {state}")
        for action in self.actions:
            if action not in self.tau_by_action or action not in self.backoff_profiles:
                raise ValueError(f"Missing parameters for action {action}")


@dataclass(frozen=True)
class CeviResult:
    policy: np.ndarray
    values: np.ndarray
    q_values: np.ndarray
    iterations: int
    residual: float


def all_joint_actions(cfg: GameConfig) -> list[JointAction]:
    return list(itertools.product(range(len(cfg.actions)), repeat=cfg.n_agents))


def action_index(joint_actions: Sequence[JointAction]) -> dict[JointAction, int]:
    return {action: index for index, action in enumerate(joint_actions)}


def action_code(cfg: GameConfig, joint_action: Sequence[int]) -> str:
    return "-".join(cfg.actions[action] for action in joint_action)


def tau(cfg: GameConfig, action: int) -> float:
    return cfg.tau_by_action[cfg.actions[action]]


def collision_probability(cfg: GameConfig, joint_action: Sequence[int], agent: int) -> float:
    """Equation (1): collision probability conditional on agent i transmitting."""

    no_other_transmission = math.prod(
        1.0 - tau(cfg, action)
        for other, action in enumerate(joint_action)
        if other != agent
    )
    return 1.0 - no_other_transmission


def reward(cfg: GameConfig, state: int, joint_action: Sequence[int], agent: int) -> float:
    """Equation (2): individual success utility minus collision cost."""

    state_name = cfg.states[state]
    tau_i = tau(cfg, joint_action[agent])
    p_collision = collision_probability(cfg, joint_action, agent)
    return cfg.alpha * tau_i * (1.0 - p_collision) - (
        cfg.beta_by_state[state_name] * tau_i * p_collision
    )


def mean_conditional_collision(cfg: GameConfig, joint_action: Sequence[int]) -> float:
    return float(
        np.mean(
            [
                collision_probability(cfg, joint_action, agent)
                for agent in range(cfg.n_agents)
            ]
        )
    )


def transition_row(cfg: GameConfig, state: int, joint_action: Sequence[int]) -> np.ndarray:
    """State transition derived from the preceding epoch's aggregate collision.

    The paper specifies the dependency but not its numeric probabilities.  The
    bias and gain are therefore explicit reproducibility parameters.
    """

    p_congested = cfg.transition_bias_by_state[cfg.states[state]] + (
        cfg.transition_collision_gain * mean_conditional_collision(cfg, joint_action)
    )
    p_congested = float(np.clip(p_congested, 0.01, 0.99))
    return np.array([1.0 - p_congested, p_congested], dtype=float)


def build_game_tensors(
    cfg: GameConfig, joint_actions: Sequence[JointAction]
) -> tuple[np.ndarray, np.ndarray]:
    rewards = np.empty((cfg.n_agents, len(cfg.states), len(joint_actions)))
    transitions = np.empty((len(cfg.states), len(joint_actions), len(cfg.states)))
    for state in range(len(cfg.states)):
        for action_idx, joint_action in enumerate(joint_actions):
            transitions[state, action_idx] = transition_row(cfg, state, joint_action)
            for agent in range(cfg.n_agents):
                rewards[agent, state, action_idx] = reward(
                    cfg, state, joint_action, agent
                )
    return rewards, transitions


def verify_game(
    cfg: GameConfig,
    joint_actions: Sequence[JointAction],
    rewards: np.ndarray,
    transitions: np.ndarray,
    tolerance: float = 1e-10,
) -> None:
    """Deterministic equivalent of the paper's Prolog admissibility filter."""

    cfg.validate_shape()
    expected_rewards = (cfg.n_agents, len(cfg.states), len(joint_actions))
    expected_transitions = (len(cfg.states), len(joint_actions), len(cfg.states))
    if rewards.shape != expected_rewards:
        raise ValueError(f"Reward tensor shape {rewards.shape} != {expected_rewards}")
    if transitions.shape != expected_transitions:
        raise ValueError(
            f"Transition tensor shape {transitions.shape} != {expected_transitions}"
        )
    if not np.all(np.isfinite(rewards)):
        raise ValueError("Rewards contain non-finite values")
    if not np.all(np.isfinite(transitions)):
        raise ValueError("Transitions contain non-finite values")
    if np.any(transitions < -tolerance) or np.any(transitions > 1.0 + tolerance):
        raise ValueError("Transition probabilities fall outside [0, 1]")
    row_sums = transitions.sum(axis=-1)
    if not np.allclose(row_sums, 1.0, atol=tolerance, rtol=0.0):
        state, action = np.argwhere(np.abs(row_sums - 1.0) > tolerance)[0]
        raise ValueError(
            f"Invalid transition row at state={cfg.states[state]}, "
            f"action={action_code(cfg, joint_actions[action])}: "
            f"sum={row_sums[state, action]}"
        )
    for profile in cfg.backoff_profiles.values():
        if profile.min_multiplier <= 0 or profile.max_multiplier < profile.min_multiplier:
            raise ValueError(f"Invalid backoff profile: {profile}")
    welfare = rewards.sum(axis=0)
    if np.allclose(welfare, 0.0, atol=tolerance):
        raise ValueError("Reward tensor is degenerate or zero-sum everywhere")


def ce_constraints(
    q_state: np.ndarray,
    cfg: GameConfig,
    joint_actions: Sequence[JointAction],
) -> tuple[np.ndarray, np.ndarray]:
    """Build Equation (4) incentive constraints in scipy's <= form."""

    lookup = action_index(joint_actions)
    rows: list[np.ndarray] = []
    for agent in range(cfg.n_agents):
        for recommended in range(len(cfg.actions)):
            for deviation in range(len(cfg.actions)):
                if recommended == deviation:
                    continue
                row = np.zeros(len(joint_actions), dtype=float)
                for idx, joint_action in enumerate(joint_actions):
                    if joint_action[agent] != recommended:
                        continue
                    deviated = list(joint_action)
                    deviated[agent] = deviation
                    deviated_idx = lookup[tuple(deviated)]
                    gain_from_obedience = (
                        q_state[agent, idx] - q_state[agent, deviated_idx]
                    )
                    row[idx] = -gain_from_obedience
                rows.append(row)
    return np.asarray(rows), np.zeros(len(rows), dtype=float)


def symmetrize_policy(
    policy: np.ndarray, joint_actions: Sequence[JointAction]
) -> np.ndarray:
    """Average over agent permutations to avoid arbitrary asymmetric LP ties."""

    lookup = action_index(joint_actions)
    symmetric = np.zeros_like(policy)
    for idx, probability in enumerate(policy):
        if probability <= 0.0:
            continue
        orbit = set(itertools.permutations(joint_actions[idx]))
        share = probability / len(orbit)
        for permuted in orbit:
            symmetric[lookup[permuted]] += share
    symmetric[symmetric < 1e-12] = 0.0
    return symmetric / symmetric.sum()


def solve_correlated_equilibrium(
    q_state: np.ndarray,
    cfg: GameConfig,
    joint_actions: Sequence[JointAction],
) -> tuple[np.ndarray, np.ndarray]:
    a_ub, b_ub = ce_constraints(q_state, cfg, joint_actions)
    result = linprog(
        c=-q_state.sum(axis=0),
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=np.ones((1, len(joint_actions))),
        b_eq=np.ones(1),
        bounds=[(0.0, 1.0)] * len(joint_actions),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Correlated-equilibrium LP failed: {result.message}")
    policy = symmetrize_policy(result.x, joint_actions)
    return policy, q_state @ policy


def verify_correlated_equilibrium(
    policy: np.ndarray,
    q_state: np.ndarray,
    cfg: GameConfig,
    joint_actions: Sequence[JointAction],
    tolerance: float = 1e-7,
) -> float:
    if np.any(policy < -tolerance) or not np.isclose(policy.sum(), 1.0, atol=tolerance):
        raise ValueError("Policy is not a probability distribution")
    a_ub, b_ub = ce_constraints(q_state, cfg, joint_actions)
    violations = a_ub @ policy - b_ub
    max_violation = float(max(0.0, np.max(violations)))
    if max_violation > tolerance:
        raise ValueError(f"CE incentive constraint violated by {max_violation:.3e}")
    return max_violation


def cevi(
    cfg: GameConfig,
    rewards: np.ndarray,
    transitions: np.ndarray,
    joint_actions: Sequence[JointAction],
    max_iterations: int = 2_000,
    tolerance: float = 1e-10,
) -> CeviResult:
    """Equations (3)-(4): value iteration with a state-wise CE LP."""

    values = np.zeros((cfg.n_agents, len(cfg.states)), dtype=float)
    policy = np.full(
        (len(cfg.states), len(joint_actions)), 1.0 / len(joint_actions)
    )
    q_values = np.zeros(
        (cfg.n_agents, len(cfg.states), len(joint_actions)), dtype=float
    )
    residual = math.inf

    for iteration in range(1, max_iterations + 1):
        previous = values.copy()
        for state in range(len(cfg.states)):
            continuation = transitions[state] @ previous.T
            q_values[:, state, :] = rewards[:, state, :] + (
                cfg.gamma * continuation.T
            )
        for state in range(len(cfg.states)):
            policy[state], values[:, state] = solve_correlated_equilibrium(
                q_values[:, state, :], cfg, joint_actions
            )
        residual = float(np.max(np.abs(values - previous)))
        if residual <= tolerance:
            break
    else:
        raise RuntimeError(
            f"CEVI did not converge after {max_iterations} iterations; "
            f"residual={residual:.3e}"
        )

    for state in range(len(cfg.states)):
        verify_correlated_equilibrium(
            policy[state], q_values[:, state, :], cfg, joint_actions
        )
    return CeviResult(policy, values, q_values, iteration, residual)


def save_formal_model(
    path: Path,
    cfg: GameConfig,
    joint_actions: Sequence[JointAction],
    rewards: np.ndarray,
    transitions: np.ndarray,
) -> None:
    config = asdict(cfg)
    config["backoff_profiles"] = {
        action: asdict(profile) for action, profile in cfg.backoff_profiles.items()
    }
    payload = {
        "schema": "ceviot-stochastic-game-v1",
        "paper_equations": {"collision": 1, "reward": 2, "cevi": [3, 4]},
        "config": config,
        "joint_actions": [
            {
                "id": idx,
                "labels": [cfg.actions[action] for action in joint_action],
            }
            for idx, joint_action in enumerate(joint_actions)
        ],
        "transitions": transitions.tolist(),
        "rewards": rewards.tolist(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_prolog_facts(
    path: Path,
    cfg: GameConfig,
    joint_actions: Sequence[JointAction],
    rewards: np.ndarray,
    transitions: np.ndarray,
) -> None:
    lines = [
        f"expected_state_count({len(cfg.states)}).",
        f"expected_action_count({len(joint_actions)}).",
        f"expected_agent_count({cfg.n_agents}).",
    ]
    for action, profile in cfg.backoff_profiles.items():
        lines.append(
            f"profile('{action}', {profile.min_multiplier}, {profile.max_multiplier})."
        )
    for state_idx, state in enumerate(cfg.states):
        for action_idx, joint_action in enumerate(joint_actions):
            code = action_code(cfg, joint_action)
            for next_idx, next_state in enumerate(cfg.states):
                lines.append(
                    "transition("
                    f"'{state}', '{code}', '{next_state}', "
                    f"{transitions[state_idx, action_idx, next_idx]:.17g})."
                )
            for agent in range(cfg.n_agents):
                lines.append(
                    f"reward('{state}', '{code}', {agent}, "
                    f"{rewards[agent, state_idx, action_idx]:.17g})."
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_prolog_verifier(mode: str, verifier: Path, facts: Path) -> str:
    executable = shutil.which("swipl")
    if executable is None:
        if mode == "required":
            raise RuntimeError("SWI-Prolog is required but 'swipl' was not found")
        return "not installed; equivalent Python verification passed"
    if mode == "off":
        return "disabled"
    completed = subprocess.run(
        [executable, "-q", "-s", str(verifier), "--", str(facts)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Prolog verification failed: {detail}")
    return completed.stdout.strip() or "passed"


def write_policy_csv(
    path: Path,
    cfg: GameConfig,
    joint_actions: Sequence[JointAction],
    result: CeviResult,
    minimum_probability: float = 1e-9,
) -> None:
    profile_fields = [
        field
        for agent in range(cfg.n_agents)
        for field in (f"node_{agent}_cw_min_multiplier", f"node_{agent}_cw_max_multiplier")
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["state", "joint_action_id", "joint_action", "probability"]
            + profile_fields,
        )
        writer.writeheader()
        for state_idx, state in enumerate(cfg.states):
            for action_idx, probability in enumerate(result.policy[state_idx]):
                if probability < minimum_probability:
                    continue
                joint_action = joint_actions[action_idx]
                row: dict[str, object] = {
                    "state": state,
                    "joint_action_id": action_idx,
                    "joint_action": action_code(cfg, joint_action),
                    "probability": f"{probability:.12g}",
                }
                for agent, action in enumerate(joint_action):
                    profile = cfg.backoff_profiles[cfg.actions[action]]
                    row[f"node_{agent}_cw_min_multiplier"] = profile.min_multiplier
                    row[f"node_{agent}_cw_max_multiplier"] = profile.max_multiplier
                writer.writerow(row)


def simulate(
    cfg: GameConfig,
    policy: np.ndarray,
    joint_actions: Sequence[JointAction],
    seconds: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Packet/retry abstraction aligned with the ns-3 offered load.

    Each source offers one 1024-byte UDP packet every 1500 microseconds.  MAC
    collisions trigger retries, so successful goodput can remain near the
    offered 21.845 Mbps while the attempt-level collision rate is around 0.28.
    """

    if seconds <= 0:
        raise ValueError("seconds must be positive")
    rng = np.random.default_rng(seed)
    current_state = 1
    rows: list[dict[str, object]] = []

    for second in range(1, seconds + 1):
        packets_per_agent = (
            math.floor(second * 1_000_000 / cfg.client_interval_us)
            - math.floor((second - 1) * 1_000_000 / cfg.client_interval_us)
        )
        action_idx = int(rng.choice(len(joint_actions), p=policy[current_state]))
        joint_action = joint_actions[action_idx]
        failed_attempts = 0
        successful_attempts = 0
        delivered_packets = 0

        for agent in range(cfg.n_agents):
            p_collision = collision_probability(cfg, joint_action, agent)
            failures_before_success = rng.geometric(
                1.0 - p_collision, size=packets_per_agent
            ) - 1
            delivered = failures_before_success <= cfg.retry_limit
            delivered_packets += int(delivered.sum())
            successful_attempts += int(delivered.sum())
            failed_attempts += int(
                np.minimum(failures_before_success, cfg.retry_limit + 1).sum()
            )

        throughput_mbps = delivered_packets * cfg.payload_bytes * 8 / 1e6
        attempts = successful_attempts + failed_attempts
        collision_rate = failed_attempts / attempts if attempts else 0.0
        efficiency = throughput_mbps / (1.0 + collision_rate)
        next_state = int(collision_rate > cfg.collision_threshold)
        rows.append(
            {
                "time_s": second,
                "state": cfg.states[current_state],
                "joint_action_id": action_idx,
                "joint_action": action_code(cfg, joint_action),
                "throughput_mbps": throughput_mbps,
                "collision_rate": collision_rate,
                "efficiency": efficiency,
                "delivered_packets": delivered_packets,
                "failed_attempts": failed_attempts,
            }
        )
        current_state = next_state

    summary = {
        "avg_throughput_mbps": float(
            np.mean([float(row["throughput_mbps"]) for row in rows])
        ),
        "avg_collision_rate": float(
            np.mean([float(row["collision_rate"]) for row in rows])
        ),
        "avg_efficiency": float(np.mean([float(row["efficiency"]) for row in rows])),
        "offered_load_mbps": (
            cfg.n_agents * cfg.payload_bytes * 8 * 1_000_000
            / cfg.client_interval_us
            / 1e6
        ),
    }
    return rows, summary


def write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("No result rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(path: Path, rows: Sequence[dict[str, object]], summary: dict[str, float]) -> None:
    time = np.asarray([row["time_s"] for row in rows], dtype=float)
    throughput = np.asarray([row["throughput_mbps"] for row in rows], dtype=float)
    collision = np.asarray([row["collision_rate"] for row in rows], dtype=float)
    efficiency = np.asarray([row["efficiency"] for row in rows], dtype=float)

    plt.style.use("seaborn-v0_8-whitegrid")
    figure = plt.figure(figsize=(10.5, 6.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[1, :]),
    ]
    series = [
        (throughput, "Throughput Over Time", "Throughput (Mbps)", "#1565c0"),
        (collision, "Collision Rate Over Time", "Collision Rate", "#d32f2f"),
        (efficiency, "Efficiency Over Time", "Efficiency", "#00897b"),
    ]
    averages = [
        summary["avg_throughput_mbps"],
        summary["avg_collision_rate"],
        summary["avg_efficiency"],
    ]
    for axis, (values, title, ylabel, color), average in zip(axes, series, averages):
        axis.plot(time, values, color=color, linewidth=1.5)
        axis.axhline(average, color=color, linestyle="--", linewidth=1.0, alpha=0.7)
        axis.set_title(title)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel(ylabel)
        axis.text(
            0.98,
            0.93,
            f"mean = {average:.3f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
    axes[1].set_ylim(0.0, max(0.5, float(collision.max()) * 1.15))
    figure.suptitle("CEVI IoT MAC Reproduction", fontsize=14, fontweight="bold")
    figure.savefig(path, dpi=200)
    plt.close(figure)


def print_policy(
    cfg: GameConfig,
    joint_actions: Sequence[JointAction],
    result: CeviResult,
) -> None:
    print("=== Correlated-equilibrium policy ===")
    for state_idx, state in enumerate(cfg.states):
        print(f"{state}:")
        for action_idx in np.flatnonzero(result.policy[state_idx] > 1e-9):
            print(
                f"  p={result.policy[state_idx, action_idx]:.6f}  "
                f"{action_code(cfg, joint_actions[action_idx])}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the paper's four-agent CEVI contention demo"
    )
    parser.add_argument("--seconds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--prolog",
        choices=("auto", "required", "off"),
        default="auto",
        help="Run SWI-Prolog verification when available",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = GameConfig()
    joint_actions = all_joint_actions(cfg)
    rewards, transitions = build_game_tensors(cfg, joint_actions)
    verify_game(cfg, joint_actions, rewards, transitions)

    model_path = output_dir / "formal_game_model.json"
    facts_path = output_dir / "generated_game_facts.pl"
    save_formal_model(model_path, cfg, joint_actions, rewards, transitions)
    write_prolog_facts(facts_path, cfg, joint_actions, rewards, transitions)
    verifier = Path(__file__).with_name("verify_model.pl")
    prolog_status = run_prolog_verifier(args.prolog, verifier, facts_path)

    result = cevi(cfg, rewards, transitions, joint_actions)
    print(f"Model verification: passed; Prolog: {prolog_status}")
    print(
        f"CEVI convergence: {result.iterations} iterations, "
        f"residual={result.residual:.3e}"
    )
    print_policy(cfg, joint_actions, result)

    policy_path = output_dir / "cevi_policy.csv"
    results_path = output_dir / "demo_results.csv"
    plot_path = output_dir / "demo_results.png"
    summary_path = output_dir / "demo_summary.json"
    write_policy_csv(policy_path, cfg, joint_actions, result)
    rows, summary = simulate(cfg, result.policy, joint_actions, args.seconds, args.seed)
    write_rows(results_path, rows)
    save_plot(plot_path, rows, summary)
    summary.update(
        {
            "seed": args.seed,
            "seconds": args.seconds,
            "cevi_iterations": result.iterations,
            "cevi_residual": result.residual,
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Reproduction summary ===")
    print(f"Average throughput : {summary['avg_throughput_mbps']:.3f} Mbps")
    print(f"Average collision  : {summary['avg_collision_rate']:.3f}")
    print(f"Average efficiency : {summary['avg_efficiency']:.3f}")
    print(f"Offered load       : {summary['offered_load_mbps']:.3f} Mbps")
    print(f"Outputs             : {output_dir}")


if __name__ == "__main__":
    main()
