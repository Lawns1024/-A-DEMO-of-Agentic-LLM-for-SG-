import unittest

import numpy as np

import demo_cevi as demo


class CeviDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = demo.GameConfig()
        cls.actions = demo.all_joint_actions(cls.cfg)
        cls.rewards, cls.transitions = demo.build_game_tensors(cls.cfg, cls.actions)
        cls.result = demo.cevi(cls.cfg, cls.rewards, cls.transitions, cls.actions)

    def test_transition_axioms(self):
        demo.verify_game(
            self.cfg, self.actions, self.rewards, self.transitions
        )
        np.testing.assert_allclose(self.transitions.sum(axis=-1), 1.0)
        self.assertTrue(np.all(self.transitions >= 0.0))

    def test_equation_one(self):
        all_aggressive = (0, 0, 0, 0)
        expected = 1.0 - (1.0 - self.cfg.tau_by_action["a_aggr"]) ** 3
        self.assertAlmostEqual(
            demo.collision_probability(self.cfg, all_aggressive, 0), expected
        )

    def test_cevi_converges_and_policy_is_valid(self):
        self.assertLessEqual(self.result.residual, 1e-10)
        for state in range(len(self.cfg.states)):
            violation = demo.verify_correlated_equilibrium(
                self.result.policy[state],
                self.result.q_values[:, state, :],
                self.cfg,
                self.actions,
            )
            self.assertLessEqual(violation, 1e-7)

    def test_congested_policy_is_correlated_and_fair(self):
        policy = self.result.policy[1]
        marginals = np.zeros(self.cfg.n_agents)
        for probability, joint_action in zip(policy, self.actions):
            for agent, action in enumerate(joint_action):
                marginals[agent] += probability * (action == 0)
        np.testing.assert_allclose(marginals, marginals[0], atol=1e-9)
        self.assertGreater(np.count_nonzero(policy > 1e-9), 1)

    def test_seeded_reproduction_matches_paper_scale(self):
        rows, summary = demo.simulate(
            self.cfg, self.result.policy, self.actions, seconds=50, seed=42
        )
        self.assertTrue(21.5 <= summary["avg_throughput_mbps"] <= 22.0)
        self.assertTrue(0.24 <= summary["avg_collision_rate"] <= 0.32)
        expected_efficiency = np.mean(
            [
                row["throughput_mbps"] / (1.0 + row["collision_rate"])
                for row in rows
            ]
        )
        self.assertAlmostEqual(summary["avg_efficiency"], expected_efficiency)


if __name__ == "__main__":
    unittest.main()
