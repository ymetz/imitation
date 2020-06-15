"""Smoke tests for CLI programs in imitation.scripts.*

Every test in this file should use `parallel=False` to turn off multiprocessing because
codecov might interact poorly with multiprocessing. The 'fast' named_config for each
experiment implicitly sets parallel=False.
"""

import os.path as osp
import sys
from unittest import mock

import pytest
import ray.tune as tune

from imitation.scripts import (
    analyze,
    eval_policy,
    expert_demos,
    parallel,
    train_adversarial,
)

ALL_SCRIPTS_MODS = [analyze, eval_policy, expert_demos, parallel, train_adversarial]


@pytest.mark.parametrize("script_mod", ALL_SCRIPTS_MODS)
def test_main_console(script_mod):
    """Smoke tests of main entry point for some cheap coverage."""
    with mock.patch.object(sys, "argv", ["sacred-pytest-stub", "print_config"]):
        script_mod.main_console()
PARALLEL_CONFIG_UPDATES = [
    dict(
        sacred_ex_name="expert_demos",
        base_named_configs=["cartpole", "fast"],
        n_seeds=2,
        search_space={
            "config_updates": {
                "init_rl_kwargs": {"learning_rate": tune.grid_search([3e-4, 1e-4])},
            }
        },
    ),
    dict(
        sacred_ex_name="train_adversarial",
        base_named_configs=["cartpole", "gail", "fast"],
        base_config_updates={
            # Need absolute path because raylet runs in different working directory.
            "rollout_path": osp.abspath(
                "tests/data/expert_models/cartpole_0/rollouts/final.pkl"
            ),
        },
        search_space={
            "config_updates": {
                "init_trainer_kwargs": {
                    "reward_kwargs": {
                        "phi_units": tune.grid_search([[16, 16], [7, 9]]),
                    },
                },
            }
        },
    ),
]

PARALLEL_CONFIG_LOW_RESOURCE = {
    # CI server only has 2 cores.
    "init_kwargs": {"num_cpus": 2},
    # Memory is low enough we only want to run one job at a time.
    "resources_per_trial": {"cpu": 2},
}


@pytest.mark.parametrize("config_updates", PARALLEL_CONFIG_UPDATES)
def test_parallel(config_updates):
    """Hyperparam tuning smoke test."""
    # CI server only has 2 cores
    config_updates = dict(config_updates)
    config_updates.update(PARALLEL_CONFIG_LOW_RESOURCE)
    # No need for TemporaryDirectory because the hyperparameter tuning script
    # itself generates no artifacts, and "debug_log_root" sets inner experiment's
    # log_root="/tmp/parallel_debug/".
    run = parallel.parallel_ex.run(
        named_configs=["debug_log_root"], config_updates=config_updates
    )
    assert run.status == "COMPLETED"


def _generate_test_rollouts(tmpdir: str, env_named_config: str) -> str:
    expert_demos.expert_demos_ex.run(
        named_configs=[env_named_config, "fast"],
        config_updates=dict(rollout_save_interval=0, log_dir=tmpdir,),
    )
    rollout_path = osp.abspath(f"{tmpdir}/rollouts/final.pkl")
    return rollout_path


def test_parallel_train_adversarial_custom_env(tmpdir):
    env_named_config = "custom_ant"
    rollout_path = _generate_test_rollouts(tmpdir, env_named_config)

    config_updates = dict(
        sacred_ex_name="train_adversarial",
        n_seeds=1,
        base_named_configs=[env_named_config, "fast"],
        base_config_updates=dict(
            init_trainer_kwargs=dict(parallel=True, num_vec=2,),
            rollout_path=rollout_path,
        ),
    )
    config_updates.update(PARALLEL_CONFIG_LOW_RESOURCE)
    run = parallel.parallel_ex.run(
        named_configs=["debug_log_root"], config_updates=config_updates
    )
    assert run.status == "COMPLETED"
