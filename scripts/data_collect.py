import argparse

import gin.tf
import imitation.util as util
import ray
from ray import tune
import tensorflow as tf


def make_PPO2(env_name):
    """
    Hyperparameters and a vectorized environment for training a PPO2 expert.
    """
    env = util.make_vec_env(env_name, 8)
    # Didn't look at rl-baselines-zoo for this, but these hyperparameters
    # seem ok. They aren't great though.
    policy = stable_baselines.PPO2(
        util.FeedForward32Policy,
        env,
        verbose=0,
        tensorboard_log="data/tensorboard",
        learning_rate=3e-3,
        nminibatches=32,
        noptepochs=10,
        n_steps=2048)
    return policy


@gin.configurable
def main(env_name, total_timesteps):
    # tf.logging.set_verbosity(tf.logging.INFO)

    # policy = make_PPO2(env_name)

    # callback = util.make_save_policy_callback("data/demos")
    # policy.learn(total_timesteps, callback=callback)
    ray.init()
    tune.run(
        "IMPALA",
        name="impala-demo-" + env_name,
        local_dir="data/",
        config={
            "env": env_name,
            "model": {
                "fcnet_activation": "relu",
                "fcnet_hiddens": [32, 32],
                # "free_log_std": True
            },
            "optimizer": {
                "lr": 3e-3
            }
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gin_config", default='configs/cartpole_data_collect.gin')
    args = parser.parse_args()

    gin.parse_config_file(args.gin_config)

    main()
