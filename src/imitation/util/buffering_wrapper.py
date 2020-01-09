from typing import List

from stable_baselines.common.vec_env import VecEnv, VecEnvWrapper

from imitation.util import rollout


class BufferingWrapper(VecEnvWrapper):
  """Saves transitions of underlying VecEnv.

  Retrieve saved transitions using `pop_transitions()`.
  """

  def __init__(self, venv: VecEnv, error_on_premature_reset: bool = True):
    """
    Args:
      venv: The wrapped VecEnv.
      error_on_premature_reset: Error if `reset()` is called on this wrapper
        and there are saved samples that haven't yet been accessed.
    """
    super().__init__(venv)
    self.error_on_premature_reset = error_on_premature_reset
    self._trajectories = []
    self._init_reset = False
    self._traj_accum = None
    self._saved_acts = None
    self.n_transitions = None

  def reset(self, **kwargs):
    if (self._init_reset and self.error_on_premature_reset
        and self.n_transitions > 0):  # noqa: E127
      raise RuntimeError(
        "BufferingWrapper reset() before samples were accessed")
    self._init_reset = True
    self.n_transitions = 0
    obs = self.venv.reset(**kwargs)
    self._traj_accum = rollout.TrajectoryAccumulator()
    for i, ob in enumerate(obs):
      self._traj_accum.add_step({"obs": ob}, key=i)
    return obs

  def step_async(self, actions):
    assert self._init_reset
    assert self._saved_acts is None
    self.acts_list.append(actions)
    self.venv.step_async()
    self._saved_acts = actions

  def step_wait(self):
    assert self._init_reset
    assert self._saved_acts is not None
    acts = self._saved_acts
    obs, rews, dones, infos = self.venv.step_wait(acts)
    finished_trajs = self._traj_accum.add_steps_and_auto_finish(
      acts, obs, rews, dones, infos)
    self._trajectories.extend(finished_trajs)
    self.n_transitions += self.num_envs
    return obs, rews, dones, infos

  def _finish_partial_trajectories(self) -> List[rollout.Trajectory]:
    """Finishes and returns partial trajectories in `self._traj_accum`."""
    trajs = []
    for i in range(self.num_envs):
      # Check that we have any transitions at all.
      n_transitions = len(self._traj_accum.partial_trajectories[i]) - 1
      assert n_transitions >= 0, "Invalid TrajectoryAccumulator state"
      if n_transitions >= 1:
        traj = self._traj_accum.finish_trajectory(i)
        trajs.append(traj)

        # Reinitialize a partial trajectory starting with the final observation.
        self._traj_accum.add_step({'obs': traj.obs[-1]})
    return trajs

  def pop_transitions(self) -> rollout.Transitions:
    """Pops recorded transitions, returning them as an instance of Transitions.
    """
    partial_trajs = self._finish_partial_trajectories()
    self._trajectories.extend(partial_trajs)
    transitions = rollout.flatten_trajectories(self.trajectories)
    assert len(transitions.obs) == self.n_transitions
    self._trajectories = []
    self.n_transitions = 0
    return transitions