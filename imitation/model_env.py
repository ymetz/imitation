"""Finite-horizon discrete environments with known transition dynamics. These
are handy when you want to perform exact maxent policy optimisation."""

import abc

import numpy as np

import gym
from gym import spaces


class ModelBasedEnv(gym.Env, metaclass=abc.ABCMeta):
    def __init__(self):
        self.cur_state = 0
        self.actions_taken = 0
        # we must cache action & observation spaces instead of reconstructing
        # anew so that the random state of the action space (!!) is preserved
        self._action_space = None
        self._observation_space = None
        self.seed()

    @property
    def action_space(self):
        if not self._action_space:
            self._action_space = spaces.Discrete(self.n_actions)
        return self._action_space

    @property
    def observation_space(self):
        if not self._observation_space:
            self._observation_space = spaces.Box(
                low=float('-inf'), high=float('inf'), shape=(self.obs_dim, ))
        return self._observation_space

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 1 << 31)
        self.rand_state = np.random.RandomState(seed=seed)
        return [seed]

    def reset(self):
        self.cur_state = 0
        self.actions_taken = 0
        return self.observation_matrix[self.cur_state]

    def step(self, action):
        old_state = self.cur_state
        out_dist = self.transition_matrix[old_state, action]
        choice_states = np.arange(self.n_states)
        next_state = int(
            self.rand_state.choice(choice_states, p=out_dist, size=()))
        self.cur_state = next_state
        self.actions_taken += 1
        done = self.actions_taken >= self.horizon
        reward = self.reward_matrix[old_state]
        assert np.isscalar(reward), reward
        obs = self.observation_matrix[next_state]
        assert obs.ndim == 1, obs.shape
        infos = {"old_state": old_state, "new_state": next_state}
        return obs, reward, done, infos

    @property
    def n_states(self):
        """Number of states in this MDP (int)."""
        return self.transition_matrix.shape[0]

    @property
    def n_actions(self):
        """Number of actions in this MDP (int)."""
        return self.transition_matrix.shape[1]

    @property
    def obs_dim(self):
        return self.observation_matrix.shape[-1]

    # ############################### #
    # METHODS THAT MUST BE OVERRIDDEN #
    # ############################### #

    @property
    @abc.abstractmethod
    def transition_matrix(self):
        """Yield a 3D transition matrix with dimensions corresponding to
        current state, current action, and next state (in that order). In other
        words, if `T` is our returned matrix, then `T[s,a,sprime]` is the
        chance of transitioning into state `sprime` after taking action `a` in
        state `s`."""
        pass

    @property
    @abc.abstractmethod
    def observation_matrix(self):
        """Yields 2D observation matrix with dimensions corresponding to
        current state (first dim) and elements of observation (second dim)."""
        pass

    @property
    @abc.abstractmethod
    def reward_matrix(self):
        """Yields 2D reward matrix with dimensions corresponding to current
        state and current action."""
        pass

    @property
    @abc.abstractmethod
    def horizon(self):
        """Number of actions that can be taken in an episode."""
        pass


def make_random_trans_mat(
        n_states,
        n_actions,
        # maximum number of successors of an action in any
        # state
        max_branch_factor,
        # give an np.random.RandomState
        rand_state=np.random):
    out_mat = np.zeros((n_states, n_actions, n_states), dtype='float32')
    state_array = np.arange(n_states)
    for start_state in state_array:
        for action in range(n_actions):
            # uniformly sample a number of successors in [1,max_branch_factor]
            # for this action
            succs = rand_state.randint(1, max_branch_factor + 1)
            next_states = np.random.choice(
                state_array, size=(succs, ), replace=False)
            # generate random vec in probability simplex
            next_vec = rand_state.dirichlet(np.ones((succs, )))
            next_vec = next_vec / np.sum(next_vec)
            out_mat[start_state, action, next_states] = next_vec
    # TODO: check reachability of every state from initial state s0, but also
    # that some states are *hard* to reach
    return out_mat


def make_obs_mat(
        n_states,
        # should we have random observations (True) or one-hot
        # observations (False)?
        is_random,
        # in case is_random==True: what should dimension of
        # observations be?
        obs_dim,
        # can pass in an np.random.RandomState if desired
        rand_state=np.random):
    if not is_random:
        assert obs_dim is None
    if is_random:
        obs_mat = np.random.normal(0, 1, (n_states, obs_dim))
    else:
        obs_mat = np.identity(n_states)
    assert obs_mat.ndim == 2 \
        and obs_mat.shape[:1] == (n_states, ) \
        and obs_mat.shape[1] > 0
    return obs_mat


class RandomMDP(ModelBasedEnv):
    def __init__(self,
                 n_states,
                 n_actions,
                 branch_factor,
                 horizon,
                 random_obs,
                 *,
                 obs_dim=None,
                 generator_seed=None):
        super().__init__()
        if generator_seed is None:
            generator_seed = np.random.randint(0, 1 << 31)
        # this generator is ONLY for constructing the MDP, not for controlling
        # random outcomes during rollouts
        rand_gen = np.random.RandomState(seed=generator_seed)
        if random_obs:
            if obs_dim is None:
                obs_dim = n_states
        else:
            assert obs_dim is None
        self._observation_matrix = make_obs_mat(
            n_states=n_states,
            is_random=random_obs,
            obs_dim=obs_dim,
            rand_state=rand_gen)
        self._transition_matrix = make_random_trans_mat(
            n_states=n_states,
            n_actions=n_actions,
            max_branch_factor=branch_factor,
            rand_state=rand_gen)
        self._horizon = horizon
        self._reward_weights = np.random.randn(
            self._observation_matrix.shape[-1])
        # TODO: should I have action-dependent rewards? If so, how do I make
        # the reward function aware of the current action?
        self._reward_matrix = self._observation_matrix @ self._reward_weights
        assert self._reward_matrix.shape == (self.n_states, )

    @property
    def observation_matrix(self):
        return self._observation_matrix

    @property
    def transition_matrix(self):
        return self._transition_matrix

    @property
    def reward_matrix(self):
        return self._reward_matrix

    @property
    def horizon(self):
        return self._horizon
