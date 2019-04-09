"""Finite-horizon discrete environments with known transition dynamics. These
are handy when you want to perform exact maxent policy optimisation."""

import abc

import numpy as np

import gym


class ModelBasedEnv(gym.Env, metaclass=abc.ABCMeta):
    def __init__(self):
        self.seed()
        self.reset()

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 1 << 31)
        self.rand_state = np.random.RandomState(seed=seed)
        return [seed]

    def reset(self):
        self.cur_state = 0
        self.actions_taken = 0
        # idea: action-dependent observations are stupid, so instead I just
        # have state-dependent observations and choose to pass my action
        # descriptions to the agent like a normal person
        raise NotImplementedError("shit, how do I get initial observation when my observation model is action-dependent?")

    def step(self, action):
        old_state = self.cur_state
        out_dist = self.transition_matrix[old_state, action]
        choice_states = np.arange(self.n_states)
        next_state = int(np.random.choice(choice_states, p=out_dist, size=()))
        self.cur_state = next_state
        self.actions_taken += 1
        done = self.actions_taken >= self.horizon
        reward = self.reward_matrix[old_state, action, next_state]
        raise NotImplementedError(
            "still need to figure out what I'll do with reward")
        return self._make_obs()
        return (next_state, reward, done, {
            "old_state": old_state,
            "new_state": next_state
        })

    @property
    def n_states(self):
        """Number of states in this MDP (int)."""
        return self.transition_matrix.shape[0]

    @property
    def n_actions(self):
        """Number of actions in this MDP (int)."""
        return self.transition_matrix.shape[1]

    # ############################### #
    # METHODS THAT MUST BE OVERRIDDEN #
    # ############################### #

    @abc.abstractmethod
    @property
    def transition_matrix(self):
        """Yield a 3D transition matrix with dimensions corresponding to
        current state, current action, and next state (in that order). In other
        words, if `T` is our returned matrix, then `T[s,a,sprime]` is the
        chance of transitioning into state `sprime` after taking action `a` in
        state `s`."""
        pass

    @abc.abstractmethod
    @property
    def observation_matrix(self):
        """Yields 3D observation matrix with dimensions corresponding to
        current state, current action, and observation dimension."""
        pass

    @abc.abstractmethod
    @property
    def reward_matrix(self):
        """Yields 3D reward matrix with dimensions corresponding to current
        state, current action, and observation dimension."""
        pass

    @abc.abstractmethod
    @property
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
    out_mat = np.zeros(size=(n_states, n_actions, n_states), dtype='float32')
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
        n_actions,
        # should we have random observations (True) or one-hot
        # observations (False)?
        is_random,
        # should observations depend only on state, and not on action?
        is_state_only,
        # in case is_random==True: what should dimension of
        # observations be?
        obs_dim,
        # can pass in an np.random.RandomState if desired
        rand_state=np.random):
    if not is_random:
        assert obs_dim is None
    if is_state_only:
        if is_random:
            obs_2d = np.random.normal(0, 1, (n_states, obs_dim))
        else:
            obs_2d = np.identity(n_states)
        target_shape = (obs_2d.shape[0], n_actions, obs_2d.shape[1])
        obs_mat = np.broadcast_to(obs_2d[:, None, :], target_shape)
    else:
        if is_random:
            flat_mat = np.random.normal(0, 1, (n_states * n_actions, obs_dim))
        else:
            flat_mat = np.identity(n_states * n_actions)
        obs_mat = flat_mat.reshape((n_states, n_actions, flat_mat.shape[1]))
    assert obs_mat.ndim == 3 \
        and obs_mat.shape[:2] == (n_states, n_actions, ) \
        and obs_mat.shape[2] > 0
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
                 obs_state_only=True,
                 generator_seed=None):
        super(self, ModelBasedEnv).__init__()
        if generator_seed is None:
            generator_seed = np.random.randint(0, 1 << 31)
        rand_gen = np.random.RandomState(seed=generator_seed)
        if random_obs:
            if obs_dim is None:
                obs_dim = n_states
        else:
            assert obs_dim is None
        self.observation_matrix = make_obs_mat(
            n_states, n_actions, random_obs, obs_dim, rand_state=rand_gen)
        self.transition_matrix = make_random_trans_mat(
            n_states, n_actions, branch_factor, rand_state=rand_gen)
        self.horizon = horizon
