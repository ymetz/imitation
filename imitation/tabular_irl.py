"""Finite-horizon tabular MCE IRL, as described in Ziebart's thesis. See
chapters 9 and 10 of
http://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf. Also includes
some Numpy-based optimisers so that this code can be run without
PyTorch/TensorFlow."""

import abc

import numpy as np


def mce_partition_fh(env, *, R=None):
    """Calculate V^soft, Q^soft, and pi using recurrences (9.1), (9.2), and
    (9.3). Stop once l-infty distance between Vs is less than linf_eps. This is
    the finite-horizon variant."""

    # TODO: write a non-finite-horizon variant of this. It will have to use
    # discounting if you want it to converge.

    # shorthand
    horizon = env.horizon
    n_states = env.n_states
    n_actions = env.n_actions
    T = env.transition_matrix
    if R is None:
        R = env.reward_matrix

    # actual algorithm
    V = np.full((horizon + 1, n_states, ), -np.inf)
    V[horizon, :] = 0  # so that Z_T(s)=exp(0) (no reward at end)
    Q = np.zeros((horizon, n_actions, n_states))
    for t in range(horizon)[::-1]:
        # TODO: figure out which states I want to constrain state sequences to
        # end in (probably not necessarily in finite-horizon case)
        V[t, :] = np.zeros((n_states, ))
        for a in range(n_actions):
            Q[t, a, :] = R + T[:, a, :] @ V[t + 1, :]
            # np.logaddexp does something equivalent to Ziebart's "stable
            # softmax" (Algorithm 9.2)
            V[t, :] = np.logaddexp(V[t, :], Q[t, a, :])

    # transpose Q so that it's states-first, actions-last
    Q = Q.transpose((0, 2, 1))
    pi = np.exp(Q - V[:horizon, :, None])  # eqn. (9.1)

    return V, Q, pi


def mce_partition_ih(env, gamma, *, R=None):
    """Infinite-horizon version of mce_partition_fh."""

    raise NotImplementedError("I still need to write this")

    # shorthand
    horizon = env.horizon
    n_states = env.n_states
    n_actions = env.n_actions
    T = env.transition_matrix
    if R is None:
        R = env.reward_matrix

    # actual algorithm
    V = np.full((horizon + 1, n_states, ), -np.inf)
    V[horizon, :] = 0  # so that Z_T(s)=exp(0) (no reward at end)
    Q = np.zeros((horizon, n_actions, n_states))
    for t in range(horizon)[::-1]:
        # TODO: figure out which states I want to constrain state sequences to
        # end in (probably not necessarily in finite-horizon case)
        V[t, :] = np.zeros((n_states, ))
        for a in range(n_actions):
            Q[t, a, :] = R + T[:, a, :] @ V[t + 1, :]
            # np.logaddexp does something equivalent to Ziebart's "stable
            # softmax" (Algorithm 9.2)
            V[t, :] = np.logaddexp(V[t, :], Q[t, a, :])

    # transpose Q so that it's states-first, actions-last
    Q = Q.transpose((0, 2, 1))
    pi = np.exp(Q - V[:horizon, :, None])  # eqn. (9.1)

    return V, Q, pi


def mce_occupancy_measures(env, *, pi=None, R=None):
    """Calculate state visitation frequency Ds for each state s under a given
    policy pi. You can get pi from mce_partition_func()."""

    # shorthand
    horizon = env.horizon
    n_states = env.n_states
    n_actions = env.n_actions
    T = env.transition_matrix
    if R is None:
        R = env.reward_matrix
    if pi is None:
        _, _, pi = mce_partition_fh(env, R=R)

    # we always start in s0, WLOG (for other distributions, just make all
    # actions in s0 take you to random state)
    init_states = np.zeros((n_states))
    init_states[0] = 1

    # TODO: do I also need to account for final state at horizon + 1? Maybe
    # that's imaginary (it certainly doesn't carry reward).
    D = np.zeros((horizon, n_states))
    D[0, :] = init_states
    for t in range(1, horizon):
        for a in range(n_actions):
            E = D[t - 1] * pi[t - 1, :, a]
            D[t, :] += E @ T[:, a, :]

    return D, D.sum(axis=0)


def maxent_irl(env, optimiser, feature_counts, linf_eps=1e-5):
    """Vanilla maxent IRL with whatever optimiser you want to use."""
    obs_mat = env.observation_matrix
    delta = linf_eps + 1
    t = 0
    while delta > linf_eps:
        rew_params = optimiser.current_params
        predicted_r = obs_mat @ rew_params
        _, visitations = mce_occupancy_measures(env, R=predicted_r)
        pol_feature_counts = visitations @ obs_mat
        grad = feature_counts - pol_feature_counts
        grad = -grad
        delta = np.max(np.abs(grad))
        if 0 == (t % 10):  # if 0 == (t % 500):
            print('Feature count error@iter % 3d: %f (||params||=%f, '
                  '||grad||=%f, ||fcount||=%f)' %
                  (t, delta, np.linalg.norm(rew_params), np.linalg.norm(grad),
                   np.linalg.norm(pol_feature_counts)))
        optimiser.step(grad)
        t += 1
    return optimiser.current_params, pol_feature_counts


def _get_grad_r_from_trajectories(env, good_pi, ntraj=100):
    traj_grads = []
    for i in range(ntraj):
        feats = env.reset()
        done = False
        t = 0
        while not done:
            policy_dist = good_pi[t, env.cur_state]
            assert np.all(np.isfinite(policy_dist)) \
                and np.sum(policy_dist) > 1e-5, good_pi
            if np.sum(policy_dist) < 1:
                # TODO: replace action randomness with something seeded,
                # somehow
                no_act_prob = 1 - np.sum(policy_dist)
                if np.random.random() < no_act_prob:
                    # no action, ends sequence
                    done = True
                    break
            policy_dist = policy_dist / np.sum(policy_dist)
            assert np.all(np.isfinite(policy_dist)) \
                and np.sum(policy_dist) > 1e-5, policy_dist
            action = np.random.choice(np.arange(env.n_actions), p=policy_dist)
            step_feats, _, done, _ = env.step(action)
            feats = feats + step_feats
            t += 1
        traj_grads.append(feats)
    return traj_grads


def maxent_irl_ng(env,
                  optimiser,
                  feature_counts,
                  linf_eps=1e-5,
                  constrained_update=False,
                  *,
                  fim_ident_eps=0.0,
                  step_denom_eps=1e-6):
    """Natural gradient IRL."""
    obs_mat = env.observation_matrix
    delta = linf_eps + 1
    t = 0
    while delta > linf_eps:
        rew_params = optimiser.current_params
        predicted_r = obs_mat @ rew_params
        _, _, pi = mce_partition_fh(env=env, R=predicted_r)
        _, visitations = mce_occupancy_measures(env, R=predicted_r, pi=pi)
        pol_feature_counts = visitations @ obs_mat
        # TODO: this is a shit way of calculating the Fisher; I should be able
        # to get it exact! If I can't get it exact then I should at least
        # figure out some way of decreasing the variance of this estimate
        # (control variates? CRN? etc.)
        traj_grads = _get_grad_r_from_trajectories(env, pi, ntraj=100)
        outer_prod_things = (t - pol_feature_counts for t in traj_grads)
        fim = np.mean([np.outer(m, m) for m in outer_prod_things], axis=0)
        fim = fim + fim_ident_eps * np.eye(len(fim))
        grad = feature_counts - pol_feature_counts
        grad = -grad
        # TODO: solve this with conjugate gradient
        step = np.linalg.solve(fim, grad)
        if constrained_update:
            # TODO: do this properly so that it works even with Adam (will
            # probably involve projecting back onto constraint set after
            # updating; see AMSGrad docs)
            sqrt_gg = np.sqrt(np.dot(grad, step))
            step = step / (sqrt_gg + step_denom_eps)
        delta = np.max(np.abs(grad))
        if 0 == (t % 10):  # if 0 == (t % 500):
            print('Feature count error@iter % 3d: %f (||params||=%f, '
                  '||grad||=%f, ||fcount||=%f, ||step||=%f)' %
                  (t, delta, np.linalg.norm(rew_params), np.linalg.norm(grad),
                   np.linalg.norm(pol_feature_counts), np.linalg.norm(step)))
        optimiser.step(step)
        t += 1
    return optimiser.current_params, pol_feature_counts


# ############################### #
# ######### OPTIMISERS ########## #
# ############################### #

# TODO: add a few different LR schedules (probably constant, 1/t, and 1/sqrt(t)
# step sizes)

# TODO: also add the ability to project back onto a constraint set for my
# experiments


class Optimiser(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def step(self, grad):
        """Take a step using the supplied gradient vector."""
        pass

    @property
    @abc.abstractmethod
    def current_params(self):
        """Return the parameters corresponding to the current iterate."""
        pass


class AMSGrad(Optimiser):
    """Fixed version of Adam optimiser, as described in
    https://openreview.net/pdf?id=ryQu7f-RZ. This should roughly correspond to
    a diagonal approximation to natural gradient, just as Adam does, but
    without the pesky non-convergence issues."""

    def __init__(self, x, alpha=1e-3, beta1=0.9, beta2=0.99, eps=1e-8):
        # x is initial parameter vector; alpha is step size; beta1 & beta2 are
        # as defined in AMSGrad paper; eps is added to sqrt(vhat) during
        # calculation of next iterate to ensure division does not overflow.
        param_size, = x.shape
        # first moment estimate
        self.m = np.zeros((param_size, ))
        # second moment estimate
        self.v = np.zeros((param_size, ))
        # max second moment
        self.vhat = np.zeros((param_size, ))
        # parameter estimate
        self.x = x
        # step sizes etc.
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def step(self, grad):
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        self.vhat = np.maximum(self.vhat, self.v)
        # 1e-5 for numerical stability
        denom = np.sqrt(self.vhat) + self.eps
        self.x = self.x - self.alpha * self.m / denom
        return self.x

    @property
    def current_params(self):
        return self.x


class SGD(Optimiser):
    """Standard gradient method."""

    def __init__(self, x, alpha=1e-3):
        self.x = x
        self.alpha = alpha
        self.cnt = 1

    def step(self, grad):
        self.x = self.x - self.alpha * grad
        return self.x

    @property
    def current_params(self):
        return self.x
