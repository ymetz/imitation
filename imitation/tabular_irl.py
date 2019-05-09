"""Finite-horizon tabular MCE IRL, as described in Ziebart's thesis. See
chapters 9 and 10 of
http://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf. Also includes
some Numpy-based optimisers so that this code can be run without
PyTorch/TensorFlow."""

import abc

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.experimental.stax as jstax
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
    V = np.full((
        horizon + 1,
        n_states,
    ), -np.inf)
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


def maxent_irl(
        env,
        optimiser,
        rmodel,
        demo_state_om,
        # TODO: change termination condition to use gradient instead
        linf_eps=1e-5,
        print_interval=100,
        occupancy_change_dest=None,
        occupancy_error_dest=None):
    """Vanilla maxent IRL with whatever optimiser you want to use."""
    obs_mat = env.observation_matrix
    delta = linf_eps + 1
    t = 0
    assert demo_state_om.shape == (len(obs_mat), )
    rew_params = optimiser.current_params
    rmodel.set_params(rew_params)
    last_occ = None
    while delta > linf_eps:
        predicted_r, out_grads = rmodel.out_grads(obs_mat)
        _, visitations = mce_occupancy_measures(env, R=predicted_r)
        pol_grad = np.mean(visitations[:, None] * out_grads, axis=0)
        # gradient of reward function w.r.t parameters, with expectation taken
        # over states
        expert_grad = np.mean(demo_state_om[:, None] * out_grads, axis=0)
        # FIXME: is this even the correct gradient? Seems negated. Hmm.
        grad = pol_grad - expert_grad
        delta = np.max(np.abs(demo_state_om - visitations))
        if print_interval is not None and 0 == (t % print_interval):
            print('Occupancy measure error@iter % 3d: %f (||params||=%f, '
                  '||grad||=%f, ||E[dr/dw]||=%f)' %
                  (t, delta, np.linalg.norm(rew_params), np.linalg.norm(grad),
                   np.linalg.norm(pol_grad)))
        optimiser.step(grad)
        rew_params = optimiser.current_params
        rmodel.set_params(rew_params)
        t += 1
        if occupancy_error_dest is not None:
            occupancy_error_dest.append(
                np.sum(np.abs(demo_state_om - visitations)))
        if occupancy_change_dest is not None:
            # store change in L1 distance
            if last_occ is None:
                occupancy_change_dest.append(0)
            else:
                occupancy_change_dest.append(
                    np.sum(np.abs(last_occ - visitations)))
            last_occ = visitations
    return optimiser.current_params, visitations


def _get_grad_r_from_trajectories(env, good_pi, out_grads, ntraj=100):
    traj_grads = []
    for i in range(ntraj):
        env.reset()
        grads = out_grads[env.cur_state]
        done = False
        t = 0
        while not done:
            policy_dist = good_pi[t, env.cur_state]
            assert np.all(np.isfinite(policy_dist)), policy_dist
            assert np.sum(policy_dist) > 1e-5, (policy_dist, t, env.cur_state)
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
            _, _, done, infos = env.step(action)
            new_state = infos['new_state']
            step_grad = out_grads[new_state]
            grads = grads + step_grad
            t += 1
        traj_grads.append(grads)
    return traj_grads


def _approximate_fisher(env, good_pi, pol_grad, out_grads, ntraj,
                        fim_ident_eps):
    """Approximate the FIM with samples."""
    traj_grads = _get_grad_r_from_trajectories(env,
                                               good_pi,
                                               out_grads,
                                               ntraj=ntraj)
    # yes, you're meant to subtract pol_grad rather than expert_grad; this
    # is expected Hessian of KL between current policy and new policy
    outer_prod_things = list(t - pol_grad for t in traj_grads)
    fim = np.mean([np.outer(m, m) for m in outer_prod_things], axis=0)
    fim = fim + fim_ident_eps * np.eye(len(fim))
    return fim


def maxent_irl_ng(env,
                  optimiser,
                  rmodel,
                  demo_state_om,
                  linf_eps=1e-5,
                  constrained_update=False,
                  *,
                  fim_ident_eps=0.0,
                  step_denom_eps=1e-6,
                  fim_ntraj=100,
                  print_interval=100,
                  exact_fim=False):
    """Natural gradient IRL."""
    obs_mat = env.observation_matrix
    delta = linf_eps + 1
    t = 0
    rew_params = optimiser.current_params
    rmodel.set_params(rew_params)
    while delta > linf_eps:
        predicted_r, out_grads = rmodel.out_grads(obs_mat)
        _, _, pi = mce_partition_fh(env=env, R=predicted_r)
        _, visitations = mce_occupancy_measures(env, R=predicted_r, pi=pi)
        pol_grad = np.mean(visitations[:, None] * out_grads, axis=0)
        expert_grad = np.mean(demo_state_om[:, None] * out_grads, axis=0)
        if exact_fim:
            # TODO: implement exact FIM and check numerically that it matches
            # normal Fisher
            raise NotImplementedError("I haven't done this yet")
        else:
            fim = _approximate_fisher(env, pi, pol_grad, out_grads, fim_ntraj,
                                      fim_ident_eps)
        grad = pol_grad - expert_grad
        try:
            step = np.linalg.solve(fim, grad)
        except Exception as ex:
            print("Exception (%s). Eigenvalues were %s and ||grad|| is %s" %
                  (ex, np.linalg.eigvalsh(fim), np.linalg.norm(grad)))
            raise
        if constrained_update:
            # TODO: do this properly so that it works even with Adam (will
            # probably involve projecting back onto constraint set after
            # updating; see AMSGrad docs)
            sqrt_gg = np.sqrt(np.dot(grad, step))
            step = step / (sqrt_gg + step_denom_eps)
        delta = np.max(np.abs(visitations - demo_state_om))
        if 0 == (t % print_interval):
            print('Occupancy measure error@iter % 3d: %f (||params||=%f, '
                  '||grad||=%f, ||step||=%f)' %
                  (t, delta, np.linalg.norm(rew_params), np.linalg.norm(grad),
                   np.linalg.norm(step)))
        optimiser.step(step)
        rew_params = optimiser.current_params
        rmodel.set_params(rew_params)
        t += 1
    return optimiser.current_params, visitations


def _compute_feature_expectation_matrix(obs_mat, policy, trans_mat):
    """For each state & time-step, compute expected feature count obtained by
    rolling forward from that state & time step until the end of time using the
    given policy and transition dynamics."""
    n_states, d_obs = obs_mat.shape
    T, n_states_prime, n_actions = policy
    assert n_states == n_states_prime, \
        "obs mat shape %s and policy shape %s are mismatched" \
        % (obs_mat.shape, policy.shape)
    out_matrix = np.zeros((T, n_states, d_obs))
    out_matrix[T - 1] = obs_mat
    # go backwards to compute everything
    for t in range(T - 2, -1, -1):
        for s in range(n_states):
            next_state_dist = trans_mat[s].T @ policy[t, s]
            exp_features = next_state_dist[:, None] * obs_mat
            out_matrix[t, s] = exp_features
    return out_matrix


def _compute_feature_deltas(obs_mat, policy, trans_mat):
    """Compute M[t,s,a] = f(s) + E_{s'|a}[F[t+1,s]] - F[t,s]."""
    expect_feat_mat = _compute_feature_expectation_matrix(
        obs_mat, policy, trans_mat)
    T, n_states, d_obs = expect_feat_mat
    T, n_states_prime, n_actions = policy
    assert n_states == n_states_prime
    # indexing: what is time step? What is current state? What is chosen action
    # in trajectory? Final axis is observation.
    out_matrix = np.zeros((T, n_states, n_actions, d_obs))
    for t in range(T - 1):
        for s in range(n_states):
            for a in range(n_actions):
                this_exp = expect_feat_mat[s, s]
                # what's the distribution over next state future feature
                # expectation vectors given action a?
                trans_prob = trans_mat[s, a]
                next_exp = trans_prob[:, None] * expect_feat_mat[t + 1]
                delta = obs_mat[t, s] + next_exp - this_exp
                out_matrix[t, s, a] = delta
    return out_matrix


def _exact_fisher(obs_mat, policy, trans_mat, feats_mat):
    # can replace feats_mat with gradients & it all works out fine
    T, n_states, n_actions = policy.shape
    n_states, d_obs = feats_mat
    deltas = _compute_feature_deltas(obs_mat, policy, trans_mat)
    accumulator = np.zeros((d_obs, d_obs))
    # this is O([|T| |S| |A|]^2). Not as bad as explicitly enumerating all
    # trajectories, but still very bad!
    # Dimensions: time of (s,a); s; a; time of (s',a'); s'; a'
    occ_matrix = np.full((T, n_states, n_actions, T, n_states, n_actions), -1)
    # FIXME: get rid of all of occ_matrix except the bits that matter (last
    # timestep)
    # FIXME: roll up the inner ~2 loops (represents O(|S|*|A|) computation, so
    # Numpy should like it)
    # FIXME: once this ALL works, and you've verified that, add Numba to it so
    # that you can speed it up :)
    for t in range(T):
        for s in range(n_states):
            for a in range(n_actions):
                # what is the probability that we'll end up in this state from
                # previous state?
                if t == 0:
                    # assume we always start in state 0 & then just take an
                    # action
                    occ_matrix[t, s, a, t, s, a] \
                        = float(s == 0) * policy[0, s, a]
                else:
                    occ_matrix[t, s, a, t, s, a] \
                        = np.sum(occ_matrix[t-1, :, :, t, s, a])
                for t_prime in range(t + 1, T):
                    for s_prime in range(n_states):
                        for a_prime in range(n_actions):
                            # previous distribution over (s,a) pairs:
                            prev_dist = occ_matrix[t, s, a, t_prime - 1]

                            # roll forward one step with transition function to
                            # get probability of visiting state sprime
                            # TODO: verify that this is correct
                            visit_prob = np.sum(prev_dist *
                                                trans_mat[:, :, s_prime])

                            # combine with new dist over policy to get dist on
                            # (s,a)
                            assert np.isscalar(visit_prob)
                            prime_prob = visit_prob * policy[t_prime, s_prime,
                                                             a_prime]
                            occ_matrix[t, s, a, t_prime, s_prime,
                                       a_prime] = prime_prob
                            dt = deltas[t, s, a]
                            dt_prime = deltas[t_prime, s_prime, a_prime]
                            accumulator += prime_prob * (np.outer(
                                dt, dt_prime) + np.outer(dt_prime, dt))
    return accumulator


# def maxent_irl_md(
#         env,
#         optimiser,
#         rmodel,
#         demo_state_om,
#         linf_eps=1e-5,
#         *,
#         # inner learning rate (alpha) for GD on surrogate objective;
#         # smaller means lower L2 change in weights when computing
#         # next update
#         inner_alpha=1e-3,
#         # outer alpha for MD on main objective; smaller means lower
#         # KL change at each step
#         outer_alpha=1e-4,
#         print_interval=100):
#     """Mirror descent maxent IRL. This uses a linearised approximation gradient
#     of log likelihood of demonstrations, plus the true gradient of the KL
#     divergence between the 'current-step policy' and the 'next-step policy'
#     that we're trying to find. We iterate until the combined gradient is zero.
#     This is not very efficient, but does give us a faithful emulation of mirror
#     descent."""
#     raise NotImplementedError("this is totally broken")
#     obs_mat = env.observation_matrix
#     delta = linf_eps + 1
#     t = 0
#     rew_params = optimiser.current_params
#     rmodel.set_params(rew_params)
#     while delta > linf_eps:
#         predicted_r, out_grads = rmodel.out_grads(obs_mat)
#         _, _, pi = mce_partition_fh(env=env, R=predicted_r)
#         _, visitations = mce_occupancy_measures(env, R=predicted_r, pi=pi)
#         pol_grad = np.mean(visitations[:, None] * out_grads, axis=0)
#         expert_grad = np.mean(demo_state_om[:, None] * out_grads, axis=0)
#         if exact_fim:
#             # TODO: implement exact FIM and check numerically that it matches
#             # normal Fisher
#             raise NotImplementedError("I haven't done this yet")
#         else:
#             fim = _approximate_fisher(env, pi, pol_grad, out_grads, fim_ntraj,
#                                       fim_ident_eps)
#         grad = pol_grad - expert_grad
#         try:
#             step = np.linalg.solve(fim, grad)
#         except Exception as ex:
#             print("Exception (%s). Eigenvalues were %s and ||grad|| is %s" %
#                   (ex, np.linalg.eigvalsh(fim), np.linalg.norm(grad)))
#             raise
#         if constrained_update:
#             # TODO: do this properly so that it works even with Adam (will
#             # probably involve projecting back onto constraint set after
#             # updating; see AMSGrad docs)
#             sqrt_gg = np.sqrt(np.dot(grad, step))
#             step = step / (sqrt_gg + step_denom_eps)
#         delta = np.max(np.abs(visitations - demo_state_om))
#         if 0 == (t % print_interval):
#             print('Occupancy measure error@iter % 3d: %f (||params||=%f, '
#                   '||grad||=%f, ||step||=%f)' %
#                   (t, delta, np.linalg.norm(rew_params), np.linalg.norm(grad),
#                    np.linalg.norm(step)))
#         optimiser.step(step)
#         rew_params = optimiser.current_params
#         rmodel.set_params(rew_params)
#         t += 1
#     return optimiser.current_params, visitations

# ############################### #
# ####### REWARD MODELS ######### #
# ############################### #


class RewardModel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def out(self, inputs):
        """Get rewards for a batch of observations."""
        pass

    @abc.abstractmethod
    def grads(self, inputs):
        """Gradients of reward with respect to a batch of input observations."""
        pass

    def out_grads(self, inputs):
        """Combination method to do forward-prop AND back-prop (trivial for
        linear models, maybe some cost saving for deep model)."""
        return self.out(inputs), self.grads(inputs)

    @abc.abstractmethod
    def set_params(self, params):
        """Set a new parameter vector for the model (from flat Numpy array)."""
        pass

    @abc.abstractmethod
    def get_params(self):
        """Get current parameter vector from model (as flat Numpy array)."""
        pass


class LinearRewardModel(RewardModel):
    def __init__(self, obs_dim, *, seed=None):
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random
        self._weights = rng.randn(obs_dim, )

    def out(self, inputs):
        """Get rewards for a batch of observations."""
        assert inputs.shape[1:] == self._weights.shape
        return inputs @ self._weights

    def grads(self, inputs):
        """Individual gradient of reward with respect to each element in a
        batch of input observations."""
        assert inputs.shape[1:] == self._weights.shape
        return inputs

    def set_params(self, params):
        """Set a new parameter vector for the model (from flat Numpy array)."""
        assert params.shape == self._weights.shape
        self._weights = params

    def get_params(self):
        """Get current parameter vector from model (as flat Numpy array)."""
        return self._weights


class JaxRewardModel(RewardModel, metaclass=abc.ABCMeta):
    def __init__(self, obs_dim, *, seed=None):
        # TODO: apply jax.jit() to everything in sight
        net_init, self._net_apply = self.make_stax_model()
        if seed is None:
            # oh well
            seed = np.random.randint((1 << 63) - 1)
        rng = jrandom.PRNGKey(seed)
        out_shape, self._net_params = net_init(rng, (-1, obs_dim))
        self._net_grads = jax.grad(self._net_apply)
        # output shape should just be batch dim, nothing else
        assert out_shape == (-1,), \
            "got a weird output shape %s" % (out_shape,)

    @abc.abstractmethod
    def make_stax_model(self):
        """Build the stax model that this thing is meant to optimise. Should
        return (net_init, net_apply) pair, just like Stax modules."""
        pass

    def _flatten(self, matrix_tups):
        """Flatten everything and concatenate it together."""
        out_vecs = [v.flatten() for t in matrix_tups for v in t]
        return jnp.concatenate(out_vecs)

    def _flatten_batch(self, matrix_tups):
        """Flatten all except leading dim & concatenate results together in
        channel dim (i.e whatever the dim after the leading dim is)."""
        out_vecs = []
        for t in matrix_tups:
            for v in t:
                new_shape = (v.shape[0], )
                if len(v.shape) > 1:
                    new_shape = new_shape + (np.prod(v.shape[1:]), )
                out_vecs.append(v.reshape(new_shape))
        return jnp.concatenate(out_vecs, axis=1)

    def out(self, inputs):
        return np.asarray(self._net_apply(self._net_params, inputs))

    def grads(self, inputs):
        in_grad_partial = jax.partial(self._net_grads, self._net_params)
        grad_vmap = jax.vmap(in_grad_partial)
        rich_grads = grad_vmap(inputs)
        flat_grads = np.asarray(self._flatten_batch(rich_grads))
        assert flat_grads.ndim == 2 and flat_grads.shape[0] == inputs.shape[0]
        return flat_grads

    def set_params(self, params):
        # have to reconstitute appropriately-shaped weights from 1D param vec
        # shit this is going to be annoying
        idx_acc = 0
        new_params = []
        for t in self._net_params:
            new_t = []
            for v in t:
                new_idx_acc = idx_acc + v.size
                new_v = params[idx_acc:new_idx_acc].reshape(v.shape)
                # this seems to cast it to Jax DeviceArray appropriately;
                # surely there's better way, though?
                new_v = 0.0 * v + new_v
                new_t.append(new_v)
                idx_acc = new_idx_acc
            new_params.append(new_t)
        self._net_params = new_params

    def get_params(self):
        return self._flatten(self._net_params)


class MLPRewardModel(JaxRewardModel):
    def __init__(self, obs_dim, hiddens, activation='Tanh', **kwargs):
        assert activation in ['Tanh', 'Relu', 'Softplus'], \
            "probably can't handle activation '%s'" % activation
        self._hiddens = hiddens
        self._activation = activation
        super().__init__(obs_dim, **kwargs)

    def make_stax_model(self):
        act = getattr(jstax, self._activation)
        layers = []
        for h in self._hiddens:
            layers.extend([jstax.Dense(h), act])
        layers.extend([jstax.Dense(1), StaxSqueeze()])
        return jstax.serial(*layers)


def StaxSqueeze(axis=-1):
    def init_fun(rng, input_shape):
        ax = axis
        if ax < 0:
            ax = len(input_shape) + ax
        assert ax < len(input_shape), \
            "invalid axis %d for %d-dimensional tensor" \
            % (axis, len(input_shape))
        assert input_shape[ax] == 1, "axis %d is %d, not 1" \
            % (axis, input_shape[ax])
        output_shape = input_shape[:ax] + input_shape[ax + 1:]
        return output_shape, ()

    def apply_fun(params, inputs, **kwargs):
        return jnp.squeeze(inputs, axis=axis)

    return init_fun, apply_fun


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

    def __init__(self, rmodel, alpha=1e-3, beta1=0.9, beta2=0.99, eps=1e-8):
        # x is initial parameter vector; alpha is step size; beta1 & beta2 are
        # as defined in AMSGrad paper; eps is added to sqrt(vhat) during
        # calculation of next iterate to ensure division does not overflow.
        init_params = rmodel.get_params()
        param_size, = init_params.shape
        # first moment estimate
        self.m = np.zeros((param_size, ))
        # second moment estimate
        self.v = np.zeros((param_size, ))
        # max second moment
        self.vhat = np.zeros((param_size, ))
        # parameter estimate
        self.x = init_params
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

    def __init__(self, rmodel, alpha=1e-3):
        init_params = rmodel.get_params()
        self.x = init_params
        self.alpha = alpha
        self.cnt = 1

    def step(self, grad):
        self.x = self.x - self.alpha * grad
        return self.x

    @property
    def current_params(self):
        return self.x
