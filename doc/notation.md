# Mathematical notation

This document defines the mathematical objects, maps, and array conventions used by jaxdp. The
state and action spaces are finite throughout. Equations omit leading batch axes. A core operation
acts on one model, and `jax.vmap` composes that operation over batches.

## Models and arrays

Let $\mathcal{S}$ be a finite state space and $\mathcal{A}$ a finite action space. States are
indexed by $s,s' \in \mathcal{S}$ and actions by $a,a' \in \mathcal{A}$. An `MDP` contains the
tuple

$$
(\mathcal{S}, \mathcal{A}, P, R, \mu, \tau),
$$

where $P$ is the transition kernel, $R$ is the transition reward, $\mu$ is the initial-state
distribution, and $\tau$ is the terminal-state indicator. The discount $\gamma$ is an operator
argument rather than an `MDP` field. Write

$$
V = \mathbb{R}^{|\mathcal{S}|},
\qquad
Q = \mathbb{R}^{|\mathcal{S}|\times|\mathcal{A}|},
\qquad
\mathcal{D}_S = \mathbb{R}^{|\mathcal{S}|},
\qquad
\mathcal{D}_{SA} = \mathbb{R}^{|\mathcal{S}|\times|\mathcal{A}|}
$$

for the state-value, action-value, state-measure, and state-action-measure vector spaces. The value
and measure spaces have different semantics despite their coordinate-space isomorphisms. Let
$\Delta(\mathcal{X})$ be the probability simplex over a finite set $\mathcal{X}$ and define the
policy space as

$$
\Pi
= \left\{\pi \mid \pi : \mathcal{S} \rightarrow \Delta(\mathcal{A})\right\}.
$$

| Symbol | Meaning | jaxdp representation |
| --- | --- | --- |
| $P(s' \mid s,a)$ | Probability of successor $s'$ | `mdp.transition[..., a, s_next, s]` |
| $R(s,a,s')$ | Reward on the transition $s \xrightarrow{a} s'$ | `mdp.reward[..., a, s, s_next]` |
| $\mu(s)$ | Initial-state probability | `mdp.initial[..., s]` |
| $\tau(s)$ | Terminal indicator in $\{0,1\}$ | `mdp.terminal[..., s]` |
| $\pi(a \mid s)$ | Policy probability | `policy[..., a, s]` |
| $v(s)$ | State value | `v_val[..., s]` |
| $q(s,a)$ | Action value | `q_val[..., a, s]` |
| $\rho(s)$ | State distribution | `dist[..., s]` |
| $\xi(s,a)$ | State-action distribution | `dist[..., a, s]` |

The transition array is column stochastic in its stored state axes:

$$
\sum_{s' \in \mathcal{S}} P(s' \mid s,a) = 1.
$$

## Mathematical objects and maps

The mathematical role of an object is separate from its representation as a Python function or
namespace. jaxdp uses the following classification.

| Class | Definition | Examples |
| --- | --- | --- |
| Model | A tuple specifying a stochastic process or decision process | `MDP`, `MRP` |
| Function | An element of a finite-dimensional function space | $v \in V$, $q,r \in Q$ |
| Stochastic kernel | A conditional probability map into a simplex | $P$, $\pi$ |
| Measure | A probability or occupancy over a finite domain | $\mu$, $\rho$, $\xi$ |
| Operator | A map between function spaces or measure spaces | $\mathcal{P}$, $\mathcal{T}$ |
| Policy selector | A map whose codomain is the policy space $\Pi$ | $\mathcal{G}:Q\to\Pi$ |
| Scalar functional | A scalar-valued map | $v\mapsto\mathbb{E}_\rho[v]$ |
| Equation | A relation to be satisfied | $v=\mathcal{T}^{\pi}_{V}v$ |
| Solver | An algorithm that solves an equation | policy evaluation, value iteration |
| Transformation | A map between mathematical objects | `make_mrp`, $\mathcal{T}^n$ |

A scalar functional is an operator with scalar codomain, but its separate name makes the output
semantics explicit. A Python namespace groups related operations and introduces no additional
mathematical category.

## Terminal states

A state with $\tau(s)=1$ is absorbing and has zero reward on every outgoing transition. A
transition into a terminal state may still produce a reward. Bellman operations include that reward
and mask all later value after arrival. Define the terminal-masked kernel

$$
\bar P(s' \mid s,a) = P(s' \mid s,a)\bigl(1-\tau(s')\bigr).
$$

This convention gives every terminal state value zero without discarding rewards received upon
entering it. Truncation is an external rollout boundary: it may reset sampler state, but it does
not enter $\tau$ and retains the observed successor for bootstrapping.

## Rewards and kernel induction

The expected immediate state-action reward is

$$
r(s,a) = \sum_{s' \in \mathcal{S}} P(s' \mid s,a)R(s,a,s').
$$

This quantity has shape `(A, S)` and is computed internally by the value operations. For a policy
$\pi$,

$$
r^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s)r(s,a),
$$

and `make_mrp(mdp, policy).reward` returns it with shape `(S,)`.

Let $\mathcal{K}_{SA\to S}$ be the finite stochastic kernels from
$\mathcal{S}\times\mathcal{A}$ to $\mathcal{S}$, and let $\mathcal{K}_{S\to S}$ be the finite
Markov kernels from $\mathcal{S}$ to itself. Policy induction is the kernel transformation

$$
\mathcal{I}_{P}:
\mathcal{K}_{SA\to S}\times\Pi
\longrightarrow
\mathcal{K}_{S\to S},
$$

defined by

$$
\mathcal{I}_{P}(P,\pi)(s' \mid s)
= P^\pi(s' \mid s)
= \sum_{a \in \mathcal{A}} \pi(a \mid s)P(s' \mid s,a),
$$

while policy reward induction is the function transformation

$$
\mathcal{I}_{r}:Q\times\Pi\longrightarrow V,
\qquad
\mathcal{I}_{r}(r,\pi)(s)
= r^\pi(s)
= \sum_{a \in \mathcal{A}}\pi(a\mid s)r(s,a)
= \sum_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}}
  \pi(a \mid s)P(s' \mid s,a)R(s,a,s').
$$

Together these transformations induce an MRP:

$$
\mathcal{I}_{\mathrm{MRP}}:
\mathrm{MDP}\times\Pi\longrightarrow\mathrm{MRP}.
$$

`make_mrp(mdp, policy)` implements $\mathcal{I}_{\mathrm{MRP}}$ and retains $\mu$ and $\tau$
from its source `MDP`. Kernel induction reduces the conditioning domain from state-action pairs to
states. Subsequent transition operators use the resulting state kernel.

## Value-to-policy mappings

A value-to-policy mapping transforms a value function into a policy kernel. For action values, the
deterministic greedy mapping is

$$
\mathcal{G}:Q\longrightarrow\Pi,
\qquad
a^*(s)=\min\!\mathop{\arg\max}_{a\in\mathcal{A}}q(s,a),
\qquad
\mathcal{G}(q)(a\mid s)=\mathbf{1}\{a=a^*(s)\}.
$$

The minimum supplies the deterministic lowest-index tie rule used by `GreedyMap.q`. Softmax and
epsilon-greedy mappings are

$$
\mathcal{S}_{\eta}(q)(a\mid s)
= \frac{\exp(q(s,a)/\eta)}
       {\sum_{b\in\mathcal{A}}\exp(q(s,b)/\eta)},
\qquad \eta>0,
$$

and

$$
\mathcal{G}_{\epsilon}(q)(a\mid s)
= (1-\epsilon)\mathcal{G}(q)(a\mid s)
  + \frac{\epsilon}{|\mathcal{A}|},
\qquad 0\leq\epsilon\leq 1.
$$

`GreedyMap.q`, `SoftGreedyMap.q`, and `EpsilonGreedy.q` implement these mappings. Their `v`
methods first apply the one-step backup $\mathcal{B}_{\gamma}:V\to Q$ defined below and then apply
the corresponding mapping. A softmax mapping alone leaves the Bellman objective unchanged. A
regularized Bellman operator additionally changes the action reduction and its fixed point.

## Transition operators

The transition kernel induces a backward operator from state functions to state-action functions:

$$
\mathcal{P}:V\longrightarrow Q,
\qquad
(\mathcal{P}v)(s,a)
= \sum_{s'\in\mathcal{S}}P(s'\mid s,a)v(s').
$$

Terminal semantics induce the terminal-masked transition operator

$$
\bar{\mathcal{P}}:V\longrightarrow Q,
\qquad
(\bar{\mathcal{P}}v)(s,a)
= \sum_{s'\in\mathcal{S}}\bar P(s'\mid s,a)v(s').
$$

The policy-induced kernel similarly induces a backward Markov operator on state functions:

$$
\mathcal{P}^{\pi}:V\longrightarrow V,
\qquad
(\mathcal{P}^{\pi}v)(s)
= \sum_{s'\in\mathcal{S}}P^{\pi}(s'\mid s)v(s').
$$

Its adjoint propagates state measures forward:

$$
(\mathcal{P}^{\pi})^*:\mathcal{D}_S
\longrightarrow\mathcal{D}_S,
\qquad
\bigl((\mathcal{P}^{\pi})^*\rho\bigr)(s')
= \sum_{s\in\mathcal{S}}P^{\pi}(s'\mid s)\rho(s).
$$

The two directions are dual under the finite pairing:

$$
\langle\rho,\mathcal{P}^{\pi}v\rangle
= \langle(\mathcal{P}^{\pi})^*\rho,v\rangle.
$$

Starting from $\rho_0=\mu$, the state and state-action distributions at time $n$ are

$$
\rho_n=\bigl((\mathcal{P}^{\pi})^*\bigr)^n\mu,
\qquad
\xi_n(s,a)=\rho_n(s)\pi(a\mid s).
$$

`Occupancy(steps=n).v` and `.q` currently compute these finite-step marginals. A discounted
occupancy measure is a cumulative weighted sum of marginals and is a separate planned object. A
stationary distribution solves

$$
\rho_{\infty}=(\mathcal{P}^{\pi})^*\rho_{\infty},
\qquad
\sum_{s\in\mathcal{S}}\rho_{\infty}(s)=1.
$$

`Stationary.v` solves this equation and `Stationary.q` combines its solution with $\pi$.

## Transition and Bellman building blocks

The terminal-aware transition operators for an MRP and an MDP are

$$
(\bar{\mathcal{P}}_S x)(s)
= \sum_{s'\in\mathcal{S}}
  P_S(s'\mid s)(1-\tau(s'))x(s'),
$$

and

$$
(\bar{\mathcal{P}}_{SA}x)(s,a)
= \sum_{s'\in\mathcal{S}}
  P(s'\mid s,a)(1-\tau(s'))x(s').
$$

`TransOp.s(mrp, vec)` applies $\bar{\mathcal{P}}_S$ and returns shape `(S,)`.
`TransOp.sa(mdp, vec)` applies $\bar{\mathcal{P}}_{SA}$ and returns shape `(A, S)`.
Both accept an arbitrary state vector; terminal successors do not contribute.

Their adjoints push state and state-action measures to continuing successor states:

$$
(\bar{\mathcal{P}}_S^*\rho)(s')
=(1-\tau(s'))\sum_{s\in\mathcal{S}}P_S(s'\mid s)\rho(s),
$$

and

$$
(\bar{\mathcal{P}}_{SA}^*\xi)(s')
=(1-\tau(s'))\sum_{s,a}P(s'\mid s,a)\xi(s,a).
$$

`AdjTransOp.s(mrp, dist)` and `AdjTransOp.sa(mdp, dist)` implement these maps. They satisfy

$$
\langle \bar{\mathcal{P}}_S x,\rho\rangle
=\langle x,\bar{\mathcal{P}}_S^*\rho\rangle,
\qquad
\langle \bar{\mathcal{P}}_{SA}x,\xi\rangle
=\langle x,\bar{\mathcal{P}}_{SA}^*\xi\rangle.
$$

Their output may have mass below one because terminal successor mass is removed. The raw adjoint
$\mathcal{P}^*$ instead preserves terminal mass and is used for full distribution propagation.

The expected immediate state-action reward and one-step Bellman backup are

$$
r(s,a)=\sum_{s'\in\mathcal{S}}P(s'\mid s,a)R(s,a,s'),
$$

and

$$
\mathcal{B}_\gamma:V\longrightarrow Q,
\qquad
\mathcal{B}_\gamma v = r + \gamma\bar{\mathcal{P}}_{SA}v.
$$

Equivalently,

$$
(\mathcal{B}_\gamma v)(s,a)
= r(s,a)
+ \gamma \sum_{s' \in \mathcal{S}} \bar P(s' \mid s,a)v(s').
$$

Its domain is $\mathbb{R}^{|\mathcal{S}|}$ and its codomain is
$\mathbb{R}^{|\mathcal{S}| \times |\mathcal{A}|}$. Bellman operators and value-to-policy mappings
compose the expected reward, discount, and `TransOp.sa` directly.

Two reductions map action values back to state values. Policy reduction is

$$
(\Pi_\pi q)(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s)q(s,a),
$$

and greedy reduction is

$$
(\mathcal{M}q)(s) = \max_{a \in \mathcal{A}} q(s,a).
$$

Policy and greedy reductions remain explicit building blocks of the corresponding Bellman
operators.

## Bellman policy operators

For a fixed policy $\pi$, the Bellman operators on the state-value and action-value spaces are

$$
\mathcal{T}^{\pi}_{V} = \Pi_\pi \mathcal{B}_\gamma,
\qquad
\mathcal{T}^{\pi}_{Q} = \mathcal{B}_\gamma \Pi_\pi.
$$

Equivalently,

$$
(\mathcal{T}^{\pi}_{V}v)(s)
= \sum_{a \in \mathcal{A}} \pi(a \mid s)
  \left[r(s,a) + \gamma \sum_{s' \in \mathcal{S}}
  \bar P(s' \mid s,a)v(s')\right],
$$

and

$$
(\mathcal{T}^{\pi}_{Q}q)(s,a)
= r(s,a) + \gamma \sum_{s' \in \mathcal{S}} \bar P(s' \mid s,a)
  \sum_{a' \in \mathcal{A}} \pi(a' \mid s')q(s',a').
$$

`BellmanOp.v(mdp, policy, v_val, gamma)` applies $\mathcal{T}^{\pi}_{V}$, while
`BellmanOp.q(mdp, policy, q_val, gamma)` applies $\mathcal{T}^{\pi}_{Q}$. Each method maps its
named value space back to the same space.

## Bellman optimality operators

The Bellman optimality operators replace policy reduction with greedy reduction:

$$
\mathcal{T}^{*}_{V} = \mathcal{M}\mathcal{B}_\gamma,
\qquad
\mathcal{T}^{*}_{Q} = \mathcal{B}_\gamma\mathcal{M}.
$$

Their expanded forms are

$$
(\mathcal{T}^{*}_{V}v)(s)
= \max_{a \in \mathcal{A}}
  \left[r(s,a) + \gamma \sum_{s' \in \mathcal{S}}
  \bar P(s' \mid s,a)v(s')\right],
$$

and

$$
(\mathcal{T}^{*}_{Q}q)(s,a)
= r(s,a) + \gamma \sum_{s' \in \mathcal{S}}
  \bar P(s' \mid s,a)\max_{a' \in \mathcal{A}} q(s',a').
$$

`BellmanOptOp.v(mdp, v_val, gamma)` applies $\mathcal{T}^{*}_{V}$, while
`BellmanOptOp.q(mdp, q_val, gamma)` applies $\mathcal{T}^{*}_{Q}$.

## Smooth Bellman operators

For temperature $\tau>0$, three smooth action reductions are

$$
(\mathcal{L}_{\tau}q)(s)
= \tau\log\sum_{a\in\mathcal{A}}\exp\left(\frac{q(s,a)}{\tau}\right),
$$

$$
(\operatorname{mm}_{\tau}q)(s)
= \tau\log\left(
  \frac{1}{|\mathcal{A}|}\sum_{a\in\mathcal{A}}
  \exp\left(\frac{q(s,a)}{\tau}\right)\right),
$$

and

$$
(\operatorname{boltz}_{\tau}q)(s)
= \sum_{a\in\mathcal{A}}
  \frac{\exp(q(s,a)/\tau)}{\sum_b\exp(q(s,b)/\tau)}q(s,a).
$$

Each reduction $F_{\tau}:Q\to V$ induces state- and action-value Bellman maps

$$
\mathcal{T}^{F}_{V,\tau}=F_{\tau}\mathcal{B}_{\gamma},
\qquad
\mathcal{T}^{F}_{Q,\tau}=\mathcal{B}_{\gamma}F_{\tau}.
$$

`SoftBellmanOptOp`, `MellowmaxBellmanOptOp`, and `BoltzmannBellmanOp` apply these
compositions through their `v` and `q` methods. Soft reduction is the conjugate of negative
Shannon entropy. Mellowmax is the conjugate of KL divergence to the uniform policy and differs
from soft reduction by $\tau\log|\mathcal{A}|$. Boltzmann expectation is distinct from both: at a
fixed temperature it is not generally a sup-norm non-expansion and is therefore not named an
optimality operator. A time-varying temperature belongs to planner state rather than these fixed
components.

## Transformations of operators

Some mathematical maps take an operator as input and construct another operator. Let
$\mathcal{T}:X\to X$ be a self-map on a finite-dimensional space. Its powers and residual are

$$
\mathcal{T}\longmapsto\mathcal{T}^{n},
\qquad
\mathcal{T}\longmapsto
\operatorname{res}(\mathcal{T})=\mathcal{T}-\operatorname{Id}_{X}.
$$

When the series converges, its lambda transform is

$$
\mathcal{T}\longmapsto\mathcal{T}_{\lambda}
=(1-\lambda)\sum_{n=0}^{\infty}\lambda^n\mathcal{T}^{n+1},
\qquad 0\leq\lambda<1.
$$

Projection constructs $\operatorname{Proj}_{F}\mathcal{T}$ for a chosen approximation space
$F\subseteq X$. Composition constructs the Bellman operators above from a backup and an action
reduction. The domains and codomains of these transformations are spaces of operators.

## Finite operator landscape

The following families remain meaningful when $\mathcal{S}$ and $\mathcal{A}$ are finite. Their
presence records the mathematical landscape. Inclusion in the public jaxdp API requires a separate
review.

- **Transition and adjoint operators** act backward on functions and forward on measures. Their
  powers produce finite-step prediction and propagation.
- **Multistep and generalized return operators** include $n$-step, lambda-return, randomized
  stopping, importance-sampling, Tree-backup, and Retrace constructions. They transform a
  one-step policy operator into an alternative operator with the same fixed point under their
  stated assumptions.
- **Projected operators** compose a Bellman operator with projection onto a selected approximation
  space. They require an approximation representation and lie outside the exact 0.4 core.
- **Regularized operators** replace greedy or policy reduction with a finite-action convex
  optimization. For a regularizer $\Omega$, the regularized reduction has the form

  $$
  \mathcal{M}_{\Omega}q(s)
  = \sup_{p\in\Delta(\mathcal{A})}
    \left(\sum_{a\in\mathcal{A}}p(a)q(s,a)-\Omega(p)\right).
  $$

- **Robust operators** replace transition expectation by an extremum over a finite-state
  uncertainty set $\mathcal{U}_{s,a}$:

  $$
  (\mathcal{P}_{\mathcal{U}}v)(s,a)
  = \inf_{p\in\mathcal{U}_{s,a}}
    \sum_{s'\in\mathcal{S}}p(s')v(s').
  $$

- **Risk-sensitive operators** replace ordinary expectation with a selected finite-model risk
  functional. The risk functional and its time-consistency assumptions are part of the model.
- **Distributional operators** map a finite state-indexed or state-action-indexed family of return
  laws to another such family. Their value space differs from the scalar spaces $V$ and $Q$.
- **Finite-horizon operators** form a time-indexed sequence of backward maps rather than one
  stationary fixed-point map.
- **Average-reward and relative operators** use gain, bias, and a normalization or quotient by
  constant functions in place of discounted fixed-point semantics.
- **Optimality-preserving and gap-increasing operators** can preserve an optimal policy while
  changing nonoptimal action values or action gaps.

jaxdp 0.4 is scoped to exact discounted scalar operations on finite MDPs and MRPs. Its core consists
of kernel induction, backward and forward transition operators, action reductions, value-to-policy
mappings, Bellman policy and optimality operators, exact policy evaluation, and finite-state
measure propagation. Each additional family requires a separate mathematical and API review.

## Resolvents

For a state transition operator $\mathcal{P}_S$ and arbitrary state-space vector $x_S$, define

$$
\mathcal{R}_{S,\gamma}(\mathcal{P}_S)x_S
= (I-\gamma\mathcal{P}_S)^{-1}x_S.
$$

`Resolvent.s(p_s, vec, gamma)` applies this map. The stored matrix uses
`p_s[s_next, s]`, while the operator in the equation denotes its backward action on state vectors.

For an arbitrary state-action-space vector, the policy-induced resolvent is

$$
\mathcal{R}^{\pi}_{SA,\gamma}x_{SA}
= (I-\gamma\bar{\mathcal{P}}\Pi_\pi)^{-1}x_{SA}.
$$

`Resolvent.sa(mdp, policy, vec, gamma)` uses the equivalent identity

$$
\mathcal{R}^{\pi}_{SA,\gamma}x_{SA}
= x_{SA} + \gamma\bar{\mathcal{P}}
  \mathcal{R}_{S,\gamma}(\Pi_\pi\bar{\mathcal{P}})\Pi_\pi x_{SA},
$$

so its linear solve has shape $S\times S$ rather than $(SA)\times(SA)$.

## Policy evaluation

For $0 \leq \gamma < 1$, exact policy evaluation returns the unique fixed points

$$
v^\pi = \mathcal{T}^{\pi}_{V}v^\pi,
\qquad
q^\pi = \mathcal{T}^{\pi}_{Q}q^\pi.
$$

With the conventional row-oriented matrix
$\bar P^\pi_{s,s'} = P^\pi(s' \mid s)(1-\tau(s'))$, the state value is the unique solution of

$$
\left(I-\gamma\bar P^\pi\right)v^\pi = r^\pi.
$$

Equivalently,

$$
v^\pi
= \mathcal{R}_{S,\gamma}(\bar{\mathcal{P}}^\pi)r^\pi,
\qquad
q^\pi
= \mathcal{R}^{\pi}_{SA,\gamma}r.
$$

`PolicyEvaluation.v(mdp, policy, gamma)` and `.q` apply these resolvents to the corresponding
expected rewards.

## Value iteration

For initial values $v_0$ and $q_0$, fixed-step value iteration computes

$$
v_n=(\mathcal{T}^{*}_{V})^n v_0,
\qquad
q_n=(\mathcal{T}^{*}_{Q})^n q_0.
$$

`ValueIteration(step=n).v(mdp, v_val, gamma)` and `.q(mdp, q_val, gamma)` apply these
iterations exactly $n$ times.

## Policy iteration

Starting from $\pi_0$, exact policy iteration alternates evaluation and greedy improvement:

$$
q^{\pi_k}=\mathcal{R}^{\pi_k}_{SA,\gamma}r,
\qquad
\pi_{k+1}=\mathcal{G}(q^{\pi_k}).
$$

`PolicyIteration(step=n).policy(mdp, policy, gamma)` returns $\pi_n$. Its `.v` and `.q` methods
return the exact values of $\pi_n$.

## Expectations

Expectations always receive their distribution explicitly. For a state distribution $\rho$ and a
state-action distribution $\xi$, write $\Delta_S=\Delta(\mathcal{S})\subset\mathcal{D}_S$ and
$\Delta_{SA}=\Delta(\mathcal{S}\times\mathcal{A})\subset\mathcal{D}_{SA}$.

$$
\mathbb{E}_{\rho}[v]
= \sum_{s \in \mathcal{S}} \rho(s)v(s),
\qquad
\mathbb{E}_{\xi}[q]
= \sum_{s \in \mathcal{S}} \sum_{a \in \mathcal{A}} \xi(s,a)q(s,a).
$$

`Expectation.s(v_val, dist)` implements the first expression and
`Expectation.sa(q_val, dist)` implements the second. The initial distribution is used only when
the caller explicitly supplies `mdp.initial` as `dist`.

## Target API map

This table specifies the intended mathematical meaning of the finite public API. An operation
marked for review has no public name until its role and composition are approved.

| Mathematical object | Domain and codomain | jaxdp API |
| --- | --- | --- |
| $\mathcal{I}_{P}$ | $\mathcal{K}_{SA\to S}\times\Pi\to\mathcal{K}_{S\to S}$ | Part of `make_mrp` |
| $\mathcal{I}_{r}$ | $Q\times\Pi\to V$ | `make_mrp(...).reward` |
| $\mathcal{I}_{\mathrm{MRP}}$ | $\mathrm{MDP}\times\Pi\to\mathrm{MRP}$ | `make_mrp` |
| $\mathcal{G}$ | $Q\to\Pi$ | `GreedyMap.q` |
| $\mathcal{S}_{\eta}$ | $Q\to\Pi$ | `SoftGreedyMap.q` |
| $\mathcal{G}_{\epsilon}$ | $Q\to\Pi$ | `EpsilonGreedy.q` |
| $\mathcal{G}\mathcal{B}_{\gamma}$ | $V\to\Pi$ | `GreedyMap.v` |
| $\mathcal{S}_{\eta}\mathcal{B}_{\gamma}$ | $V\to\Pi$ | `SoftGreedyMap.v` |
| $\mathcal{G}_{\epsilon}\mathcal{B}_{\gamma}$ | $V\to\Pi$ | `EpsilonGreedy.v` |
| $\bar{\mathcal{P}}_{SA}$ | $V\to Q$ | `TransOp.sa` |
| $\bar{\mathcal{P}}_S$ | $V\to V$ | `TransOp.s` |
| $\bar{\mathcal{P}}_{SA}^*$ | $\mathcal{D}_{SA}\to\mathcal{D}_S$ | `AdjTransOp.sa` |
| $\bar{\mathcal{P}}_S^*$ | $\mathcal{D}_S\to\mathcal{D}_S$ | `AdjTransOp.s` |
| $(\mathcal{P}^{\pi})^*$ | $\mathcal{D}_S\to\mathcal{D}_S$ | Used by `Occupancy` |
| $\rho_n$ | $\mathrm{MDP}\times\Pi\times\mathbb{N}\to\Delta_S$ | `Occupancy.v` |
| $\xi_n$ | $\mathrm{MDP}\times\Pi\times\mathbb{N}\to\Delta_{SA}$ | `Occupancy.q` |
| $\rho_{\infty}$ | $\mathrm{MDP}\times\Pi\to\Delta_S$ | `Stationary.v` |
| $\xi_{\infty}$ | $\mathrm{MDP}\times\Pi\to\Delta_{SA}$ | `Stationary.q` |
| $\mathcal{B}_\gamma$ | $V \rightarrow Q$ | Expected reward plus `TransOp.sa` |
| $\mathcal{M}$ | $Q \rightarrow V$ | Used by `BellmanOptOp` |
| $\mathcal{R}_{S,\gamma}$ | $V \rightarrow V$ | `Resolvent.s` |
| $\mathcal{R}^{\pi}_{SA,\gamma}$ | $Q \rightarrow Q$ | `Resolvent.sa` |
| $\mathcal{T}^{\pi}_{V}$ | $V \rightarrow V$ | `BellmanOp.v` |
| $\mathcal{T}^{\pi}_{Q}$ | $Q \rightarrow Q$ | `BellmanOp.q` |
| $\mathcal{T}^{*}_{V}$ | $V \rightarrow V$ | `BellmanOptOp.v` |
| $\mathcal{T}^{*}_{Q}$ | $Q \rightarrow Q$ | `BellmanOptOp.q` |
| $\mathcal{T}^{\mathrm{soft}}_{V,\tau}$ | $V \rightarrow V$ | `SoftBellmanOptOp.v` |
| $\mathcal{T}^{\mathrm{soft}}_{Q,\tau}$ | $Q \rightarrow Q$ | `SoftBellmanOptOp.q` |
| $\mathcal{T}^{\mathrm{mm}}_{V,\tau}$ | $V \rightarrow V$ | `MellowmaxBellmanOptOp.v` |
| $\mathcal{T}^{\mathrm{mm}}_{Q,\tau}$ | $Q \rightarrow Q$ | `MellowmaxBellmanOptOp.q` |
| $\mathcal{T}^{\mathrm{boltz}}_{V,\tau}$ | $V \rightarrow V$ | `BoltzmannBellmanOp.v` |
| $\mathcal{T}^{\mathrm{boltz}}_{Q,\tau}$ | $Q \rightarrow Q$ | `BoltzmannBellmanOp.q` |
| $v^\pi$ | $\mathrm{MDP} \times \Pi \times [0,1) \rightarrow V$ | `PolicyEvaluation.v` |
| $q^\pi$ | $\mathrm{MDP} \times \Pi \times [0,1) \rightarrow Q$ | `PolicyEvaluation.q` |
| $(\mathcal{T}^{*}_{V})^n$ | $V \rightarrow V$ | `ValueIteration.v` |
| $(\mathcal{T}^{*}_{Q})^n$ | $Q \rightarrow Q$ | `ValueIteration.q` |
| $\pi_n$ | $\Pi \rightarrow \Pi$ | `PolicyIteration.policy` |
| $r^\pi$ | $\mathrm{MDP} \times \Pi \rightarrow V$ | `make_mrp(...).reward` |
| $r$ | $\mathrm{MDP} \rightarrow Q$ | Internal expected reward |
| $\mathbb{E}_{\rho}[v]$ | $V\times\Delta_S\to\mathbb{R}$ | `Expectation.s` |
| $\mathbb{E}_{\xi}[q]$ | $Q\times\Delta_{SA}\to\mathbb{R}$ | `Expectation.sa` |

## Literature map

- Yu, Mahmood, and Sutton develop
  [generalized Bellman operators](https://jmlr.org/papers/v19/17-283.html) through randomized
  stopping times.
- Munos et al. define
  [Retrace and related multistep return operators](https://arxiv.org/abs/1606.02647).
- Tsitsiklis and Van Roy analyze
  [projected Bellman equations](http://www.stanford.edu/~bvr/psfiles/td.pdf) for value
  approximation.
- Geist, Scherrer, and Pietquin give a unified
  [regularized MDP operator theory](https://arxiv.org/abs/1901.11275).
- Iyengar develops finite-state
  [robust dynamic programming](https://pubsonline.informs.org/doi/10.1287/moor.1040.0129).
- Bellemare, Dabney, and Munos study
  [distributional Bellman operators](https://arxiv.org/abs/1707.06887).
- Bellemare et al. characterize
  [optimality-preserving and gap-increasing operators](https://arxiv.org/abs/1512.04860).
- Lee and Ryu analyze
  [relative value iteration for average-reward MDPs](https://arxiv.org/abs/2504.09913).
