# jaxdp mathematics

## Naming and notation

Let \(\mathcal{S}\) and \(\mathcal{A}\) be finite state and action spaces. Core operations act on
one model; leading batch axes are introduced with `jax.vmap`.

| Mathematics | Code | Meaning and shape |
| --- | --- | --- |
| \(P(s'\mid s,a)\) | `mdp.transition[a, s_next, s]` | Transition kernel, `(A, S, S)` |
| \(R(s,a,s')\) | `mdp.reward[a, s, s_next]` | Transition reward, `(A, S, S)` |
| \(\mu(s)\) | `mdp.initial[s]` | Initial state distribution, `(S,)` |
| \(\tau(s)\) | `mdp.terminal[s]` | Terminal-state indicator, `(S,)` |
| \(\pi(a\mid s)\) | `policy[a, s]` | Policy, `(A, S)` |
| \(P_S(s'\mid s)\) | `p_s[s_next, s]` | State transition operator, `(S, S)` |
| \(\gamma\) | `gamma` | Scalar discount in `[0, 1)` |
| \(v\in\mathbb{R}^{\lvert\mathcal{S}\rvert}\) | `v_val[s]` | State values, `(S,)` |
| \(q\in\mathbb{R}^{\lvert\mathcal{S}\rvert\times\lvert\mathcal{A}\rvert}\) | `q_val[a, s]` | Action values, `(A, S)` |
| \(\rho(s)\) | `dist[s]` | State distribution, `(S,)` |
| \(\xi(s,a)\) | `dist[a, s]` | State-action distribution, `(A, S)` |
| \(x_S,x_{SA}\) | `vec` | Generic operator input in `(S,)` or `(A, S)` |
| \((\mathrm{MDP},\pi)\mapsto\mathrm{MRP}\) | `make_mrp` | Policy-induced reward process |
| \(\bar{\mathcal{P}}_Sx_S\) | `TransOp.s` | Terminal-aware MRP transition, `(S,)` |
| \(\bar{\mathcal{P}}_{SA}x_S\) | `TransOp.sa` | Terminal-aware MDP transition, `(A, S)` |
| \(\bar{\mathcal{P}}_S^*x_S\) | `AdjTransOp.s` | Continuing MRP successor mass, `(S,)` |
| \(\bar{\mathcal{P}}_{SA}^*x_{SA}\) | `AdjTransOp.sa` | Continuing MDP successor mass, `(S,)` |
| \((I-\gamma\mathcal{P}_S)^{-1}\) | `Resolvent.s` | State-space resolvent |
| \((I-\gamma\mathcal{P}^{\pi}_{SA})^{-1}\) | `Resolvent.sa` | State-action resolvent via an `S`-sized solve |
| \(\mathcal{T}^{\pi}_{V},\mathcal{T}^{\pi}_{Q}\) | `BellmanOp.v`, `BellmanOp.q` | Bellman policy operators |
| \(\mathcal{T}^{*}_{V},\mathcal{T}^{*}_{Q}\) | `BellmanOptOp.v`, `BellmanOptOp.q` | Bellman optimality operators |
| \(\mathcal{T}^{\mathrm{soft}}_{V,\tau},\mathcal{T}^{\mathrm{soft}}_{Q,\tau}\) | `SoftBellmanOptOp.v`, `.q` | Entropy-regularized Bellman operators |
| \(\mathcal{T}^{\mathrm{mm}}_{V,\tau},\mathcal{T}^{\mathrm{mm}}_{Q,\tau}\) | `MellowmaxBellmanOptOp.v`, `.q` | KL-regularized Bellman operators |
| \(\mathcal{T}^{\mathrm{boltz}}_{V,\tau},\mathcal{T}^{\mathrm{boltz}}_{Q,\tau}\) | `BoltzmannBellmanOp.v`, `.q` | Boltzmann-expectation Bellman operators |
| \(\mathcal{G},\mathcal{S}_{\eta},\mathcal{G}_{\epsilon}\) | `Greedy`, `Soft`, `EpsilonGreedy` | Value-to-policy selectors |
| \(\mathbb{E}_{\rho}[v],\mathbb{E}_{\xi}[q]\) | `Expectation.s`, `Expectation.sa` | Scalar expectations |
| \(\rho_n,\xi_n\) | `Occupancy.v`, `Occupancy.q` | Finite-step distributions |
| \(\rho_\infty,\xi_\infty\) | `Stationary.v`, `Stationary.q` | Invariant distributions |
| \(v^{\pi},q^{\pi}\) | `PolicyEvaluation.v`, `PolicyEvaluation.q` | Exact policy values |
| \((\mathcal{T}^{*}_{V})^n,(\mathcal{T}^{*}_{Q})^n\) | `ValueIteration.v`, `ValueIteration.q` | Fixed-step value iteration |
| \(\pi_{k+1}=\mathcal{G}(q^{\pi_k})\) | `PolicyIteration.policy` | Fixed-step policy iteration |
