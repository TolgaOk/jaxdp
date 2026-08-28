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
| \(\bar{\mathcal{P}}_Sx_S\) | `trans_op.s` | Terminal-aware MRP transition, `(S,)` |
| \(\bar{\mathcal{P}}_{SA}x_S\) | `trans_op.sa` | Terminal-aware MDP transition, `(A, S)` |
| \(\bar{\mathcal{P}}_S^*x_S\) | `adj_trans_op.s` | Continuing MRP successor mass, `(S,)` |
| \(\bar{\mathcal{P}}_{SA}^*x_{SA}\) | `adj_trans_op.sa` | Continuing MDP successor mass, `(S,)` |
| \((I-\gamma\mathcal{P}_S)^{-1}\) | `resolvent.s` | State-space resolvent |
| \((I-\gamma\mathcal{P}^{\pi}_{SA})^{-1}\) | `resolvent.sa` | State-action resolvent via an `S`-sized solve |
| \(\mathcal{T}^{\pi}_{V},\mathcal{T}^{\pi}_{Q}\) | `bellman_op.v`, `bellman_op.q` | Bellman policy operators |
| \(\mathcal{T}^{*}_{V},\mathcal{T}^{*}_{Q}\) | `bellman_opt_op.v`, `bellman_opt_op.q` | Bellman optimality operators |
| \(\mathcal{T}^{\mathrm{soft}}_{V,\tau},\mathcal{T}^{\mathrm{soft}}_{Q,\tau}\) | `SoftBellmanOptOp.v`, `.q` | Entropy-regularized Bellman operators |
| \(\mathcal{T}^{\mathrm{mm}}_{V,\tau},\mathcal{T}^{\mathrm{mm}}_{Q,\tau}\) | `MellowMaxBellmanOptOp.v`, `.q` | KL-regularized Bellman operators |
| \(\mathcal{T}^{\mathrm{boltz}}_{V,\tau},\mathcal{T}^{\mathrm{boltz}}_{Q,\tau}\) | `BoltzmannBellmanOp.v`, `.q` | Boltzmann-expectation Bellman operators |
| \(\mathcal{G},\mathcal{S}_{\eta},\mathcal{G}_{\epsilon}\) | `greedy_map`, `SoftGreedyMap`, `EpsilonGreedy` | Value-to-policy mappings |
| \(\operatorname{proj}_{\Delta_A}:Q\to\Pi\) | `proj_simplex.q` | Euclidean action-simplex projection |
| \(\operatorname{mm}_{\tau}:Q\to V\) | `MellowMax.q` | Normalized log-mean-exp reduction |
| \(r(s,a),r^{\pi}(s)\) | `reward.sa`, `reward.s` | Expected immediate rewards |
| \(\mathbb{E}_{\rho}[v],\mathbb{E}_{\xi}[q]\) | `expectation.s`, `expectation.sa` | Scalar expectations |
| \(d_k^{\gamma},\xi_k^{\gamma}\) | `Occupancy.v`, `Occupancy.q` | Normalized discounted occupancies |
| \(\rho_\infty,\xi_\infty\) | `stationary.v`, `stationary.q` | Invariant distributions |
| \(v^{\pi},q^{\pi}\) | `policy_eval.v`, `policy_eval.q` | Exact policy values |
| \(v_{k+1}=\mathcal{T}^{*}_{V}v_k\) | `ValueIteration.update` | One state-value iteration update |
| \(q_{k+1}=\mathcal{T}^{*}_{Q}q_k\) | `QValueIteration.update` | One action-value iteration update |
| \(\pi_{k+1}=\mathcal{G}\mathcal{B}_{\gamma}v^{\pi_k}\) | `PolicyIteration.update` | One exact policy iteration update |
