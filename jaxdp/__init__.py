"""Exact dynamic programming for finite Markov decision processes in JAX."""

from jaxdp import mapping as mapping
from jaxdp import operator as operator
from jaxdp import planning as planning
from jaxdp.mapping import (
    EpsilonGreedy,
    MellowMax,
    Occupancy,
    SoftGreedyMap,
    eigenvalues,
    expectation,
    greedy_map,
    proj_simplex,
    reward,
    stationary,
)
from jaxdp.mdp import MDP, MRP, make_mrp
from jaxdp.operator import (
    BoltzmannBellmanOp,
    MellowMaxBellmanOptOp,
    SoftBellmanOptOp,
    adj_trans_op,
    bellman_op,
    bellman_opt_op,
    resolvent,
    trans_op,
)
from jaxdp.planning import (
    AnchoredQValueIteration,
    AnchoredValueIteration,
    MomentumValueIteration,
    PolicyIteration,
    QValueIteration,
    RankOneValueIteration,
    SafeAcceleratedValueIteration,
    ValueIteration,
    policy_eval,
)

__all__ = [
    "mapping",
    "operator",
    "planning",
    "MDP",
    "MRP",
    "make_mrp",
    "trans_op",
    "adj_trans_op",
    "resolvent",
    "bellman_op",
    "bellman_opt_op",
    "SoftBellmanOptOp",
    "MellowMaxBellmanOptOp",
    "BoltzmannBellmanOp",
    "greedy_map",
    "SoftGreedyMap",
    "EpsilonGreedy",
    "proj_simplex",
    "MellowMax",
    "reward",
    "policy_eval",
    "ValueIteration",
    "QValueIteration",
    "AnchoredValueIteration",
    "AnchoredQValueIteration",
    "RankOneValueIteration",
    "SafeAcceleratedValueIteration",
    "MomentumValueIteration",
    "PolicyIteration",
    "expectation",
    "Occupancy",
    "stationary",
    "eigenvalues",
]
