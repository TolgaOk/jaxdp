"""Exact dynamic programming for finite Markov decision processes in JAX."""

from jaxdp import mapping as mapping
from jaxdp import operator as operator
from jaxdp import planning as planning
from jaxdp.mapping import (
    EpsilonGreedy,
    Expectation,
    GreedyMap,
    Occupancy,
    SoftGreedyMap,
    Stationary,
    eigenvalues,
)
from jaxdp.mdp import MDP, MRP, make_mrp
from jaxdp.operator import (
    AdjTransOp,
    BellmanOp,
    BellmanOptOp,
    BoltzmannBellmanOp,
    MellowmaxBellmanOptOp,
    Resolvent,
    SoftBellmanOptOp,
    TransOp,
)
from jaxdp.planning import PolicyEvaluation, PolicyIteration, ValueIteration

__all__ = [
    "mapping",
    "operator",
    "planning",
    "MDP",
    "MRP",
    "make_mrp",
    "TransOp",
    "AdjTransOp",
    "Resolvent",
    "BellmanOp",
    "BellmanOptOp",
    "SoftBellmanOptOp",
    "MellowmaxBellmanOptOp",
    "BoltzmannBellmanOp",
    "GreedyMap",
    "SoftGreedyMap",
    "EpsilonGreedy",
    "PolicyEvaluation",
    "ValueIteration",
    "PolicyIteration",
    "Expectation",
    "Occupancy",
    "Stationary",
    "eigenvalues",
]
