"""Exact dynamic programming for finite Markov decision processes in JAX."""

from jaxdp import distribution as distribution
from jaxdp import operator as operator
from jaxdp import planning as planning
from jaxdp import policy as policy
from jaxdp.distribution import Expectation, Occupancy, Stationary, eigenvalues
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
from jaxdp.policy import EpsilonGreedy, Greedy, Soft

__all__ = [
    "distribution",
    "operator",
    "planning",
    "policy",
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
    "Greedy",
    "Soft",
    "EpsilonGreedy",
    "PolicyEvaluation",
    "ValueIteration",
    "PolicyIteration",
    "Expectation",
    "Occupancy",
    "Stationary",
    "eigenvalues",
]
