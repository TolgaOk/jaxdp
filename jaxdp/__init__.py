from typing import Union, Tuple, Optional
import jax
from jaxtyping import Float, Array
from jaxdp.mdp.mdp import MDP, flatten_mdp, unflatten_mdp


jax.tree_util.register_pytree_node(MDP, flatten_mdp, unflatten_mdp)
