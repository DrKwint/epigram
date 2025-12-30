from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jaxtyping import Array, Float

from src.net import ENN
from src.zono import AbstractENN, GeneratorGroup, SensitivityResult, Zonotope, box_to_zonotope
from src.polytope import Polytope

MAX_STEPS = 10

class SearchStrategy:
    """
    Abstract base class for search strategies.
    Decides how to expand a node into children.
    """
    def step(self, solver: 'ReachabilitySolver', node: 'SearchNode') -> List['SearchNode']:
        """
        Produce child nodes from the current node.
        Returns a list of new nodes to add to the queue.
        If empty list, the branch is terminated (pruned or safe leaf).
        """
        raise NotImplementedError


@dataclass(frozen=True)
class ConstraintState:
    """
    Immutable recipe for a search branch.
    Stores the constraints applied to the transition that created a node.
    """
    # Action Bounds: ((min_vals...), (max_vals...))
    action_bounds: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None
    
    # Z Bounds: ((min_vals...), (max_vals...)) - Legacy/Fast Box Constraints
    z_bounds: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None
    
    # Z Polytope: Polytope(hz <= d) - Rigorous Linear Constraints
    z_polytope: Optional[Polytope] = None
    
    # ReLU Splits: Map layer_name -> ((neuron_idx, 'active'|'inactive'), ...)
    # Stored as tuple-of-tuples to be hashable
    relu_splits: Dict[str, Tuple[Tuple[int, str], ...]] = field(default_factory=dict)
    
    def probability_mass(self, z_dim: int = 1, z_range: float = 6.0) -> float:
        """
        Estimates volume of the Z-set (proxy for probability).
        Order of precedence:
        1. z_polytope (Rigorous)
        2. z_bounds (Fast)
        3. Unconstrained (1.0)
        """
        total_vol = z_range ** z_dim
        
        # 1. Polytope Volume (Approximate via Cheby Ball or Sampling?)
        # Exact volume of high-dim polytope is #P-hard.
        # Cheap proxy: Volume of Bounding Box of Polytope?
        # Or just use the Box constraints if they exist?
        
        # For now, if we have a polytope, we assume it refines the box.
        # But calculating its volume is hard.
        # Let's fallback to the z_bounds volume if available, 
        # or if z_polytope is present but no bounds, return a heuristic?
        
        # Refined Plan: 'z_bounds' should always track the Bounding Box of 'z_polytope'.
        # If we add a linear constraint, we should update the BB.
        
        if self.z_bounds is None:
            return 1.0
            
        mins, maxs = self.z_bounds
        diffs = np.array(maxs) - np.array(mins)
        
        # Clip negative volumes (infeasible)
        diffs = np.maximum(diffs, 0.0)
        
        vol = float(np.prod(diffs))
        return vol / max(total_vol, 1e-9)


@dataclass(order=True, frozen=True)
class SearchNode:
    priority: Tuple[int, float] = field(compare=True) # (-timestep, -mass)
    timestep: int = field(compare=False)
    zonotope: Zonotope = field(compare=False)
    z_zonotope: Optional[Zonotope] = field(compare=False, default=None)
    parent: Optional['SearchNode'] = field(compare=False, default=None)
    constraints: ConstraintState = field(compare=False, default_factory=ConstraintState)
    
    # We remove __post_init__ to force explicit priority calculation by the Solver.
    
class ReachabilitySolver:
    def __init__(
        self, 
        model: AbstractENN, 
        unsafe_direction: Float[Array, "out"], 
        unsafe_threshold: float, 
        target_safe_prob: Optional[float] = None, 
        verbosity: int = 2
    ):
        self.model = model
        self.unsafe_h = unsafe_direction
        self.unsafe_thresh = unsafe_threshold
        self.target_safe_prob = target_safe_prob
        self.verbosity = verbosity
        
        self.queue: List[SearchNode] = []
        self.safe_leaves: List[SearchNode] = []
        self.nodes_explored = 0
        
        # Track best node for fallback (deepest, then highest probability mass)
        self.best_node: Optional[SearchNode] = None
        
        self.root: Optional[SearchNode] = None
        self.children_map: Dict[SearchNode, List[SearchNode]] = {} # Parent -> Children
        self.split_type_map: Dict[SearchNode, str] = {} # Node -> SplitType that created its children
        
        # Default Strategy: CompletePrioritizedDFS
        self.strategy: SearchStrategy = CompletePrioritizedDFS()

    def set_strategy(self, strategy: SearchStrategy):
        self.strategy = strategy

    def push(self, node: SearchNode):
        if self.root is None and node.timestep == 0:
            self.root = node
            
        heapq.heappush(self.queue, node)
        
        # Record relationship
        if node.parent:
            if node.parent not in self.children_map:
                self.children_map[node.parent] = []
            self.children_map[node.parent].append(node)

class CompletePrioritizedDFS(SearchStrategy):
    """
    Existing rigorous strategy:
    1. Check Safety.
    2. If Safe, propagate (keeping Z-constraints).
    3. If Unsafe, perform Sensitivity Analysis and split (Action, Z, ReLU).
    4. Prioritize branches by Depth then Mass.
    """


class RRTStrategy(SearchStrategy):
    """
    Randomized exploration:
    1. Sample random concrete action.
    2. Propagate forward (keeping all Z).
    3. Check Safety:
       - If Safe: Keep all Z.
       - If Unsafe: Calculate Valid Z-Polytope (Inverse Set).
    """
    def step(self, solver: 'ReachabilitySolver', node: 'SearchNode') -> List['SearchNode']:
        # 1. Horizon Check
        if node.timestep >= solver.MAX_STEPS:
            solver.safe_leaves.append(node)
            return []
            
        # 2. Sample Random Action
        # Assuming action space is [-1, 1] per dimension for now, or use solver defaults
        act_dim = solver.model.act_dim
        key = jax.random.PRNGKey(random.randint(0, 100000))
        u_sample = jax.random.uniform(key, (act_dim,), minval=-3.0, maxval=3.0) # Using wider range to find solutions?
        
        # Create Action Constraints (Concrete Point)
        u_tuple = tuple(u_sample.tolist())
        c_action = replace(node.constraints, action_bounds=(u_tuple, u_tuple))
        
        # 3. Propagate (One Step)
        # Reuse DFS propagation logic but for a point action
        # Note: DFS propagate handles constraints. We can reuse it or call specific helpers.
        # Let's reuse a simplified version to avoid duplicating the Z-alignment logic.
        # But we need to handle the "Unsafe but Recoverable" case differently.
        
        # ...Or just call a helper that returns the NEXT Zonotope without creating the node yet?
        # Accessing private method `_propagate` from DFS class is messy.
        # We should probably expose `solver.propagate_one_step(node, action, z_poly)`?
        # For now, inline the logic:
        
        # A. Recover Z
        z_input = node.z_zonotope if node.z_zonotope else solver._get_constrained_z(node.constraints)
        
        # B. Restrict Z (if Polytope exists)
        # TODO: Implement Polytope Restriction on Zonotope if needed for accurate forward prop?
        # For now, just use the Z-bounds which encompass the polytope.
        z_constrained = z_input 
        
        # C. Action Zonotope (Point)
        u_zono = box_to_zonotope(u_sample, u_sample, source_type='action')
        
        # D. Stack
        x_aligned, u_aligned = node.zonotope.stack_independent(u_zono)
        xu_input = x_aligned.concatenate([u_aligned], axis=-1)
        
        # E. Align Z
        pad_len = xu_input.generators.shape[1] - z_constrained.generators.shape[1]
        if pad_len > 0:
            padding = jnp.zeros((z_constrained.generators.shape[0], pad_len, z_constrained.generators.shape[2]))
            new_gens = jnp.concatenate([z_constrained.generators, padding], axis=1)
            z_final = Zonotope(z_constrained.center, new_gens, xu_input.history)
        else:
            z_final = Zonotope(z_constrained.center, z_constrained.generators, xu_input.history)
            
        # F. Forward
        # RRT doesn't split ReLUs? Or does it?
        # Standard RRT is for concrete systems.
        # For Robust RRT, we propagate the bundle.
        # Minimal splitting? Let's assume no ReLU splits for now (DeepPoly default).
        next_zonos = solver.model(xu_input, z_final, split_idxs={})
        next_zono = next_zonos[0] # Should be 1 if no splits
        
        # 4. Check Safety & Inverse Constraint
        _, ub = next_zono.project_bounds(solver.unsafe_h)
        if float(jnp.max(ub)) <= solver.unsafe_thresh:
            # Safe!
            priority = (-(node.timestep + 1), -node.constraints.probability_mass(solver.model.z_dim))
            return [SearchNode(
                priority=priority,
                timestep=node.timestep + 1,
                zonotope=next_zono,
                z_zonotope=z_constrained, # Keep same Z
                parent=node,
                constraints=c_action
            )]
        else:
            # Unsafe! Can we restrict Z?
            # Constraint: h^T x <= thresh
            # Linearize Z dependence: x(z) ~= c + G z_gens
            # This is hard because x depends on z non-linearly through ReLUs?
            # AbstractENN preserves linearity w.r.t Z generators?
            # Yes, Zonotope structure preserves affine dependence on epsilon.
            # x = center + Sum(gens * eps)
            # Some eps correspond to Z.
            
            # Project constraint onto Z-space generators.
            # h^T (c + G_z * eps_z + G_other * eps_other) <= thresh
            # Worst case "other" noise?
            # To be rigorous: h^T G_z * eps_z <= thresh - h^T c - max(h^T G_other eps_other)
            
            # 1. Separate generators
            # We need to know which generators map to Z dimensions.
            # This requires tracking "GeneratorGroup" closely.
            # Using `flatten()` or similar?
            # `project_gens` method on Zonotope?
            
            # Let's assume we can get linear constraint on eps_z.
            # For now, return empty list (Killing branch) 
            # UNLESS we implement the Inverse Set calculation fully.
            # Given the complexity, maybe just "Pure RRT" that kills unsafe branches?
            # "Probabilistic RRT": If unsafe, kill.
            # The "Robust RRT" with inverse sets is Advanced.
            
            # For this step, let's implement the Pure RRT (Reject unsafe).
            # Future: Inverse Set.
            return []



    def compute_safe_probability(self, root_node: SearchNode) -> float:
        """
        Iterative MiniMax Aggregation (Post-Order).
        Calculates the lower bound of safe probability for the subtree rooted at 'root_node'.
        """
        if root_node is None:
            return 0.0
            
        # 1. Collect all nodes in subtree via DFS
        traversal_stack = [root_node]
        all_nodes = []
        visited = set()
        
        while traversal_stack:
            curr = traversal_stack.pop()
            if curr in visited:
                continue
            visited.add(curr)
            all_nodes.append(curr)
            
            children = self.children_map.get(curr, [])
            for c in children:
                traversal_stack.append(c)
                
        # 2. Iterate in reverse (Bottom-Up)
        node_values = {}
        
        for node in reversed(all_nodes):
            # Base Case: Safe Leaf
            if node in self.safe_leaves:
                node_values[node] = node.constraints.probability_mass(self.model.z_dim)
                continue
                
            children = self.children_map.get(node, [])
            if not children:
                # Dead end / Pruned / Unsafe Leaf
                node_values[node] = 0.0
                continue
                
            # Recursive Step
            child_vals = [node_values.get(c, 0.0) for c in children]
            
            split_type = self.split_type_map.get(node, 'unknown')
            
            if split_type == 'action':
                # Maximize over Action choice
                val = max(child_vals)
            else:
                # Sum over Z / ReLU / Unknown (Probability Mass)
                val = sum(child_vals)
                
            node_values[node] = val
            
        return node_values.get(root_node, 0.0)

    def get_best_action(self, default_action: Array) -> Tuple[Array, float, bool]:
        """
        Returns (action, total_safe_mass_lower_bound, is_verified_safe)
        """
        if self.root is None:
             return default_action, 0.0, False
             
        # 1. Compute Global Safe Probability Lower Bound
        total_safe_mass = self.compute_safe_probability(self.root)
        
        # 2. Find Action by tracing the max-mass path for Action splits
        curr = self.root
        
        # We generally assume the "Action" decision happens at t=0 (before first propagation)
        # But our SearchTree might split Action at any refinement step of the Root.
        while curr and curr.timestep == 0:
            children = self.children_map.get(curr, [])
            if not children:
                break
                
            split_type = self.split_type_map.get(curr, 'unknown')
            
            if split_type == 'action':
                # Pick child with highest safe probability (ArgMax behavior)
                best_child = max(children, key=self.compute_safe_probability)
                curr = best_child
            elif split_type in ['z', 'relu', 'input']:
                # The action constraints are shared/consistent across these splits.
                # We can pick any valid child to continue tracing constraints.
                valid_children = [c for c in children if self.compute_safe_probability(c) > 0]
                if not valid_children:
                    break
                # Heuristic: Follow largest mass chunk?
                curr = valid_children[0]
            else:
                # Transition (t=0 -> t=1)
                curr = children[0]
                
        c = curr.constraints
        if c.action_bounds:
            mins, maxs = c.action_bounds
            # Center of the interval
            action = (np.array(mins) + np.array(maxs)) / 2.0
            return jnp.array(action), total_safe_mass, (total_safe_mass > 0.0)
        else:
            return default_action, total_safe_mass, (total_safe_mass > 0.0)

    def get_best_trajectory(self) -> List[Dict]:
        """
        Returns a list of steps in the trajectory from t=0 to safety.
        """
        target_node = None
        if self.safe_leaves:
            # Pick leaf with highest mass? Or just any safe leaf?
            # Ideally, the leaf that contributes to the 'best action' path.
            # But just returning *a* safe path is fine for viz.
            sorted_leaves = sorted(self.safe_leaves, key=lambda n: n.constraints.probability_mass(self.model.z_dim), reverse=True)
            target_node = sorted_leaves[0]
        elif self.best_node is not None:
            target_node = self.best_node
            
        if target_node is None:
            return []
            
        # Backtrack
        path = []
        curr = target_node
        while curr is not None:
            if curr.timestep > 0: 
                c = curr.constraints
                action_info = {'timestep': curr.timestep - 1, 'action': None, 'bounds': None}
                
                if c.action_bounds:
                    mins, maxs = c.action_bounds
                    a_val = (np.array(mins) + np.array(maxs)) / 2.0
                    action_info['action'] = a_val
                    action_info['bounds'] = (mins, maxs)
                else:
                    action_info['action'] = np.array([0.0])
                    
                path.append(action_info)
                
            curr = curr.parent
            
        return list(reversed(path))

    def _generate_action_splits(self, node: SearchNode, result: SensitivityResult) -> List[ConstraintState]:
        current_c = node.constraints
        u_zono = self._get_constrained_action(current_c)
        lb, ub = u_zono.concrete_bounds()
        
        dim_idx = result.meta.get('local_index', 0)
        
        # Calculate optimal split to fix 'w' impact
        global_gen_idx = result.indices[0] + dim_idx
        proj_gens = jnp.dot(node.zonotope.generators, self.unsafe_h) 
        w_vec = proj_gens[:, global_gen_idx]
        w = float(w_vec[0])
        
        _, current_ub = node.zonotope.project_bounds(self.unsafe_h)
        current_safe_gap = float(jnp.max(current_ub)) - self.unsafe_thresh
        
        split_val = None
        SAFE_MARGIN = 1e-5
        target_reduction = current_safe_gap + SAFE_MARGIN
        
        if target_reduction < abs(w):
            # Analytical Split
            if w > 0:
                s_limit = 1.0 - target_reduction / w
                # Interval [-1, s_limit] is safe
                split_frac = s_limit if s_limit > -1.0 else 0.0
            else: # w < 0
                s_limit = -1.0 + target_reduction / abs(w)
                # Interval [s_limit, 1] is safe
                split_frac = s_limit if s_limit < 1.0 else 0.0
            
            # Clip
            split_frac = max(-0.95, min(0.95, split_frac))
            
            # Map back to concrete
            c_min = float(lb[0, dim_idx])
            c_max = float(ub[0, dim_idx])
            split_val = c_min + (c_max - c_min) * (split_frac + 1.0) / 2.0
            # print(f"    [Calculated Split] Gap={current_safe_gap:.4f}, w={w:.4f} -> SplitFrac={split_frac:.2f}")
        else:
            # Fallback Bisection
            split_val = (float(lb[0, dim_idx]) + float(ub[0, dim_idx])) / 2.0

        results = []
        for (start, end) in [(lb, ub.at[:, dim_idx].set(split_val)), 
                             (lb.at[:, dim_idx].set(split_val), ub)]:
            
            min_t = tuple(start.flatten().tolist())
            max_t = tuple(end.flatten().tolist())
            results.append(replace(current_c, action_bounds=(min_t, max_t)))
            
        return results

    def _generate_z_splits(self, current_c: ConstraintState, result: SensitivityResult) -> List[ConstraintState]:
        z_zono = self._get_constrained_z(current_c)
        lb, ub = z_zono.concrete_bounds()
        
        dim_idx = result.meta.get('local_index', 0)
        mid = (float(lb[0, dim_idx]) + float(ub[0, dim_idx])) / 2.0
        
        results = []
        for (start, end) in [(lb, ub.at[:, dim_idx].set(mid)), 
                             (lb.at[:, dim_idx].set(mid), ub)]:
            
            min_t = tuple(start.flatten().tolist())
            max_t = tuple(end.flatten().tolist())
            results.append(replace(current_c, z_bounds=(min_t, max_t)))
            
        return results

    def _generate_relu_splits(self, current_c: ConstraintState, result: SensitivityResult) -> List[ConstraintState]:
        layer = result.meta.get('layer', 'unknown')
        idx = result.meta.get('index', -1)
        
        if layer == 'unknown' or idx == -1:
            return []

        curr_layer_splits = current_c.relu_splits.get(layer, ())
        existing_indices = {s[0] for s in curr_layer_splits}
        
        if idx in existing_indices:
            return []

        results = []
        for status in ['active', 'inactive']:
            new_splits = curr_layer_splits + ((idx, status),)
            new_map = current_c.relu_splits.copy()
            new_map[layer] = new_splits
            results.append(replace(current_c, relu_splits=new_map))
            
        return results

    # --- Helpers ---

    def _get_constrained_action(self, c: ConstraintState) -> Zonotope:
        # Default Action Bounds [1 dimension]
        default_min = jnp.array([-3.0]) 
        default_max = jnp.array([3.0])
        
        if c.action_bounds:
            mins, maxs = c.action_bounds
            u_min = jnp.array(mins)
            u_max = jnp.array(maxs)
        else:
            u_min, u_max = default_min, default_max
            
        return box_to_zonotope(u_min, u_max, source_type='action')

    def _get_constrained_z(self, c: ConstraintState) -> Zonotope:
        # Initial Z Bounds (from model dim)
        z_dim = self.model.z_dim
        default_min = jnp.full((z_dim,), -3.0)
        default_max = jnp.full((z_dim,), 3.0)
        
        if c.z_bounds:
            mins, maxs = c.z_bounds
            z_min = jnp.array(mins)
            z_max = jnp.array(maxs)
        else:
            z_min, z_max = default_min, default_max
            
        return box_to_zonotope(z_min, z_max, source_type='z')