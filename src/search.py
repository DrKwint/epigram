import heapq
from typing import List, Optional, Tuple, Union, Dict
from dataclasses import dataclass, replace, field
import jax
import jax.numpy as jnp
import numpy as np
import random
from flax import nnx
from src.zono import GeneratorGroup, SensitivityResult, Zonotope, box_to_zonotope, AbstractENN

MAX_STEPS = 10


@dataclass(frozen=True)
class ConstraintState:
    """
    Immutable recipe for a search branch.
    Stores the constraints applied to the transition that created a node.
    """
    # Action Bounds: ((min_vals...), (max_vals...))
    action_bounds: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None
    
    # Z Bounds: ((min_vals...), (max_vals...))
    z_bounds: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None
    
    # ReLU Splits: Map layer_name -> ((neuron_idx, 'active'|'inactive'), ...)
    # Stored as tuple-of-tuples to be hashable
    relu_splits: Dict[str, Tuple[Tuple[int, str], ...]] = field(default_factory=dict)

    @property
    def probability_mass(self) -> float:
        """Estimates volume of the Z-box (proxy for probability)."""
        if self.z_bounds is None: return 1.0
        mins, maxs = self.z_bounds
        # Simple volume calculation
        diffs = jnp.array(maxs) - jnp.array(mins)
        vol = float(jnp.prod(diffs))
        # Total volume for 4 dims [-3, 3] is 6^4 = 1296.0
        return vol / 1296.0

@dataclass(order=True, frozen=True)
class SearchNode:
    priority: Tuple[int, float] # (-timestep, -mass) -> Deepest then Highest Mass
    timestep: int = field(compare=False)
    zonotope: Zonotope = field(compare=False)
    z_zonotope: Optional[Zonotope] = field(compare=False, default=None)
    parent: Optional['SearchNode'] = field(compare=False, default=None)
    constraints: ConstraintState = field(compare=False, default_factory=ConstraintState)
    
    def __post_init__(self):
        # Auto-calculate priority if missing.
        # Prioritized DFS: Deepest first (-timestep), then highest mass (-mass)
        # We use a Min-Heap, so negative values ensure Max behavior.
        if isinstance(self.priority, float) and self.priority == 0.0:
            object.__setattr__(self, 'priority', (-self.timestep, -self.constraints.probability_mass))

class ReachabilitySolver:
    def __init__(self, model: AbstractENN, unsafe_direction: jax.Array, unsafe_threshold: float, target_safe_prob: Optional[float] = None, verbosity: int = 2):
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
        
        # Track explored nodes for graph reconstruction (if needed)
        # Map: node_id -> node? No, just keep parent pointers.
        # But for "MiniMax" aggregation, we need to traverse the tree from Root down?
        # Or Root up? 
        # Actually, SearchNode object is the state. If we keep the Root, we can traverse it.
        self.root: Optional[SearchNode] = None
        self.children_map: Dict[SearchNode, List[SearchNode]] = {} # Parent -> Children
        self.split_type_map: Dict[SearchNode, str] = {} # Node -> SplitType that created its children

    def push(self, node: SearchNode):
        if self.root is None and node.timestep == 0:
            self.root = node
            
        heapq.heappush(self.queue, node)
        
        # Record relationship
        if node.parent:
            if node.parent not in self.children_map:
                self.children_map[node.parent] = []
            self.children_map[node.parent].append(node)

    def is_safe(self, zono: Zonotope) -> bool:
        """
        Checks if zonotope intersects the unsafe set (h^T x > d).
        Returns True if the Upper Bound of projection is <= threshold.
        """
        _, ub = zono.project_bounds(self.unsafe_h)
        # We take max over batch (though usually batch=1 in search)
        max_ub = jnp.max(ub)
        return float(max_ub) <= self.unsafe_thresh

    def step(self) -> bool:
        """
        Executes one iteration of the BFS loop.
        Returns False if queue is empty (done).
        """
        if not self.queue:
            return False

        current_node = heapq.heappop(self.queue)
        
        self.nodes_explored += 1
        
        # Track Best Node
        if self.best_node is None:
            self.best_node = current_node
        else:
            # Criteria: Deeper is better. If same depth, higher PROB MASS is better.
            if current_node.timestep > self.best_node.timestep:
                self.best_node = current_node
            elif current_node.timestep == self.best_node.timestep:
                if current_node.constraints.probability_mass > self.best_node.constraints.probability_mass:
                    self.best_node = current_node
        
        if self.nodes_explored % 10 == 0 and self.verbosity >= 2:
            print(f"Step {self.nodes_explored}: Queue Size {len(self.queue)}, Safe Leaves {len(self.safe_leaves)}")
            
        # 1. Horizon Check
        if current_node.timestep >= MAX_STEPS:
            self.safe_leaves.append(current_node)
            # Do NOT return True (early stop). Continue searching to accumulate mass.
            # But stop expanding this branch.
            return True
            
            # Early Stopping Check (removed for Exhaustive Search)
            # if self.target_safe_prob is not None: ...
            
            return True

        # 2. Safety Check
        is_safe_val = self.is_safe(current_node.zonotope)
        # print(f"  [Trace] Step {self.nodes_explored}: Safe={is_safe_val} t={current_node.timestep}")
        
        if is_safe_val:
            # Node is Safe -> Propagate forward to t+1
            # We start with FRESH constraints for the next step (unless we want to inherit global ones)
            # For this algo, we assume Action/ReLU constraints are local to the step, 
            # but Z constraints are global.
            
            # Create next step constraints inheriting ONLY Z
            next_constraints = ConstraintState(z_bounds=current_node.constraints.z_bounds)
            self._propagate(current_node, next_constraints)
        else:
            # Node is Unsafe -> Refine the transition that created it
            # print(f"  [Unsafe] Node at t={current_node.timestep}. Refining...")
            self._refine(current_node)
            
        return True

    def _propagate(self, parent_node: SearchNode, constraints: ConstraintState):
        """
        Runs the model forward one step using specific constraints.
        [Parent State] + [Constrained Action] + [Constrained Z] -> [Child State]
        """
        # 1. Recover Z Zonotope (Threaded)
        if parent_node.z_zonotope is not None:
             z_input = parent_node.z_zonotope
        else:
             # Fallback/Root initialization (assumed unconstrained/default if not set)
             z_input = self._get_constrained_z(constraints)

        # 2. Apply Z Constraints via Generator Restriction
        # If the constraints define tighter Z bounds than the parent's Z was built with,
        # we must restrict the generators of BOTH x (state) and z (input).
        
        # We assume Z is initially a Box centered at 0 or whatever _get_constrained_z gave at Root.
        # To map absolute bounds [a, b] to relative [-1, 1], we need the BASELINE Z props.
        # Ideally, we should store z_baseline in the solver or node.
        # Approximation: We assume z_input matches the *parent's* constraints.
        # We need to apply the DELTA from parent constraints to current constraints.
        
        restrictions = []
        if constraints.z_bounds and parent_node.constraints.z_bounds:
            p_mins, p_maxs = parent_node.constraints.z_bounds
            c_mins, c_maxs = constraints.z_bounds
            
            for i in range(len(p_mins)):
                # If bounds tightened
                if c_mins[i] > p_mins[i] or c_maxs[i] < p_maxs[i]:
                    # Find the generator index for Z dimension i.
                    # We assume Z history is simple: [GeneratorGroup('z', start, count)]
                    # We search z_input history
                    gen_idx = -1
                    base_radius = 1.0 
                    base_center = 0.0
                    
                    found = False
                    for g in z_input.history:
                        if g.source_type == 'z':
                            # Assuming Z dims map 1-to-1 to generators in order
                            if 0 <= i < g.count:
                                gen_idx = g.start_idx + i
                                
                                # Retrieve current geometry of this generator from z_input
                                # Center[i] ?? No, z_input center is [Batch, Dim]
                                # But generators are [Batch, N, Dim]
                                # For a Box Z, Z_i = c_i + g_idx_i * eps
                                # We need (new_min - current_center)/current_radius
                                
                                # Current projected bounds of this dimension:
                                # We can't easily invert the zonotope.
                                # BUT we know Z is a box source.
                                # The generator logic: restrict_generators takes [-1, 1] relative fractions.
                                # So we need to know what [-1, 1] corresponds to in PROPERTY space.
                                # Wait, if z_input has already been restricted, its "[-1, 1]" is a subset of the original.
                                # "restrict_generators" restricts the *current* generator (which defines the current [-1,1] range)
                                # to a subset.
                                
                                # Relative split:
                                # New interval [L_new, R_new] is sub-segment of [L_old, R_old].
                                # We map [L_old, R_old] -> [-1, 1].
                                # frac_min = -1 + 2 * (c_min - p_min) / (p_max - p_min)
                                # frac_max = -1 + 2 * (c_max - p_min) / (p_max - p_min)
                                
                                width = p_maxs[i] - p_mins[i]
                                if width > 1e-9:
                                    frac_min = -1.0 + 2.0 * (c_mins[i] - p_mins[i]) / width
                                    frac_max = -1.0 + 2.0 * (c_maxs[i] - p_mins[i]) / width
                                    
                                    # Clip for safety
                                    frac_min = max(-1.0, frac_min)
                                    frac_max = min(1.0, frac_max)
                                    
                                    restrictions.append((gen_idx, frac_min, frac_max))
                                found = True
                                break
                    if not found:
                        print(f"Warning: Could not find generator for Z dim {i}")

        # Apply restrictions
        if restrictions:
             # Apply to Z (Input)
             z_constrained = z_input.restrict_generators(restrictions)
             # Apply to X (State) - x shares the same generator indices!
             x_constrained = parent_node.zonotope.restrict_generators(restrictions)
        else:
             z_constrained = z_input
             x_constrained = parent_node.zonotope

        # 3. Generate Action (Independent)
        u_input = self._get_constrained_action(constraints)
        
        # 4. Combine State and Action
        x_aligned, u_aligned = x_constrained.stack_independent(u_input)
        xu_input = x_aligned.concatenate([u_aligned], axis=-1)
        
        # 5. Align Z
        # Now we align the CARRIED z_constrained with the new system.
        # Since z_constrained's generators are already a subset of x_constrained's generators (by ID),
        # stack_independent might get confused if we don't treat them properly?
        # NO. stack_independent assumes *disjoint* generators. 
        # But here X and Z *share* generators.
        # If we behave as if they are independent, we duplicate them -> ERROR.
        
        # We need a "Align Shared" or "Pass Through".
        # If Z is already part of the history, we don't need to stack it?
        # AbstractENN expects (xu, z).
        # Inside AbstractENN, it concatenates xu and z.
        # If they share history, concatenate should "just work" (inherit history)?
        # Zonotope.concatenate logic:
        # "Assuming aligned inputs, we just inherit the history of self."
        # If we pass z_constrained (length K) and xu_input (length K+A).
        # We need to PAD z_constrained to length K+A using zeros for the 'Action' columns?
        # Yes.
        
        def pad_to_match(target_z: Zonotope, reference_shape_1: int) -> Zonotope:
            pad_len = reference_shape_1 - target_z.generators.shape[1]
            if pad_len > 0:
                padding = jnp.zeros((target_z.generators.shape[0], pad_len, target_z.generators.shape[2]))
                new_gens = jnp.concatenate([target_z.generators, padding], axis=1)
                return Zonotope(target_z.center, new_gens, target_z.history) # History?
            return target_z

        # xu_input has [Shared_Z, Action_Gens].
        # z_constrained has [Shared_Z].
        # We pad Z to [Shared_Z, 0] to match xu.
        
        z_final = pad_to_match(z_constrained, xu_input.generators.shape[1])
        # We must ensure z_final has the same history object to avoid validation errors if we added checks?
        # For now, Zonotope.concatenate just trusts us and takes self.history.
        # So we update z_final history to match xu_input?
        z_final.history = xu_input.history # Hacky but effectively true (Z is subset)

        # 6. Prepare Split Dictionary
        split_dict = {k: list(v) for k, v in constraints.relu_splits.items()}
        
        # 7. Forward Pass
        next_zonos = self.model(
            xu_input, 
            z_final, 
            split_idxs=split_dict
        )
        
        # 8. Create Child Nodes
        for z in next_zonos:
            self.push(SearchNode(
                priority=parent_node.priority, 
                timestep=parent_node.timestep + 1,
                zonotope=z,
                z_zonotope=z_constrained, # Pass the constrained Z forward as the new baseline
                parent=parent_node,
                constraints=constraints 
            ))

    def _refine(self, node: SearchNode):
        if node.parent is None:
            # Debug Root Safety
            _, ub = node.zonotope.project_bounds(self.unsafe_h)
            max_ub = float(jnp.max(ub))
            print(f"  [Error] Root Node is unsafe. MaxProj={max_ub:.4f} > Thresh={self.unsafe_thresh}")
            return

        ranking = node.zonotope.get_sensitivity_ranking(self.unsafe_h)
        if not ranking:
            print("  [Warning] No sensitivity ranking found.")
            return

        # Stochastic Selection
        # Filter to valid options
        candidates = [r for r in ranking if r.gain > 1e-9]
        if not candidates:
            candidates = ranking[:1] # Fallback to best if all are zero

        # Weighted sampling
        weights = [r.gain for r in candidates]
        best = random.choices(candidates, weights=weights, k=1)[0]
        
        if self.verbosity >= 2:
            print(f"  -> {best.desc}")
        
        if best.source_type == 'action':
            new_constraint_sets = self._generate_action_splits(node, best)
        elif best.source_type == 'z':
            new_constraint_sets = self._generate_z_splits(node.constraints, best)
        elif best.source_type == 'relu':
            new_constraint_sets = self._generate_relu_splits(node.constraints, best)
        elif best.source_type == 'input':
            if self.verbosity >= 1:
                print(f"  [Skip] Splitting 'input' source not implemented yet (requires root refinement).")
            return
        else:
            raise Exception(f"Unknown source type: {best.source_type}")
            
        # Record Split Type for Aggregation
        self.split_type_map[node] = best.source_type
        
        # Re-Propagate
        parent = node.parent
        for new_const in new_constraint_sets:
            # Pruning: Check mass before propagating
            if new_const.probability_mass < 1e-6:
                continue
            self._propagate(parent, new_const)

    def compute_safe_probability(self, root_node: SearchNode) -> float:
        """
        Iterative MiniMax Aggregation (Post-Order).
        Calculates the lower bound of safe probability for the subtree rooted at 'root_node'.
        """
        if root_node is None:
            return 0.0
            
        # 1. Post-Order Traversal to process children before parents
        # We need to compute values bottom-up.
        stack = [root_node]
        visit_order = []
        
        # Standard iterative post-order (approximate: push children, then add to visit list reversed)
        # Or just:
        # Pre-order traversal, then reverse it? 
        # Yes, children map allows this easily.
        
        # Traverse to find all reachable nodes from root
        # (Note: self.children_map might contain nodes not in this subtree if we reuse solver? 
        # But we usually clear solver. So assuming tree is clean.)
        
        # BFS/DFS to get all nodes
        traversal_stack = [root_node]
        all_nodes = []
        while traversal_stack:
            curr = traversal_stack.pop()
            all_nodes.append(curr)
            children = self.children_map.get(curr, [])
            for c in children:
                traversal_stack.append(c)
                
        # Now iterate in reverse (bottom-up)
        node_values = {}
        
        for node in reversed(all_nodes):
            # Check if leaf (safe or dead end)
            if node in self.safe_leaves:
                node_values[node] = node.constraints.probability_mass
                continue
                
            children = self.children_map.get(node, [])
            if not children:
                # Dead end / Pruned
                node_values[node] = 0.0
                continue
                
            # Compute based on children
            child_vals = [node_values.get(c, 0.0) for c in children]
            
            split_type = self.split_type_map.get(node, 'unknown')
            
            if split_type == 'action':
                val = max(child_vals)
            else:
                # Z / ReLU / Unknown -> Sum
                val = sum(child_vals)
                
            node_values[node] = val
            
        return node_values.get(root_node, 0.0)

    def get_best_action(self, default_action: np.ndarray = np.array([0.0])) -> Tuple[np.ndarray, float, bool]:
        """
        Returns (action, total_safe_mass_lower_bound, is_verified_safe)
        """
        if self.root is None:
             return default_action, 0.0, False
             
        # 1. Compute Global Safe Probability Lower Bound
        total_safe_mass = self.compute_safe_probability(self.root)
        
        # 2. Find Action
        # We want the action that leads to this mass.
        # The root itself connects to t=0 nodes (or is t=0).
        # Wait, dev_main initializes Root at t=0. 
        # The first "Step" propogates t=0 -> t=1. This creates children.
        # This propagation is usually 1-to-1 unless we split the input state/action/z immediately?
        # Actually `solve.step()` pops `current_node`.
        # If Safe -> Propagate (t+1). Split Type = 'transition' (implicitly)
        # If Unsafe -> Refine. Split Type = 'action'/'z'/etc.
        
        # If the Root was immediately Safe (rare), we propagate unique child at t=1.
        # If the Root was Unsafe, we Refine it.
        # So Root will have Children (the splits).
        
        # We trace down from Root, following the 'argmax' path for Action splits,
        # and just taking the first/best path for 'sum' splits?
        # Actually, for MPC we need the action at t=0.
        # SearchNode at t=0 contains `constraints`.
        # Initial constraints usually have Action [-3, 3].
        # If we split Action at Root, we get children with [-3, 0], [0, 3].
        # We pick the child with higher `compute_safe_probability`.
        # We recurse until we find a child that represents a "Transition" (propagation to t=1).
        # The constraints of that node define the action we commit to.
        
        curr = self.root
        best_mass = total_safe_mass
        
        while curr and curr.timestep == 0:
            children = self.children_map.get(curr, [])
            if not children:
                # Dead end at t=0? No safe action found.
                break
                
            split_type = self.split_type_map.get(curr, 'unknown')
            
            if split_type == 'action':
                # Pick best child
                best_child = max(children, key=self.compute_safe_probability)
                curr = best_child
            elif split_type in ['z', 'relu', 'input']:
                # The action is valid for ALL these scenarios?
                # No, if we split on Z, we are essentially saying "The policy works for Z_a AND Z_b".
                # The action bounds should be consistent/shared?
                # Actually, `refine` might split Z, but the Action bounds on both children remain same.
                # So ANY child (with mass > 0) represents the same Action constraint.
                # We can just pick the one with most mass to follow? 
                # Or just pick the first one?
                valid_children = [c for c in children if self.compute_safe_probability(c) > 0]
                if not valid_children:
                    break
                curr = valid_children[0]
            else:
                # Transition / Propagate?
                # If we propagated, curr.timestep increases. Loop ends.
                # But here we are inside existing nodes.
                curr = children[0]
                
        # Now curr is the node we decided on.
        # If curr.timestep > 0, it means we propagated.
        # The PROPOSAL/ACTION is in the constraints of the node *before* propagation?
        # `SearchNode` stores `constraints`.
        # If we are at t=0, `constraints.action_bounds` defines our commit.
        
        c = curr.constraints
        if c.action_bounds:
            mins, maxs = c.action_bounds
            action = (np.array(mins) + np.array(maxs)) / 2.0
        else:
            action = default_action
            
        return action, total_safe_mass, (total_safe_mass > 0.0)

    def get_best_trajectory(self) -> List[Dict]:
        """
        Returns a list of steps in the trajectory:
        [{'timestep': t, 'action': a, 'interval': (min, max)}, ...]
        from t=0 to leaf.
        """
        target_node = None
        if self.safe_leaves:
            sorted_leaves = sorted(self.safe_leaves, key=lambda n: n.constraints.probability_mass, reverse=True)
            target_node = sorted_leaves[0]
        elif self.best_node is not None:
            target_node = self.best_node
            
        if target_node is None:
            return []
            
        # Backtrack
        path = []
        curr = target_node
        while curr is not None:
            # We want to record the action taken to reach this node.
            # Node at timestep t was reached by action at t-1.
            # But SearchNode stores constraints including 'action_bounds' of that transition.
            
            if curr.timestep > 0: # Root (t=0) has no previous action
                c = curr.constraints
                action_info = {'timestep': curr.timestep - 1, 'action': None, 'bounds': None}
                
                if c.action_bounds:
                    mins, maxs = c.action_bounds
                    # Center
                    a_val = (np.array(mins) + np.array(maxs)) / 2.0
                    action_info['action'] = a_val
                    action_info['bounds'] = (mins, maxs)
                else:
                    action_info['action'] = np.array([0.0])
                    
                path.append(action_info)
                
            curr = curr.parent
            
        return list(reversed(path))
        
        # Logging is cleaner now
        print(f"  -> {best.desc}")
        
        # Access via dot notation
        if best.source_type == 'action':
            new_constraint_sets = self._generate_action_splits(node, best)
        elif best.source_type == 'z':
            new_constraint_sets = self._generate_z_splits(node.constraints, best)
        elif best.source_type == 'relu':
            new_constraint_sets = self._generate_relu_splits(node.constraints, best)
        elif best.source_type == 'input':
            # Splitting initial state (x_0) requires restarting from root or carrying x constraints.
            # For now, we skip to avoid crash.
            print(f"  [Skip] Splitting 'input' source not implemented yet (requires root refinement).")
            return
        else:
            raise Exception
            
        # Re-Propagate
        parent = node.parent
        for new_const in new_constraint_sets:
            self._propagate(parent, new_const)

    def _generate_action_splits(self, node: SearchNode, result: SensitivityResult) -> List[ConstraintState]:
        current_c = node.constraints
        u_zono = self._get_constrained_action(current_c)
        lb, ub = u_zono.concrete_bounds()
        
        # Access via object
        dim_idx = result.meta.get('local_index', 0)
        
        # Attempt Calculated Split (Safety Margin)
        # We want to find a split point 's' in concrete domain such that one side is safe.
        
        # 1. Get Generator info
        # Global generator index usually corresponds to where 'action' block starts + local
        global_gen_idx = result.indices[0] + dim_idx
        
        # 2. Project Generator onto Unsafe Vector
        # w = h^T g_i
        # We need the projection of this specific generator. 
        # But wait, 'u_zono' is strictly the action part. 'node.zonotope' is the output.
        # We need the impact on the OUTPUT.
        # The 'result' object gave us the 'gain' based on the OUTPUT projection.
        # We need to re-calculate 'w' exactly.
        
        # node.zonotope.generators: [Batch, N_err, OutDim]
        # self.unsafe_h: [OutDim]
        # projected: [Batch, N_err]
        proj_gens = jnp.dot(node.zonotope.generators, self.unsafe_h) # [Batch, N_err]
        w_vec = proj_gens[:, global_gen_idx] # [Batch]
        
        # We take the worst-case w (largest magnitude? No, valid w).
        # Actually usually batch=1.
        w = float(w_vec[0])
        
        # 3. Calculate Required Reduction
        # UB = CenterProj + Sum(|w_j|)
        # We want NewUB <= Threshold
        # NewUB = OldUB - |w| + NewContribution
        # Gap = OldUB - Threshold
        # We need Reduction >= Gap.
        
        _, current_ub = node.zonotope.project_bounds(self.unsafe_h)
        current_safe_gap = float(jnp.max(current_ub)) - self.unsafe_thresh
        
        # If already safe, we shouldn't be here.
        # If gap is huge > |w|, we can't fix it with this feature alone.
        
        split_val = None
        
        # Epsilon for numerical stability
        SAFE_MARGIN = 1e-5
        target_reduction = current_safe_gap + SAFE_MARGIN
        
        if target_reduction < abs(w):
            # It IS possible to fix it with this feature!
            
            # Derivation from plan:
            # If w > 0: s <= 1 - R/w
            # If w < 0: s >= -1 + R/|w|
            
            if w > 0:
                s_limit = 1.0 - target_reduction / w
                # We want the interval [-1, s_limit] to be the SAFE one.
                # So split point is s_limit.
                # Valid s must be > -1.
                if s_limit > -1.0:
                    split_frac = s_limit
                else:
                    split_frac = 0.0 # Fallback
            else: # w < 0
                s_limit = -1.0 + target_reduction / abs(w)
                # We want the interval [s_limit, 1] to be safe.
                # So split point is s_limit.
                if s_limit < 1.0:
                    split_frac = s_limit
                else:
                    split_frac = 0.0
                    
            # Clamp split_frac to reasonable bounds to avoid slivers
            # e.g. [-0.95, 0.95]
            split_frac = max(-0.95, min(0.95, split_frac))
            
            # Map split_frac [-1, 1] to Concrete Domain [L, R]
            c_min = lb[0, dim_idx]
            c_max = ub[0, dim_idx]
            
            split_val = c_min + (c_max - c_min) * (split_frac + 1.0) / 2.0
            
            print(f"    [Calculated Split] Gap={current_safe_gap:.4f}, w={w:.4f} -> SplitFrac={split_frac:.2f}")
            
        else:
            # Cannot fix with single split, fallback to bisection
            split_val = (lb[0, dim_idx] + ub[0, dim_idx]) / 2.0
            # print(f"    [Bisection] Gap={current_safe_gap:.4f} > |w|={abs(w):.4f}")

        results = []
        # Create Lower Half and Upper Half using split_val
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
        mid = (lb[0, dim_idx] + ub[0, dim_idx]) / 2.0
        
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
        
        # 1. Sanity Check
        if layer == 'unknown' or idx == -1:
            return []

        # 2. Check Existing Constraints
        curr_layer_splits = current_c.relu_splits.get(layer, ())
        
        # Extract just the indices that are already constrained
        existing_constrained_indices = {s[0] for s in curr_layer_splits}
        
        if idx in existing_constrained_indices:
            # The solver wants to split a neuron we have already fixed.
            # This usually means the sensitivity gain is negligible (numerical noise)
            # or the constraint didn't work. We skip to avoid infinite loops.
            # print(f"  [Skip] Neuron {layer}[{idx}] is already constrained.")
            return []

        results = []
        
        # 3. Generate Branches
        for status in ['active', 'inactive']:
            # Append new split
            new_splits = curr_layer_splits + ((idx, status),)
            
            # Update dictionary
            new_map = current_c.relu_splits.copy()
            new_map[layer] = new_splits
            
            results.append(replace(current_c, relu_splits=new_map))
            
        return results

    # --- Helpers ---

    def _get_constrained_action(self, c: ConstraintState) -> Zonotope:
        # Default Action Bounds (Update these to your env specs)
        default_min = jnp.array([-3.0]) 
        default_max = jnp.array([3.0])
        
        if c.action_bounds:
            mins, maxs = c.action_bounds
            u_min = jnp.array(mins)
            u_max = jnp.array(maxs)
        else:
            u_min, u_max = default_min, default_max
            
        u_zono = box_to_zonotope(u_min, u_max)
        # Log for sensitivity
        u_zono.history = (GeneratorGroup('action', 0, u_zono.generators.shape[1]),)
        return u_zono

    def _get_constrained_z(self, c: ConstraintState) -> Zonotope:
        # Default Z Bounds
        default_min = jnp.array([-3.0, -3.0, -3.0, -3.0]) 
        default_max = jnp.array([3.0, 3.0, 3.0, 3.0])
        
        if c.z_bounds:
            mins, maxs = c.z_bounds
            z_min = jnp.array(mins)
            z_max = jnp.array(maxs)
        else:
            z_min, z_max = default_min, default_max
            
        z_zono = box_to_zonotope(z_min, z_max)
        z_zono.history = (GeneratorGroup('z', 0, z_zono.generators.shape[1]),)
        return z_zono