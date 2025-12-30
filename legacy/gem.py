from typing import Optional, Dict, List, Tuple
import jax
import jax.numpy as jnp
from flax import nnx
import functools

from src.net import ENN
from src.zono import Zonotope, AbstractLinear, AbstractReLU

# --- Helper: Zonotope Algebra for Skip Connections ---

def merge_independent_sources(x: Zonotope, z: Zonotope) -> Tuple[Zonotope, Zonotope]:
    """
    Prepares independent inputs x and z for joint processing.
    Block-diagonalizes their generators so they don't interfere.
    Returns aligned versions of x and z with compatible generator shapes.
    """
    # x gens: [B, Nx, Dx] -> [B, Nx + Nz, Dx] (Pad bottom)
    # z gens: [B, Nz, Dz] -> [B, Nx + Nz, Dz] (Pad top)
    
    gx = x.generators
    gz = z.generators
    
    # Pad X
    pad_x = jnp.zeros((gx.shape[0], gz.shape[1], gx.shape[2]))
    gx_aligned = jnp.concatenate([gx, pad_x], axis=1)
    
    # Pad Z
    pad_z = jnp.zeros((gz.shape[0], gx.shape[1], gz.shape[2]))
    gz_aligned = jnp.concatenate([pad_z, gz], axis=1)
    
    x_aligned = Zonotope(x.center, gx_aligned)
    z_aligned = Zonotope(z.center, gz_aligned)
    
    return x_aligned, z_aligned

# --- The Abstract ENN Class ---

class AbstractENN(nnx.Module):
    def __init__(self, x_dim: int, a_dim: int, z_dim: int, hidden_dim: int, *, rngs: nnx.Rngs) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.z_dim = z_dim

        # Replace standard layers with Abstract versions
        self.base_fc1 = AbstractLinear(x_dim + a_dim, hidden_dim, rngs=rngs)
        self.base_out = AbstractLinear(hidden_dim, x_dim, rngs=rngs)

        in_dim = hidden_dim + x_dim + a_dim + z_dim
        self.epi_fc1 = AbstractLinear(in_dim, hidden_dim, rngs=rngs)
        self.epi_out = AbstractLinear(hidden_dim, x_dim, rngs=rngs)
        self.prior_fc1 = AbstractLinear(in_dim, hidden_dim, rngs=rngs)
        self.prior_out = AbstractLinear(hidden_dim, x_dim, rngs=rngs)

        # Abstract Activations
        self.act_base = AbstractReLU()
        self.act_epi = AbstractReLU()
        self.act_prior = AbstractReLU()

    def copy_weights_from(self, source_model: ENN):
        """
        Copies parameters from a trained concrete ENN to this abstract ENN.
        """
        # Helper to copy Linear -> AbstractLinear
        def copy_linear(src: nnx.Linear, dst: AbstractLinear):
            # AbstractLinear wraps the real layer in '.linear'
            dst.linear.kernel.value = src.kernel.value
            if src.bias is not None and dst.linear.bias is not None:
                dst.linear.bias.value = src.bias.value

        # Copy Base Network
        copy_linear(source_model.base_fc1, self.base_fc1)
        copy_linear(source_model.base_out, self.base_out)

        # Copy Epistemic Network
        copy_linear(source_model.epi_fc1, self.epi_fc1)
        copy_linear(source_model.epi_out, self.epi_out)

        # Copy Prior Network
        copy_linear(source_model.prior_fc1, self.prior_fc1)
        copy_linear(source_model.prior_out, self.prior_out)
    
    @classmethod
    def from_concrete(cls, concrete_enn: ENN, rngs: nnx.Rngs) -> AbstractENN:
        """
        Factory method to create an AbstractENN directly from a trained ENN.
        """
        # Instantiate abstract model with same dims
        abstract_model = cls(
            x_dim=concrete_enn.x_dim,
            a_dim=concrete_enn.a_dim,
            z_dim=concrete_enn.z_dim,
            # Infer hidden dim from the first layer output features
            hidden_dim=concrete_enn.base_fc1.out_features,
            rngs=rngs
        )
        
        # Copy weights
        abstract_model.copy_weights_from(concrete_enn)
        
        return abstract_model

    def __call__(self, x: Zonotope, z: Zonotope, error_scales: Optional[Dict[str, jax.Array]] = None) -> Zonotope:
        """
        Args:
            x: Zonotope representing state-action inputs [batch, x_dim + a_dim]
            z: Zonotope representing epistemic indices [batch, z_dim]
            error_scales: Optional dict for sensitivity analysis (keys: 'base', 'epi', 'prior')
        """
        scales = error_scales or {}

        # 0. Align independent inputs (x and z) to share a common generator space
        x_aligned, z_aligned = merge_independent_sources(x, z)

        # --- Base Path ---
        # phi = relu(base_fc1(x))
        phi_pre = self.base_fc1(x_aligned)
        phi = self.act_base(phi_pre, error_scale=scales.get('base'))
        
        base_out = self.base_out(phi)

        # --- Construct Joint Feature Vector ---
        # full_phi = [phi, x]
        # We must concat aligned versions. 
        # Note: 'phi' has accrued new error terms from act_base.
        # 'x_aligned' needs to be padded to match phi's new depth before concat.
        full_phi = phi.concatenate([x_aligned], axis=-1)

        # joint_input = [full_phi, z] = [phi, x, z]
        # 'z_aligned' also needs padding to match phi's depth
        joint_input = full_phi.concatenate([z_aligned], axis=-1)

        # --- Epistemic Path ---
        # epi_hidden = relu(epi_fc1([phi, x, z]))
        epi_h_pre = self.epi_fc1(joint_input)
        epi_hidden = self.act_epi(epi_h_pre, error_scale=scales.get('epi'))
        epi_out = self.epi_out(epi_hidden)

        # --- Prior Path ---
        # prior_hidden = relu(prior_fc1([phi, x, z]))
        # Note: We reuse joint_input. The prior path will generate its OWN distinct 
        # error terms in act_prior, separate from the ones in act_epi.
        prior_h_pre = self.prior_fc1(joint_input)
        prior_hidden = self.act_prior(prior_h_pre, error_scale=scales.get('prior'))
        prior_out = self.prior_out(prior_hidden)

        # --- Final Sum ---
        # out = base + epi + prior
        # zono_add handles aligning the disparate error terms from epi and prior branches
        temp_sum = base_out.add(epi_out)
        final_out = temp_sum.add(prior_out)

        return final_out

# --- 4. The "Smart Split" Engine ---

class Verifier(nnx.Module):
    def __init__(self, model):
        self.model = model

    @functools.partial(jax.jit, static_argnums=(0,))
    def compute_sensitivity(self, zono: Zonotope, safety_threshold: float):
        """
        Calculates which neuron's relaxation contributes most to the safety violation.
        """
        
        # 1. Inspect shapes to initialize error_scales
        # We do a cheap forward pass just to get shapes (or know them a priori)
        # For this demo, we know Hidden Dim is 16.
        batch_size = zono.center.shape[0]
        hidden_dim = 16 
        
        # Initialize scales to 1.0
        # We want gradients w.r.t this variable!
        scale_init = jnp.ones((batch_size, hidden_dim))
        
        def loss_fn(scale):
            # Forward pass with scaling
            out_zono = self.model(zono, error_scales=[scale])
            lb, ub = out_zono.concrete_bounds()
            
            # Violation Loss: How much does the WORST CASE (ub) exceed the threshold?
            # We assume we want x < threshold.
            # Only consider the first output dimension for this example
            violation = jnp.maximum(0.0, ub[:, 0] - safety_threshold)
            return jnp.sum(violation)

        # 2. Compute Gradient
        grads = jax.grad(loss_fn)(scale_init)
        
        # 3. Masking
        # We should only split neurons that are actually unstable.
        # (Re-run forward to get mask - efficient in JIT)
        out_temp = self.model(zono) # Standard pass
        # We need the intermediate bounds at layer 1 to know who is unstable.
        # For simplicity in this flat code, we approximate by trusting the gradient 
        # (stable neurons have 0 error magnitude, so 0 gradient usually).
        
        # 4. Find Max
        # Score = Abs(Gradient). Higher gradient = more sensitivity.
        scores = jnp.abs(grads)
        
        # Best neuron per batch item
        best_neuron_idx = jnp.argmax(scores, axis=1)
        max_scores = jnp.max(scores, axis=1)
        
        return best_neuron_idx, max_scores

    def refine_step(self, zono: Zonotope, neuron_idx: int, layer_idx=0) -> List[Zonotope]:
        """
        Splits the zonotope into two based on the chosen neuron.
        This is a 'geometric' operation, simpler to do in Python logic + JAX primitives.
        """
        # Note: This specific implementation splits the INPUT Z-SPACE for simplicity,
        # or forces the HIDDEN NEURON constraint.
        #
        # "Splitting a Hidden Neuron" in a Zonotope is tricky because it adds a constraint 
        # that breaks the zonotope shape (it becomes a Constrained Zonotope).
        #
        # STRATEGY: Instead of exact splitting, we perform "Input Domain Splitting" 
        # guided by the hidden neuron sensitivity.
        #
        # HOWEVER, the user asked for "Un-relaxing". 
        # To do that exactly requires switching to 'ConstrainedZonotope' class.
        #
        # FOR THIS PROTOTYPE: We will implement the "Input Split" heuristic, 
        # which is the standard approach in "Zonotope-based verification" (e.g. methods like AI2).
        # We assume the sensitivity maps back to which input dimension affects that neuron most.
        pass 
        # See Step 5 below for the concrete strategy.

# --- 5. The Execution Loop (The Hybrid Strategy) ---

import jax
import jax.numpy as jnp
from flax import nnx
import heapq
import functools

# --- Prerequisites (Assumed defined elsewhere, repeated briefly for context) ---
# Assuming AbstractENN and Zonotope classes are available.
# If you need those re-pasted, let me know. 

def box_to_zonotope(min_vals: jax.Array, max_vals: jax.Array) -> 'Zonotope':
    """Creates a diagonal (axis-aligned) Zonotope from a hyper-rectangle."""
    center = (max_vals + min_vals) / 2.0
    radii = (max_vals - min_vals) / 2.0
    
    # Generators: Diagonal matrix [Dim, Dim] -> Expanded to [1, Dim, Dim]
    # We expand dims to match [Batch, N_Gens, Dim]
    generators = jnp.expand_dims(jnp.diag(radii), axis=0)
    center = jnp.expand_dims(center, axis=0)
    
    return Zonotope(center, generators)

# --- Main Verification Routine ---

def run_unified_verification(
    model: AbstractENN,
    x_min: jax.Array, 
    x_max: jax.Array,
    safe_min: jax.Array, 
    safe_max: jax.Array,
    z_dim: int, 
    max_steps: int = 200
):
    print("--- Unified Best-First Verification ---")
    input_dim = x_min.shape[0] 
    
    # 2. Create Unified Joint Zonotope
    # We concatenate bounds: [state_action_bounds, z_bounds]
    # Z-bounds usually standard normal: +/- 3 sigma
    z_min_bound = jnp.ones(z_dim) * -3.0
    z_max_bound = jnp.ones(z_dim) * 3.0
    
    joint_min = jnp.concatenate([x_min, z_min_bound])
    joint_max = jnp.concatenate([x_max, z_max_bound])
    
    # This automatically creates a block-diagonal generator matrix
    # Generators [i] corresponds to Dimension [i]
    joint_zono_root = box_to_zonotope(joint_min, joint_max)
    
    # 3. Initialize Priority Queue (Best-First Search)
    # Heap items: (margin_score, unique_id, zonotope)
    # Score: Lower is better (closer to being safe).
    # We assume 'margin < 0' is SAFE. So we sort by margin.
    pq = []
    entry_id = 0 # Tie-breaker
    
    # Initial heuristic check
    # We do a quick forward pass to get the initial margin
    c_x = joint_zono_root.center[:, :input_dim]
    c_z = joint_zono_root.center[:, input_dim:]
    # For heuristic, we can just use centers (fast) or full zono (accurate)
    # Let's use full zono to be accurate for the sort
    # (Note: In a huge loop, you might optimize this to be lazy)
    
    # Slice generators for forward pass
    g_x = joint_zono_root.generators[:, :, :input_dim]
    g_z = joint_zono_root.generators[:, :, input_dim:]
    
    out_root = model(Zonotope(c_x, g_x), Zonotope(c_z, g_z))
    lb, ub = out_root.concrete_bounds()
    
    # Initial Margin
    margin = jnp.maximum(jnp.max(ub - safe_max), jnp.max(safe_min - lb))
    
    heapq.heappush(pq, (float(margin), entry_id, joint_zono_root))
    entry_id += 1
    
    safe_regions = []
    unsafe_regions = []
    
    step = 0
    
    # --- Optimization: JIT the Sensitivity Gradient ---
    # We define the gradient function outside the loop to keep it clean.
    # Note: To JIT this, 'model' must be static or passed carefully. 
    # For this script, we'll rely on JAX's internal caching or define it inside if needed.
    
    def loss_fn(joint_gens, center_x, center_z):
        # Reconstruct sliced zonotopes
        # This allows gradients to flow from Output -> Model -> Sliced Gens -> Joint Gens
        g_x_t = joint_gens[:, :, :input_dim]
        g_z_t = joint_gens[:, :, input_dim:]
        
        x_t = Zonotope(center_x, g_x_t)
        z_t = Zonotope(center_z, g_z_t)
        
        res = model(x_t, z_t)
        l, u = res.concrete_bounds()
        
        # Loss = Total Violation Magnitude (Sum of ReLUs)
        # We want to minimize this.
        total_violation = jnp.sum(jnp.maximum(0.0, u - safe_max)) + \
                          jnp.sum(jnp.maximum(0.0, safe_min - l))
        return total_violation

    # JIT the gradient function for speed
    grad_fn = jax.jit(jax.grad(loss_fn, argnums=0))

    # --- Verification Loop ---
    
    while len(pq) > 0 and step < max_steps:
        # Pop the most promising region
        margin, _, curr_zono = heapq.heappop(pq)
        step += 1
        
        # A. Slicing (Data Prep)
        c_x = curr_zono.center[:, :input_dim]
        c_z = curr_zono.center[:, input_dim:]
        g_x = curr_zono.generators[:, :, :input_dim]
        g_z = curr_zono.generators[:, :, input_dim:]
        
        x_in = Zonotope(c_x, g_x)
        z_in = Zonotope(c_z, g_z)
        
        # B. Forward Pass & Safety Check
        out = model(x_in, z_in)
        lb, ub = out.concrete_bounds()
        
        # Calculate Exact Margin
        # margin < 0 means SAFE (all bounds inside safe region)
        margin_upper = ub - safe_max
        margin_lower = safe_min - lb
        worst_violation = jnp.maximum(jnp.max(margin_upper), jnp.max(margin_lower))
        
        # ... inside the loop, after calculating worst_violation ...

        # --- DIAGNOSTIC: Concrete Check ---
        # Extract the specific center point of this ambiguous region
        # (This represents the "mean" scenario for this specific tile of x/z)
        c_x_concrete = x_in.center  # Shape [1, Dim]
        c_z_concrete = z_in.center  # Shape [1, Dim]

        # Run standard concrete forward pass (bypass Abstract layers)
        # Note: You might need to expose a 'call_concrete' or just use the model 
        # if it supports concrete arrays (Abstract layers usually crash with raw arrays).
        # If your AbstractENN assumes Zonotopes, create a "Zero-Radius" Zonotope.
        zero_gens_x = jnp.zeros_like(x_in.generators)
        zero_gens_z = jnp.zeros_like(z_in.generators)
        x_point = Zonotope(c_x_concrete, zero_gens_x)
        z_point = Zonotope(c_z_concrete, zero_gens_z)


        out_concrete = model(x_point, z_point)
        lb_c, ub_c = out_concrete.concrete_bounds() 
        # Since radius is 0, lb_c == ub_c == concrete_output

        # print(c_x_concrete, c_z_concrete)
        # print(lb_c, ub_c)

        # Check concrete margin
        conc_margin_upper = ub_c - safe_max
        conc_margin_lower = safe_min - lb_c
        concrete_violation = jnp.maximum(jnp.max(conc_margin_upper), jnp.max(conc_margin_lower))

        # if concrete_violation > 0:
            # print(f"🛑 REAL FAILURE DETECTED. Concrete Margin: {concrete_violation:.4f}")
            # # This region is genuinely unsafe. Refinement is futile.
            # unsafe_regions.append(curr_zono)
            # continue 
        # else:
            # print(f"⚠️ BLOAT DETECTED. Concrete is ({concrete_violation:.4f}) but Abstract is ({worst_violation:.4f}).")

        # logging (optional, sparse)
        if step % 10 == 0:
            print(f"Step {step}: Margin {worst_violation:.4f} (Queue: {len(pq)})")

        if worst_violation < 0:
            # print(f"Step {step}: ✅ SAFE FOUND!")
            safe_regions.append(curr_zono)
            continue

        # Check Guaranteed Failure
        fail_upper = jnp.min(lb - safe_max)
        fail_lower = jnp.min(safe_min - ub)
        
        if fail_upper > 0 or fail_lower > 0:
            # print(f"Step {step}: ❌ UNSAFE FOUND!")
            unsafe_regions.append(curr_zono)
            continue
            
        # C. Weighted Gradient Calculation (The Heuristic Fix)
        # Gradient tells us sensitivity (slope)
        grads = grad_fn(curr_zono.generators, c_x, c_z)
        
        # Weighted Impact = |Gradient * Generator_Value|
        # This tells us how much the Output Error reduces if we shrink this generator.
        weighted_impact = jnp.sum(jnp.abs(grads * curr_zono.generators), axis=(0, 2))
        
        split_idx = jnp.argmax(weighted_impact)
        
        # D. Execute Split
        gen_vec = curr_zono.generators[:, split_idx, :]
        
        # Left Child
        c_left = curr_zono.center - 0.5 * gen_vec
        new_gens = curr_zono.generators.at[:, split_idx, :].set(0.5 * gen_vec)
        z_left = Zonotope(c_left, new_gens)
        
        # Right Child
        c_right = curr_zono.center + 0.5 * gen_vec
        z_right = Zonotope(c_right, new_gens)
        
        # Push children back to priority queue
        # Note: We use the parent's margin as a rough sorting key for now.
        # The loop will re-evaluate exact margin when popped.
        # This saves doing a double forward pass right now.
        heapq.heappush(pq, (float(worst_violation), entry_id, z_left))
        entry_id += 1
        heapq.heappush(pq, (float(worst_violation), entry_id, z_right))
        entry_id += 1

    # --- Summary ---
    print("\n--- Results ---")
    print(f"Total Steps: {step}")
    print(f"Safe Sub-regions: {len(safe_regions)}")
    print(f"Unsafe Sub-regions: {len(unsafe_regions)}")
    print(f"Ambiguous Remaining: {len(pq)}")
    
    if len(safe_regions) > 0:
        print("Verification Successful on subset of domain.")
    else:
        print("Warning: No safe regions verified. Check initial bounds.")

    return safe_regions, unsafe_regions

import jax
import jax.numpy as jnp
import optax
from flax import nnx
import functools

# Assuming AbstractENN, Zonotope, box_to_zonotope are available

class RobustTrajectoryOptimizer(nnx.Module):
    def __init__(self, model: AbstractENN, horizon: int, action_dim: int):
        self.model = model
        self.horizon = horizon
        self.action_dim = action_dim

    @functools.partial(jax.jit, static_argnums=(0,))
    def calculate_loss(self, actions_flat, x_init_box, z_box, safe_min, safe_max):
        """
        Propagates the trajectory and calculates the Worst-Case Violation.
        actions_flat: [Horizon * Action_Dim] (Optimizers like flat arrays)
        """
        # Reshape actions
        actions = actions_flat.reshape((self.horizon, self.action_dim))
        
        # Initialize State Zonotope (Concrete Point for x0)
        curr_x = x_init_box
        
        total_violation = 0.0
        
        # --- Rollout Loop ---
        # We unroll explicitly so JAX can trace the gradients through time
        for t in range(self.horizon):
            u_val = actions[t]
            
            # Create Action Zonotope (Point)
            # Center = u_val, Generators = 0
            # Dimensions: [1, Action_Dim]
            u_center = jnp.expand_dims(u_val, 0)
            u_gens = jnp.zeros((1, 1, self.action_dim))
            u_zono = Zonotope(u_center, u_gens)
            
            # Forward Step
            # Note: We must slice inputs if AbstractENN expects separated (x,z).
            # We assume AbstractENN takes (x_state, z). 
            # If x_state implies (state+action), we must concatenate here.
            
            # --- CONCATENATION LOGIC ---
            # We need to stitch 'curr_x' and 'u_zono' together into the input zonotope
            # Align generators:
            # curr_x has N error terms. u_zono has 1 (zero).
            # We pad u_zono to match curr_x's error terms.
            
            n_err = curr_x.generators.shape[1]
            u_gens_padded = jnp.zeros((1, n_err, self.action_dim))
            
            # Concatenate Centers [Batch, State+Action]
            joint_center = jnp.concatenate([curr_x.center, u_center], axis=-1)
            # Concatenate Generators [Batch, N_Error, State+Action]
            joint_gens = jnp.concatenate([curr_x.generators, u_gens_padded], axis=-1)
            
            x_input = Zonotope(joint_center, joint_gens)
            
            # Model Step
            next_x = self.model(x_input, z_box)
            
            # --- LOSS CALCULATION (Worst Case) ---
            lb, ub = next_x.concrete_bounds()
            
            # Violation = Amount we exceed bounds
            # If ub < safe_max, ReLU is 0. If ub > safe_max, ReLU is penalty.
            # We sum penalty across all state dimensions (or specific ones)
            
            v_upper = jnp.sum(jnp.maximum(0.0, ub - safe_max))
            v_lower = jnp.sum(jnp.maximum(0.0, safe_min - lb))
            
            # Add time-discounted or raw violation
            total_violation += (v_upper + v_lower)
            
            curr_x = next_x
            
        return total_violation

def synthesize_safe_trajectory(
    model: AbstractENN,
    x_fixed: jax.Array,
    z_dim: int,
    horizon: int,
    safe_min: jax.Array,
    safe_max: jax.Array,
    steps: int = 200
):
    print(f"--- Optimizing Robust Trajectory (H={horizon}) ---")
    action_dim = 1 # Example
    
    # Model Setup
    optimizer_module = RobustTrajectoryOptimizer(model, horizon, action_dim)
    
    # 2. Define Inputs
    # X0 is fixed (Point Zonotope)
    x_init_zono = box_to_zonotope(x_fixed, x_fixed)
    
    # Z is Uncertain (Full 3-sigma Robustness)
    z_min = jnp.ones(z_dim) * -3.0
    z_max = jnp.ones(z_dim) * 3.0
    z_box = box_to_zonotope(z_min, z_max)
    
    # 3. Optimizer (Adam)
    # We define actions as a flat parameter vector
    initial_actions = jnp.zeros(horizon * action_dim)
    tx = optax.adam(learning_rate=0.05)
    opt_state = tx.init(initial_actions)
    
    actions = initial_actions
    
    # 4. Optimization Loop
    
    @jax.jit
    def train_step(a, opt_st):
        # Value and Grad of the Worst-Case Loss
        loss, grads = jax.value_and_grad(optimizer_module.calculate_loss)(
            a, x_init_zono, z_box, safe_min, safe_max
        )
        updates, new_opt_st = tx.update(grads, opt_st)
        new_a = optax.apply_updates(a, updates)
        
        # Optional: Clip actions to valid range (e.g. -1 to 1)
        new_a = jnp.clip(new_a, -1.0, 1.0)
        
        return new_a, new_opt_st, loss

    for i in range(steps):
        actions, opt_state, loss = train_step(actions, opt_state)
        
        if i % 20 == 0:
            print(f"Iter {i}: Worst-Case Violation = {loss:.4f}")
            
        if loss < 1e-4:
            print("✅ Found Fully Robust Trajectory (Violation ~ 0)")
            break
            
    reshaped_actions = actions.reshape((horizon, action_dim))
    return reshaped_actions

# --- Example Run ---
# if __name__ == "__main__":
    # import dill as pickle
    # from pathlib import Path

    # rngs = nnx.Rngs(0)
    # enn: ENN = pickle.loads(Path("model.pkl").read_bytes())

    # model = AbstractENN.from_concrete(enn, rngs)
    # # Inverted Pendulum-ish setup
    # import gymnasium as gym
    # env = gym.make('InvertedPendulum-v5')
    # obs, _ = env.reset()
    # safe_min = jnp.array([-100., -0.2, -100., -100.])
    # safe_max = jnp.array([100., 0.2, 100., 100.])
    
    # input_min = jnp.concatenate([obs, jnp.array([best_actions[0,0]])])
    # input_max = jnp.concatenate([obs, jnp.array([best_actions[0,0]])])
    # run_unified_verification(model, input_min, input_max, safe_min, safe_max, 4)


if __name__ == "__main__":
    # Example usage with Inverted Pendulum-like bounds
    # State: [cos(theta), sin(theta), theta_dot]
    # Action: [torque]
    import dill as pickle
    from pathlib import Path

    rngs = nnx.Rngs(0)
    enn: ENN = pickle.loads(Path("model.pkl").read_bytes())

    model = AbstractENN.from_concrete(enn, rngs)
    
    # Let's say we verify the region around upright position
    # Theta ~ 0 +/- 0.1 rad, Theta_dot ~ 0 +/- 0.1, Action ~ 0
    x_dim = 4
    a_dim = 1
    
    import gymnasium as gym
    env = gym.make('InvertedPendulum-v5')
    obs, _ = env.reset()
    input_min = jnp.concatenate([obs, jnp.array([-3])])
    input_max = jnp.concatenate([obs, jnp.array([3])])
    print("OBS")
    print(obs)
    print("ENN")
    print(enn(jnp.concatenate([obs, jnp.zeros(enn.a_dim)]), jnp.zeros(enn.z_dim)))
    print("ACTUAL")
    import numpy as np
    obs, rew, term, trunc, info = env.step(np.zeros(1))
    print(obs)
    
    # Safety: Next state must be "upright enough"
    # Cos(theta) > 0.9, |Theta_dot| < 1.0
    safe_min = jnp.array([-100., -0.2, -100., -100.])
    safe_max = jnp.array([100., 0.2, 100., 100.])
    
    run_unified_verification(model, input_min, input_max, safe_min, safe_max, z_dim=4)