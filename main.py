from collections import defaultdict
from pathlib import Path
import numpy as np
from src.data import TrajectoryDataset, collect_data
import gymnasium as gym
from src.net import ENN, train_step
from flax import nnx
import jax.numpy as jnp
import optax
import jax
from tqdm import tqdm
from util import TensorboardLogger, EarlyStopper
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import io
from PIL import Image
from src.eval import compute_val_diagnostics
import dill as pickle
from src.star import Star
import polytope as pc
from src.affine import Affine
from src.polytope import Polytope


def train_enn(model, data: TrajectoryDataset, max_epochs, batch_size, rngs, logger=None, global_step=0):
    optimizer = nnx.Optimizer(
        model, optax.adamw(1e-3, weight_decay=1e-4), wrt=nnx.Param
    )

    key = jax.random.PRNGKey(10)
    train_ds, val_ds = data.split(0.15)
    
    # Handle Logger
    if logger is None:
        logger = TensorboardLogger('./runs/', 'enn')
        should_close = True
    else:
        should_close = False
        
    early_stopping = EarlyStopper(patience=20)
    
    for epoch in range(max_epochs):
        train_metrics = defaultdict(list)
        for batch in tqdm(
            train_ds.iterate_transitions(
                batch_size=batch_size, shuffle=True, seed=epoch
            ),
            total=int(np.ceil(len(train_ds.actions) / batch_size)),
            desc=f"Epoch {epoch}",
            leave=False
        ):
            key, k1 = jax.random.split(key)
            s, a, ns, _ = batch
            x_batch = jnp.concatenate([s, a], axis=-1)
            y_batch = jnp.asarray(ns)

            loss = train_step(
                model, optimizer, x_batch, y_batch, k1, num_heads=64*4, bootstrap_p=0.8
            )
            train_metrics['loss'].append(loss)

            global_step += 1
            
        # Log Train
        for k, v in train_metrics.items():
            logger.writer.add_scalar(f'train/{k}', float(np.mean(v)), global_step)
        
        val_metrics = defaultdict(list)
        for batch in val_ds.iterate_transitions(batch_size=batch_size):
            key, k1 = jax.random.split(key)
            s, a, ns, _ = batch
            x_batch = jnp.concatenate([s, a], axis=-1)
            y_batch = jnp.asarray(ns)

            loss = train_step(
                model, None, x_batch, y_batch, k1, num_heads=64*4, bootstrap_p=1.0
            )
            val_metrics['loss'].append(loss)

        # Log Val
        val_loss = float(np.mean(val_metrics['loss']))
        logger.writer.add_scalar('val/loss', val_loss, global_step)
        
        if early_stopping.update(val_loss, model):
            break

    early_stopping.restore_best(model)
    
    if should_close:
        pickle.dump(model, open('model.pkl', 'wb'))
        logger.close()
        
    return model, global_step

def rollout_experiment(
    env_name='InvertedPendulum-v5', initial_env_steps=1_000, max_epochs=1000, batch_size=32,
):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    def select_safe_action(model: ENN, state, rngs):
        # Constrain s=state, a \in action_space, z \in (-inf, inf).
        # We define the input set over (a, z) and map to (s, a, z) via affine transform
        # to avoid "flat" polytopes (equality constraints on s) which can cause is_empty() checks to fail.
        z_dim = model.z_dim
        
        # Input set: a \in [low, high], z \in R. Variables: [a, z]
        I_a = jnp.eye(act_dim)
        Z_az = jnp.zeros((act_dim, z_dim))
        
        # Constraints: -a <= -low, a <= high
        A_in = jnp.concatenate([jnp.concatenate([-I_a, Z_az], axis=1), jnp.concatenate([I_a, Z_az], axis=1)], axis=0)
        b_in = jnp.concatenate([-env.action_space.low, env.action_space.high])
        
        # Transform: [a, z] -> [s, a, z]
        zeros_s = jnp.zeros((obs_dim, act_dim + z_dim))
        I_az = jnp.eye(act_dim + z_dim)
        A_trans = jnp.concatenate([zeros_s, I_az], axis=0)
        b_trans = jnp.concatenate([state, jnp.zeros(act_dim + z_dim)])
        
        star = Star(Polytope(A_in, b_in), Affine(A_trans, b_trans))
        stars = model.propagate_star_set(star)
        return stars

    (states, actions, next_states, dones), (env, s) = collect_data(env, steps=initial_env_steps)
    data = TrajectoryDataset(states, actions, next_states, dones)
    Path('data.pkl').write_bytes(pickle.dumps(data))

    rngs = nnx.Rngs(params=0, epistemic=1)
    model = ENN(
        x_dim = obs_dim,
        a_dim = act_dim,
        z_dim = 4,
        hidden_dim=8,
        rngs=rngs,
    )
    model, _ = train_enn(model, data, max_epochs, batch_size, rngs)

    for _ in range(200):
        a = select_safe_action(model, s, rngs)
        s_next, _, terminated, truncated, _ = env.step(a)
        data.append_transition(s, a, s_next, terminated or truncated)

        if terminated or truncated:
            s, _ = env.reset()
        else:
            s = s_next

if __name__ == "__main__":
    rollout_experiment()