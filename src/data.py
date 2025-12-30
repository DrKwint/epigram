from __future__ import annotations

import numpy as np
from typing import List, Tuple, Iterator, Optional, Literal


Episode = Tuple[int, int]  # (start, end), end exclusive


def collect_data(env, steps=50_000):
    states = []
    actions = []
    next_states = []
    dones = []

    s, _ = env.reset()

    for _ in range(steps):
        a = env.action_space.sample()
        s_next, _, terminated, truncated, _ = env.step(a)

        states.append(s.astype(np.float32))
        actions.append(a.astype(np.float32))
        next_states.append(s_next.astype(np.float32))
        dones.append(terminated or truncated)

        if terminated or truncated:
            s, _ = env.reset()
        else:
            s = s_next

    return (
        np.stack(states),
        np.stack(actions),
        np.stack(next_states),
        np.stack(dones),
    ), (env, s)


class TrajectoryDataset:
    """
    Optimized Trajectory Dataset.

    Improvements:
      1. Dynamic Padding: sample_trajectories pads to batch_max, not global_max.
      2. Buffered Append: append_transition is O(1), flushing occurs on sampling.
    """

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        *,
        episodes: List[Episode] | None = None,
    ):
        # Ensure flat arrays
        assert states.ndim == 2
        assert actions.ndim == 2
        assert next_states.ndim == 2
        assert dones.ndim == 1

        self.states = states
        self.actions = actions
        self.next_states = next_states
        self.dones = dones.astype(bool)

        # Parse episodes if not provided
        if episodes is None:
            episodes = self._parse_episodes(self.dones)

        self.episodes: List[Episode] = episodes
        self.lengths = np.array([e - s for s, e in episodes], dtype=np.int32)

        # Buffer for new transitions (optimization for append_transition)
        self._buf_states: List[np.ndarray] = []
        self._buf_actions: List[np.ndarray] = []
        self._buf_next_states: List[np.ndarray] = []
        self._buf_dones: List[bool] = []

        # Cache for valid indices
        self._transition_idxs: np.ndarray | None = None

    @property
    def E(self) -> int:
        """Number of episodes (including buffered ones implicitly)."""
        self._consolidate_if_needed()
        return len(self.episodes)

    @property
    def L_max(self) -> int:
        """Global maximum length."""
        self._consolidate_if_needed()
        return int(self.lengths.max()) if len(self.lengths) > 0 else 0

    # ------------------------------------------------------------
    # Core Sampling Logic (Optimized)
    # ------------------------------------------------------------
    def sample_trajectories(
        self,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample trajectories with dynamic padding.
        Returns: (states, actions, next_states, dones, valid_lengths)
        """
        self._consolidate_if_needed()

        # 1. Select Indices
        idxs = np.random.randint(0, len(self.episodes), size=batch_size)

        # 2. Determine Batch Geometry (Dynamic Padding)
        batch_lengths = self.lengths[idxs]
        max_batch_len = int(batch_lengths.max())

        B = batch_size
        L = max_batch_len
        s_dim = self.states.shape[1]
        a_dim = self.actions.shape[1]

        # 3. Pre-allocate buffer (Zero-padded by default)
        states = np.zeros((B, L, s_dim), dtype=self.states.dtype)
        actions = np.zeros((B, L, a_dim), dtype=self.actions.dtype)
        next_states = np.zeros((B, L, s_dim), dtype=self.next_states.dtype)
        dones = np.zeros((B, L), dtype=bool)

        # 4. Fill Buffer
        for b, epi_idx in enumerate(idxs):
            start, end = self.episodes[epi_idx]
            actual_len = end - start

            # Direct slice copy
            states[b, :actual_len] = self.states[start:end]
            actions[b, :actual_len] = self.actions[start:end]
            next_states[b, :actual_len] = self.next_states[start:end]
            dones[b, :actual_len] = self.dones[start:end]

        return states, actions, next_states, dones, batch_lengths

    def sample_transitions(
        self,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample transitions without replacement (respecting episode bounds)."""
        self._consolidate_if_needed()

        transition_idxs = self._get_transition_indices()

        chosen = np.random.choice(transition_idxs, size=batch_size, replace=(batch_size > len(transition_idxs)))

        return (
            self.states[chosen],
            self.actions[chosen],
            self.next_states[chosen],
            self.dones[chosen],
        )

    # ------------------------------------------------------------
    # Data Management (Append & Buffer)
    # ------------------------------------------------------------
    def append_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Fast append. Adds to a list buffer.
        Merges into main arrays only when sampling is requested.
        """
        self._buf_states.append(np.asarray(state))
        self._buf_actions.append(np.asarray(action))
        self._buf_next_states.append(np.asarray(next_state))
        self._buf_dones.append(bool(done))

        # Invalidate cache immediately
        self._transition_idxs = None

    def _consolidate_if_needed(self):
        """Merges buffered transitions into the main NumPy arrays."""
        if not self._buf_states:
            return

        # 1. Stack new data
        new_states = np.stack(self._buf_states)
        new_actions = np.stack(self._buf_actions)
        new_next = np.stack(self._buf_next_states)
        new_dones = np.array(self._buf_dones, dtype=bool)

        # 2. Update Episode Indexing
        # We need to know where the old data ended to index the new episodes correctly
        start_idx = len(self.states)

        # Determine if the last old transition was open-ended
        # If dataset was empty, or last transition was DONE, we start fresh.
        # If last transition was NOT done, the first new transition extends it.
        if start_idx > 0 and not self.dones[-1]:
            # The last episode is being extended.
            # We need to modify the last tuple in self.episodes later.
            extending_last = True
        else:
            extending_last = False

        # Parse LOCAL episodes within the new block
        new_ep_relative = self._parse_episodes(new_dones)

        # 3. Merge Arrays
        self.states = np.concatenate([self.states, new_states])
        self.actions = np.concatenate([self.actions, new_actions])
        self.next_states = np.concatenate([self.next_states, new_next])
        self.dones = np.concatenate([self.dones, new_dones])

        # 4. Merge Episode Metadata
        if extending_last:
            # The first "segment" in the new data belongs to the last existing episode
            # Get the first relative episode (0, e)
            first_rel_start, first_rel_end = new_ep_relative[0]

            # Extend the actual last episode
            last_global_start, _ = self.episodes[-1]
            self.episodes[-1] = (last_global_start, start_idx + first_rel_end)

            # Add the remaining new episodes, offset by start_idx
            for s, e in new_ep_relative[1:]:
                self.episodes.append((start_idx + s, start_idx + e))
        else:
            # Just append all new episodes, offset by start_idx
            for s, e in new_ep_relative:
                self.episodes.append((start_idx + s, start_idx + e))

        # 5. Recompute Lengths & Clear Buffer
        self.lengths = np.array([e - s for s, e in self.episodes], dtype=np.int32)

        self._buf_states.clear()
        self._buf_actions.clear()
        self._buf_next_states.clear()
        self._buf_dones.clear()

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    @staticmethod
    def _parse_episodes(dones: np.ndarray) -> List[Episode]:
        """Parses episodes from a bool array of done flags."""
        N = len(dones)
        if N == 0:
            return []

        starts = [0]
        for i in range(N - 1):
            if dones[i]:
                starts.append(i + 1)

        # Handle end of array
        if starts[-1] != N:
            starts.append(N)

        episodes = []
        for i in range(len(starts) - 1):
            s, e = starts[i], starts[i + 1]
            if s < e:
                episodes.append((s, e))
        return episodes

    def _get_transition_indices(self) -> np.ndarray:
        if self._transition_idxs is not None:
            return self._transition_idxs

        idxs = []
        for start, end in self.episodes:
            if end - start >= 1:  # Valid transitions
                # We can use all steps as (s,a)->s'
                # Note: original code used end-1. Usually for (s,a,s') all steps are valid.
                # If you need (s, s') pairs for next-step prediction, all are valid.
                idxs.append(np.arange(start, end, dtype=np.int64))

        if len(idxs) == 0:
            self._transition_idxs = np.empty((0,), dtype=np.int64)
        else:
            self._transition_idxs = np.concatenate(idxs)
        return self._transition_idxs

    def split(self, proportion: float) -> Tuple[TrajectoryDataset, TrajectoryDataset]:
        """Split dataset by episodes."""
        self._consolidate_if_needed()
        assert 0.0 < proportion < 1.0

        total_steps = self.lengths.sum()
        target = int(total_steps * proportion)

        current_steps = 0
        split_idx = len(self.episodes)

        # Simple split from the back
        for i in reversed(range(len(self.episodes))):
            current_steps += self.lengths[i]
            if current_steps >= target:
                split_idx = i
                break

        # Indices
        episodes_a = self.episodes[:split_idx]
        episodes_b = self.episodes[split_idx:]

        def make_subset(eps):
            if not eps:
                raise ValueError("Split resulted in empty dataset")
            # Gather indices
            indices = np.concatenate([np.arange(s, e) for s, e in eps])
            return TrajectoryDataset(
                self.states[indices],
                self.actions[indices],
                self.next_states[indices],
                self.dones[indices],
            )

        return make_subset(episodes_a), make_subset(episodes_b)

    def iterate_trajectories(self, batch_size: int, shuffle: bool = True):
        """
        Yields batches of trajectories, traversing the entire dataset exactly once.
        Applies dynamic padding per batch.
        """
        self._consolidate_if_needed()

        # 1. Get List of Episode Indices
        indices = np.arange(len(self.episodes))

        # 2. Shuffle (for IID assumption)
        if shuffle:
            np.random.shuffle(indices)

        # 3. Iterate
        for start_idx in range(0, len(indices), batch_size):
            # A. Select Batch Indices
            end_idx = min(start_idx + batch_size, len(indices))
            batch_idxs = indices[start_idx:end_idx]

            # B. Determine Batch Geometry (Dynamic Padding)
            batch_lengths = self.lengths[batch_idxs]
            max_batch_len = int(batch_lengths.max())

            B = len(batch_idxs)
            L = max_batch_len
            s_dim = self.states.shape[1]
            a_dim = self.actions.shape[1]

            # C. Allocate Buffer
            states = np.zeros((B, L, s_dim), dtype=self.states.dtype)
            actions = np.zeros((B, L, a_dim), dtype=self.actions.dtype)
            next_states = np.zeros((B, L, s_dim), dtype=self.next_states.dtype)
            dones = np.zeros((B, L), dtype=bool)

            # D. Fill Buffer
            for i, epi_idx in enumerate(batch_idxs):
                start, end = self.episodes[epi_idx]
                actual_len = end - start

                states[i, :actual_len] = self.states[start:end]
                actions[i, :actual_len] = self.actions[start:end]
                next_states[i, :actual_len] = self.next_states[start:end]
                dones[i, :actual_len] = self.dones[start:end]

            yield states, actions, next_states, dones, batch_lengths

    # ------------------------------------------------------------
    # Epoch Iterators (NEW)
    # ------------------------------------------------------------
    def iterate_transitions(
        self,
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: Optional[int] = None,
        drop_last: bool = False,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Iterate over ALL transitions exactly once per epoch (no replacement).
        Yields: (states, actions, next_states, dones) with shape (B, ...)
        """
        self._consolidate_if_needed()

        idxs = (
            self._get_transition_indices()
        )  # cached, respects episodes :contentReference[oaicite:2]{index=2}
        if idxs.size == 0:
            return

        if shuffle:
            rng = np.random.default_rng(seed)
            idxs = idxs[rng.permutation(idxs.size)]

        N = idxs.size
        end = (N // batch_size) * batch_size if drop_last else N

        for i in range(0, end, batch_size):
            batch = idxs[i : i + batch_size]
            yield (
                self.states[batch],
                self.actions[batch],
                self.next_states[batch],
                self.dones[batch],
            )

    def iterate_windows(
        self,
        horizon: int,
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: Optional[int] = None,
        allow_short: bool = False,
        drop_last: bool = False,
    ) -> Iterator[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ]:
        """
        Iterate over fixed-horizon windows within episodes exactly once per epoch.

        If allow_short=False:
          Only yields full windows of length == horizon.

        If allow_short=True:
          Includes ONE window per episode start even if the episode is shorter than horizon.
          Short windows are padded to horizon and accompanied by a boolean mask.

        Yields:
          (states, actions, next_states, dones, mask, valid_lengths)
            states/actions/next_states: (B, H, dim)
            dones/mask:               (B, H)
            valid_lengths:            (B,)
        """
        self._consolidate_if_needed()
        assert horizon >= 1

        starts: list[int] = []
        lens: list[int] = []

        for s, e in self.episodes:
            L = e - s
            if L <= 0:
                continue

            if allow_short:
                # one window per episode start, length=min(L, H)
                starts.append(s)
                lens.append(min(L, horizon))
            else:
                # all full windows
                if L >= horizon:
                    for off in range(0, L - horizon + 1):
                        starts.append(s + off)
                        lens.append(horizon)

        if not starts:
            return

        starts = np.asarray(starts, dtype=np.int64)
        lens = np.asarray(lens, dtype=np.int32)

        if shuffle:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(starts.size)
            starts = starts[perm]
            lens = lens[perm]

        N = starts.size
        end = (N // batch_size) * batch_size if drop_last else N

        s_dim = self.states.shape[1]
        a_dim = self.actions.shape[1]
        H = horizon

        for i in range(0, end, batch_size):
            batch_starts = starts[i : i + batch_size]
            batch_lens = lens[i : i + batch_size]
            B = batch_starts.shape[0]

            states = np.zeros((B, H, s_dim), dtype=self.states.dtype)
            actions = np.zeros((B, H, a_dim), dtype=self.actions.dtype)
            next_states = np.zeros((B, H, s_dim), dtype=self.next_states.dtype)
            dones = np.zeros((B, H), dtype=bool)
            mask = np.zeros((B, H), dtype=bool)

            for b in range(B):
                st = int(batch_starts[b])
                ln = int(batch_lens[b])
                en = st + ln

                states[b, :ln] = self.states[st:en]
                actions[b, :ln] = self.actions[st:en]
                next_states[b, :ln] = self.next_states[st:en]
                dones[b, :ln] = self.dones[st:en]
                mask[b, :ln] = True

            yield states, actions, next_states, dones, mask, batch_lens

    def iterate_mixed_epoch(
        self,
        *,
        transition_batch_size: int,
        window_batch_size: int,
        horizon: int,
        transition_per_window: int = 4,
        shuffle: bool = True,
        seed: Optional[int] = None,
        allow_short_windows: bool = False,
        schedule: Literal["round_robin", "two_phase"] = "round_robin",
    ):
        """
        Deterministic epoch iterator that mixes:
          - transition batches (1-step)
          - window batches (H-step)

        Yields tuples: (kind, batch)
          kind == "transition": (s, a, s_next, done)
          kind == "window":     (S, A, S_next, done, mask, lengths)
        """
        self._consolidate_if_needed()

        trans_it = self.iterate_transitions(
            transition_batch_size,
            shuffle=shuffle,
            seed=None if seed is None else seed + 101,
        )
        win_it = self.iterate_windows(
            horizon,
            window_batch_size,
            shuffle=shuffle,
            seed=None if seed is None else seed + 202,
            allow_short=allow_short_windows,
        )

        if schedule == "two_phase":
            for batch in trans_it:
                yield "transition", batch
            for batch in win_it:
                yield "window", batch
            return

        # round_robin
        trans_exhausted = False
        win_exhausted = False

        while not (trans_exhausted and win_exhausted):
            # transitions
            for _ in range(transition_per_window):
                if trans_exhausted:
                    break
                try:
                    yield "transition", next(trans_it)
                except StopIteration:
                    trans_exhausted = True
                    break

            # window
            if not win_exhausted:
                try:
                    yield "window", next(win_it)
                except StopIteration:
                    win_exhausted = True
