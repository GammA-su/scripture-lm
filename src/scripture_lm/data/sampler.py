"""Dual sampling strategies (natural and temperature) with exact exposure accounting."""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
from torch.utils.data import Sampler

from scripture_lm.data.chunk_index import ChunkMetadata


def calculate_temperature_parameters(
    family_chunk_counts: dict[str, int],
    family_raw_chars: dict[str, int],
    alpha: float,
) -> dict[str, dict[str, float]]:
    """Compute exact mathematical temperature parameters for each family.

    Args:
        family_chunk_counts: Mapping from family name to number of chunks K_i.
        family_raw_chars: Mapping from family name to total natural target characters n_i.
        alpha: Temperature exponent in [0.0, 1.0].

    Returns:
        Mapping of family -> {
            "natural_raw_chars": float,
            "natural_raw_share": float,
            "target_exposure_share": float (r_i),
            "mean_chunk_chars": float (mu_i),
            "family_draw_probability": float (q_i),
        }
    """
    families = sorted(family_chunk_counts.keys())
    total_n = sum(family_raw_chars[f] for f in families)

    # Calculate target exposure shares: r_i = n_i^alpha / sum(n_j^alpha)
    if total_n == 0:
        weights = {f: 1.0 for f in families}
    elif alpha == 0.0:
        weights = {f: 1.0 for f in families}
    else:
        weights = {f: float(family_raw_chars[f]) ** alpha for f in families}
    sum_w = sum(weights.values())
    target_shares = {f: weights[f] / sum_w for f in families}

    # Mean target characters per chunk: mu_i = n_i / K_i
    mean_chunk_chars: dict[str, float] = {}
    for f in families:
        k = family_chunk_counts[f]
        mean_chunk_chars[f] = family_raw_chars[f] / k if k > 0 else 1.0

    # Draw probability adjusted for chunk size: q_i \propto r_i / mu_i
    raw_draw_probs = {
        f: (target_shares[f] / mean_chunk_chars[f] if mean_chunk_chars[f] > 0 else 0.0)
        for f in families
    }
    sum_q = sum(raw_draw_probs.values())
    draw_probs = {
        f: (raw_draw_probs[f] / sum_q if sum_q > 0 else 1.0 / len(families)) for f in families
    }

    result: dict[str, dict[str, float]] = {}
    for f in families:
        result[f] = {
            "natural_raw_chars": float(family_raw_chars[f]),
            "natural_raw_share": family_raw_chars[f] / total_n if total_n > 0 else 0.0,
            "target_exposure_share": target_shares[f],
            "mean_chunk_chars": mean_chunk_chars[f],
            "family_draw_probability": draw_probs[f],
        }
    return result


def simulate_sampling(
    family_chunk_counts: dict[str, int],
    family_raw_chars: dict[str, int],
    alpha: float,
    draws: int = 100_000,
    seed: int = 1337,
) -> dict[str, dict[str, Any]]:
    """Simulate finite draws from temperature sampling to measure observed exposure.

    Returns:
        Mapping of family -> {
            "observed_draws": int,
            "observed_draw_pct": float,
            "observed_raw_chars": int,
            "observed_raw_pct": float,
        }
    """
    params = calculate_temperature_parameters(family_chunk_counts, family_raw_chars, alpha)
    families = sorted(family_chunk_counts.keys())
    draw_p = [params[f]["family_draw_probability"] for f in families]

    rng = np.random.default_rng(seed)
    chosen_families = rng.choice(families, size=draws, p=draw_p)

    draw_counts = {f: 0 for f in families}
    for f in chosen_families:
        draw_counts[f] += 1

    # Approximate raw characters drawn based on mean chunk size
    # (Or simulated with uniform chunk choice if chunk sizes differ)
    raw_counts = {f: draw_counts[f] * params[f]["mean_chunk_chars"] for f in families}
    total_raw = sum(raw_counts.values())

    result: dict[str, dict[str, Any]] = {}
    for f in families:
        result[f] = {
            "observed_draws": draw_counts[f],
            "observed_draw_pct": draw_counts[f] / draws if draws > 0 else 0.0,
            "observed_raw_chars": raw_counts[f],
            "observed_raw_pct": raw_counts[f] / total_raw if total_raw > 0 else 0.0,
        }
    return result


class NaturalSampler(Sampler[int]):
    """Without-replacement traversal of all chunks, shuffled deterministically each epoch."""

    def __init__(
        self,
        chunks: list[ChunkMetadata],
        seed: int = 1337,
        epoch: int = 0,
    ) -> None:
        super().__init__()
        self.chunks = chunks
        self.seed = seed
        self.epoch = epoch
        self.position_in_epoch = 0
        self.current_permutation: list[int] = []
        self._generate_permutation()

    def _generate_permutation(self) -> None:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.current_permutation = rng.permutation(len(self.chunks)).tolist()

    def set_epoch(self, epoch: int) -> None:
        """Advance to given epoch and reshuffle."""
        self.epoch = epoch
        self.position_in_epoch = 0
        self._generate_permutation()

    def __iter__(self) -> Iterator[int]:
        while self.position_in_epoch < len(self.current_permutation):
            idx = self.current_permutation[self.position_in_epoch]
            self.position_in_epoch += 1
            yield idx

        # When traversal finishes, automatically advance epoch
        self.epoch += 1
        self.position_in_epoch = 0
        self._generate_permutation()

    def __len__(self) -> int:
        return len(self.current_permutation) - self.position_in_epoch

    def state_dict(self) -> dict[str, Any]:
        """Serialize sampler state for exact checkpoint resumption."""
        return {
            "epoch": self.epoch,
            "seed": self.seed,
            "position_in_epoch": self.position_in_epoch,
            "current_permutation": list(self.current_permutation),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore sampler state from checkpoint."""
        self.epoch = int(state_dict["epoch"])
        self.seed = int(state_dict["seed"])
        self.position_in_epoch = int(state_dict["position_in_epoch"])
        self.current_permutation = list(state_dict["current_permutation"])


class TemperatureSampler(Sampler[int]):
    """With-replacement temperature sampling over scripture families with exposure budgeting."""

    def __init__(
        self,
        chunks: list[ChunkMetadata],
        alpha: float = 0.5,
        seed: int = 1337,
    ) -> None:
        super().__init__()
        self.chunks = chunks
        self.alpha = float(alpha)
        self.seed = seed

        # Group chunk indices by family
        self.family_indices: dict[str, list[int]] = {}
        for idx, chunk in enumerate(chunks):
            self.family_indices.setdefault(chunk.family, []).append(idx)

        self.families = sorted(self.family_indices.keys())

        # Compute n_i and K_i
        family_chunk_counts = {f: len(self.family_indices[f]) for f in self.families}
        family_raw_chars = {
            f: sum(self.chunks[i].raw_character_count for i in self.family_indices[f])
            for f in self.families
        }
        self.N = sum(family_raw_chars.values())

        # Derive draw probabilities q_i
        params = calculate_temperature_parameters(family_chunk_counts, family_raw_chars, self.alpha)
        self.family_params = params
        self.draw_probs = [params[f]["family_draw_probability"] for f in self.families]
        self.expected_chars_per_draw = sum(
            params[f]["family_draw_probability"] * params[f]["mean_chunk_chars"]
            for f in self.families
        )

        # Persistent counters across the entire experiment
        self.effective_epochs_completed = 0
        self.cumulative_raw_chars = 0
        self.cumulative_model_tokens = 0
        self.draw_count = 0

        self.rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[int]:
        # Effective epoch boundary target: (effective_epochs_completed + 1) * N
        target_exposure = (self.effective_epochs_completed + 1) * self.N

        while self.cumulative_raw_chars < target_exposure:
            # 1. Sample family from categorical distribution q
            fam_idx = self.rng.choice(len(self.families), p=self.draw_probs)
            chosen_family = self.families[fam_idx]

            # 2. Sample chunk uniformly at random from that family's pool
            pool = self.family_indices[chosen_family]
            chunk_pool_idx = self.rng.choice(len(pool))
            chunk_idx = pool[chunk_pool_idx]

            chunk = self.chunks[chunk_idx]
            self.cumulative_raw_chars += chunk.raw_character_count
            self.cumulative_model_tokens += chunk.target_token_count
            self.draw_count += 1

            yield chunk_idx

        # Completed one effective epoch (carrying forward any overshoot)
        self.effective_epochs_completed += 1

    def __len__(self) -> int:
        target_exposure = (self.effective_epochs_completed + 1) * self.N
        remaining_chars = max(0, target_exposure - self.cumulative_raw_chars)
        if self.expected_chars_per_draw > 0 and remaining_chars > 0:
            return max(1, int(round(remaining_chars / self.expected_chars_per_draw)))
        return 1

    def state_dict(self) -> dict[str, Any]:
        """Serialize sampler state for exact checkpoint resumption."""
        return {
            "seed": self.seed,
            "alpha": self.alpha,
            "effective_epochs_completed": self.effective_epochs_completed,
            "cumulative_raw_chars": self.cumulative_raw_chars,
            "cumulative_model_tokens": self.cumulative_model_tokens,
            "draw_count": self.draw_count,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore sampler state from checkpoint."""
        self.seed = int(state_dict["seed"])
        self.alpha = float(state_dict["alpha"])
        self.effective_epochs_completed = int(state_dict["effective_epochs_completed"])
        self.cumulative_raw_chars = int(state_dict["cumulative_raw_chars"])
        self.cumulative_model_tokens = int(state_dict["cumulative_model_tokens"])
        self.draw_count = int(state_dict["draw_count"])
        self.rng.bit_generator.state = state_dict["rng_state"]


class SequentialSampler(Sampler[int]):
    """Deterministic sequential sampler for validation and test splits."""

    def __init__(self, data_source: list[ChunkMetadata]) -> None:
        super().__init__()
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        yield from range(len(self.data_source))

    def __len__(self) -> int:
        return len(self.data_source)
