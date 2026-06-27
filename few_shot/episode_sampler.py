"""
few_shot/episode_sampler.py

EpisodeSampler produces N-way K-shot episodes from any dataset that
exposes a class_to_indices dict mapping label_id -> [sample indices].

Both WaveformDataset and MammalDataset satisfy this interface.

An episode is:
    support: [N, K, 1, T]  -- K clips per class for prototype computation
    query:   [N, Q, 1, T]  -- Q clips per class to classify
    labels:  [N]           -- episode-local class ids 0..N-1
                              (not the global label_ids from the dataset)

Episode-local labels mean the classification head / distance comparison
always works over exactly N classes regardless of how many total classes
the dataset has.
"""

import random
import torch
from torch.utils.data import Dataset


class EpisodeSampler:
    """
    Samples few-shot episodes from a dataset.

    Args:
        dataset:    any Dataset with class_to_indices: dict[int, list[int]]
                    and __getitem__ returning (Tensor[1, T], label_id)
        n_way:      number of classes per episode
        k_shot:     number of support clips per class
        n_query:    number of query clips per class
        n_episodes: how many episodes to generate per "epoch"
        seed:       optional fixed seed for reproducible eval episodes
    """

    def __init__(
        self,
        dataset: Dataset,
        n_way: int = 5,
        k_shot: int = 5,
        n_query: int = 10,
        n_episodes: int = 600,
        seed: int = None,
    ):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        self.rng = random.Random(seed)

        # filter out classes that don't have enough clips for K support + Q query
        min_clips = k_shot + n_query
        self.eligible_classes = [
            label_id
            for label_id, indices in dataset.class_to_indices.items()
            if len(indices) >= min_clips
        ]

        if len(self.eligible_classes) < n_way:
            raise ValueError(
                f"Not enough eligible classes for {n_way}-way episodes. "
                f"Need at least {n_way} classes with >={min_clips} clips each, "
                f"but only {len(self.eligible_classes)} qualify.\n"
                f"  Reduce n_way, k_shot, or n_query."
            )

        print(
            f"[EpisodeSampler] {n_way}-way {k_shot}-shot {n_query}-query "
            f"| {len(self.eligible_classes)} eligible classes "
            f"| {n_episodes} episodes"
        )

    def _sample_episode(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns one episode:
            support [N, K, 1, T]
            query   [N, Q, 1, T]
            labels  [N]           episode-local 0..N-1
        """
        # sample N classes without replacement
        chosen_classes = self.rng.sample(self.eligible_classes, self.n_way)

        support_clips = []  # will be [N, K, 1, T]
        query_clips = []    # will be [N, Q, 1, T]

        for label_id in chosen_classes:
            all_indices = self.dataset.class_to_indices[label_id]
            selected = self.rng.sample(all_indices, self.k_shot + self.n_query)

            support_indices = selected[:self.k_shot]
            query_indices = selected[self.k_shot:]

            # load clips -- each __getitem__ returns (Tensor[1, T], label_id)
            s_clips = torch.stack([self.dataset[i][0] for i in support_indices])
            q_clips = torch.stack([self.dataset[i][0] for i in query_indices])

            support_clips.append(s_clips)   # [K, 1, T]
            query_clips.append(q_clips)     # [Q, 1, T]

        support = torch.stack(support_clips)    # [N, K, 1, T]
        query   = torch.stack(query_clips)      # [N, Q, 1, T]
        labels  = torch.arange(self.n_way)      # [N] episode-local 0..N-1

        return support, query, labels

    def __iter__(self):
        """Yields n_episodes episodes. Reshuffle order each time."""
        for _ in range(self.n_episodes):
            yield self._sample_episode()

    def __len__(self) -> int:
        return self.n_episodes
