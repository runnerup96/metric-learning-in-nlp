import torch
import random
import numpy as np
from typing import Dict, List, Optional


class TripletGenerator:
    def __init__(
        self,
        class2idx_dict: Dict[str, int],
        attention_storage: torch.Tensor,
        token_ids_storage: torch.Tensor,
    ):
        self.class2idx_dict = class2idx_dict
        self.attention_storage = attention_storage
        self.token_ids_storage = token_ids_storage

        self.classes = list(self.class2idx_dict.keys())

    @staticmethod
    def random_choose(
        sequence: List[int], exclude_sample: Optional[int], max_retires: int = 5
    ) -> int:
        """Randomly sample index from sequence. If exclude_sample is not None,
        then try to make sample != exclude_sample withing max_retires retries.

        Args:
            sequence (List[int]): _description_
            exclude_sample (Optional[int]): _description_. Defaults to None.
            retires (int, optional): _description_. Defaults to 5.

        Returns:
            int: random sample from sequence
        """
        for _ in range(max_retires):
            choice = np.random.choice(sequence)
            if exclude_sample is not None and choice == exclude_sample:
                continue
            else:
                break
        return choice

    def get_triplet(self) -> Dict[str, List[torch.Tensor]]:
        """Triplet builds as follows:
            * randomly select anchor class,
            * randomly select anchor from anchor class,
            * randomly select positive sample from anchor class
                (note: anchor sample != positive sample),
            * randomly select negative class,
            * randomly select negative sample from negative class
                (note: negative class != anchor class)

        Returns:
            Dict[str, List[torch.Tensor]]: mapping from anchor/positive/negative
                to Tuple of tokens_ids and attention mask.
        """
        anchor_class = np.random.choice(self.classes)
        anchor_sample_idx = np.random.choice(self.class2idx_dict[anchor_class])
        positive_sample_idx = TripletGenerator.random_choose(
            self.class2idx_dict[anchor_class], anchor_sample_idx
        )
        negative_class = TripletGenerator.random_choose(self.classes, anchor_class)
        negative_sample_idx = random.choice(self.class2idx_dict[negative_class])

        return {
            "anchor": [
                self.token_ids_storage[anchor_sample_idx],
                self.attention_storage[anchor_sample_idx],
            ],
            "positive": [
                self.token_ids_storage[positive_sample_idx],
                self.attention_storage[positive_sample_idx],
            ],
            "negative": [
                self.token_ids_storage[negative_sample_idx],
                self.attention_storage[negative_sample_idx],
            ],
        }

    def __call__(self, n_triplets: int) -> Dict[str, Dict[str, torch.Tensor]]:
        """Generate n_triplets

        Args:
            n_triplets (int): number of triplets to generate

        Returns:
            Dict[str, List[torch.Tensor]]: mapping from anchor/positive/negative
                to mapping from tokens_ids/attention_mask to tensors
        """
        (
            anchor_token_ids,
            positive_token_ids,
            negative_token_ids,
            anchor_attention_mask,
            positive_attention_mask,
            negative_attention_mask,
        ) = ([], [], [], [], [], [])
        for _ in range(n_triplets):
            sample_dict = self.get_triplet()
            anchor_token_ids.append(sample_dict["anchor"][0])
            anchor_attention_mask.append(sample_dict["anchor"][1])
            positive_token_ids.append(sample_dict["positive"][0])
            positive_attention_mask.append(sample_dict["positive"][1])
            negative_token_ids.append(sample_dict["negative"][0])
            negative_attention_mask.append(sample_dict["negative"][1])

        return {
            "anchor": {
                "token_ids": torch.vstack(anchor_token_ids),
                "attention_mask": torch.vstack(anchor_attention_mask),
            },
            "positive": {
                "token_ids": torch.vstack(positive_token_ids),
                "attention_mask": torch.vstack(positive_attention_mask),
            },
            "negative": {
                "token_ids": torch.vstack(negative_token_ids),
                "attention_mask": torch.vstack(negative_attention_mask),
            },
        }
