import torch
import numpy as np
from typing import Callable, Dict
import sklearn
import random
from tqdm import tqdm


def l2_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    distance_matrix = torch.cdist(a, b, p=2)
    diag_sum = torch.sum(torch.diagonal(distance_matrix))
    return diag_sum


def compute_triplet_loss(
    anchor_sample: torch.Tensor,
    pos_sample: torch.Tensor,
    neg_sample: torch.Tensor,
    distance_metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str,
    delta=1,
) -> torch.Tensor:
    """Calculate triplet loss, for more visit
    https://towardsdatascience.com/triplet-loss-advanced-intro-49a07b7d8905

    Returns:
        torch.Tensor: loss
    """
    pos_sim = distance_metric(anchor_sample, pos_sample)
    neg_sim = distance_metric(anchor_sample, neg_sample)
    loss = torch.max(torch.Tensor([0]).to(device), delta - neg_sim + pos_sim)
    return loss


def compute_silhouette_score(
    number_of_classes: int,
    target_indexes_dict: Dict[str, int],
    token_ids_array: torch.Tensor,
    attention_mask_array: torch.Tensor,
    encoder: torch.nn.Module,
) -> float:
    """Compute silhouette score using encoder and given data

    Args:
        number_of_classes (int): Amount of classes to use in the subset
        target_indexes_dict (Dict[str, int]): mapping from class to index
        token_ids_array (torch.Tensor): prompts tokens
        attention_mask_array (torch.Tensor): attention masks for tokens
        encoder (torch.nn.Module): encoder to use

    Returns:
        float: silhouette score
    """
    random.seed(42)
    targets = random.sample(list(target_indexes_dict.keys()), number_of_classes)
    embeddings_list, targets_list = [], []

    for target in tqdm(targets):
        samples_idx = target_indexes_dict[target]
        np.random.shuffle(samples_idx)
        samples_idx = samples_idx[: int(len(samples_idx) * 0.1)]
        token_ids, attention_masks = (
            token_ids_array[samples_idx],
            attention_mask_array[samples_idx],
        )
        encoder.eval()
        with torch.no_grad():
            embeddings = (
                encoder(token_ids, attention_mask=attention_masks)["pooler_output"]
                .cpu()
                .detach()
                .numpy()
            )
        embeddings_list.append(embeddings)
        targets_list += [target] * len(samples_idx)
    embeddings_numpy = np.concatenate(embeddings_list, axis=0)
    silhouette_score = sklearn.metrics.silhouette_score(embeddings_numpy, targets_list)
    return silhouette_score
