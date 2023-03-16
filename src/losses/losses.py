import torch
import random
import sklearn
import numpy as np
from tqdm import tqdm
from typing import Callable, Dict, Optional, Literal


def l2_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    dist = (a - b).pow(2).sum(1)
    return dist


def compute_triplet_loss(
    anchor_sample: torch.Tensor,
    pos_sample: torch.Tensor,
    neg_sample: torch.Tensor,
    distance_metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    margin: float,
) -> torch.Tensor:
    """Calculate triplet loss, for more visit
    https://towardsdatascience.com/triplet-loss-advanced-intro-49a07b7d8905

    Returns:
        torch.Tensor: loss
    """
    pos_sim = distance_metric(anchor_sample, pos_sample)
    neg_sim = distance_metric(anchor_sample, neg_sample)
    loss_batch = torch.nn.functional.relu(pos_sim - neg_sim + margin)
    loss = loss_batch.mean()
    return loss


def hard_triplet_loss(
    anchor_sample: torch.Tensor,
    pos_sample: torch.Tensor,
    neg_sample: torch.Tensor,
    distance_metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    margin: float,
    mining_type: Optional[Literal["easy", "hard", "semihard"]],
    abscence_strategy: Optional[Literal["ignore", "sample"]] = "sample",
    cut_strategy: Optional[Literal["soft-margin", "hard-margin"]] = "soft-margin",
) -> torch.Tensor:
    """Computes triplet loss with batch mining

    Args:
        anchor_sample (torch.Tensor): anchors, (bs, DIM)
        pos_sample (torch.Tensor): positives, (bs, DIM)
        neg_sample (torch.Tensor): negatives, (bs, DIM)
        distance_metric (...): Callable function to calculate distances
        margin (float): margin to use
        mining_type (...): whether to use hard or easy triplets for training
        abscence_strategy (Optional[]], optional): If none of batch samples are
            suitable for mining type. Defaults to "sample".
        cut_strategy (...): whether to use hard margin or soft margin, Defaults to "soft-margin".

    Returns:
        torch.Tensor: averaged batch loss
    """

    pos_sim = distance_metric(anchor_sample, pos_sample)
    neg_sim = distance_metric(anchor_sample, neg_sample)
    triplet_margin = neg_sim - pos_sim
    if mining_type == "easy":
        threshold_condition = triplet_margin > margin
    else:
        threshold_condition = triplet_margin <= margin
        if mining_type == "hard":
            threshold_condition &= triplet_margin <= 0
        elif mining_type == "semihard":
            threshold_condition &= triplet_margin > 0
        else:
            raise ValueError("Unknown passed value for mining_type")

    if not torch.any(threshold_condition):
        if abscence_strategy == "ignore":
            pass
        elif abscence_strategy == "sample":
            threshold_condition[0] = True

    anchor_sample = anchor_sample[threshold_condition]
    pos_sample = pos_sample[threshold_condition]
    neg_sample = neg_sample[threshold_condition]

    pos_sim = distance_metric(anchor_sample, pos_sample)
    neg_sim = distance_metric(anchor_sample, neg_sample)

    if cut_strategy == "hard-margin":
        loss_batch = torch.nn.functional.relu(pos_sim - neg_sim + margin)
    elif cut_strategy == "soft-margin":
        loss_batch = torch.log(1 + torch.exp(pos_sim - neg_sim))
    elif cut_strategy == "softmax":
        pass
    loss = loss_batch.mean()
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
