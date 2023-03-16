import math
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
from typing import Callable, Any, Dict, List, Optional

from losses.losses import l2_distance
from losses.losses import compute_triplet_loss, hard_triplet_loss
from losses.losses import compute_silhouette_score

DELTA = 0.1


def plot_metrics(
    metrics: Dict[str, List[float]],
    ewm: Optional[bool] = True,
    ncols: Optional[int] = 2,
    figsize: Optional[int] = (16, 12),
):
    """Plot metrics using provided mapping

    Args:
        metrics (Dict[str, List[float]]): mapping from name to its values
        ewm (Optional[bool], optional): whether to use moving average. Defaults to True.
        ncols (Optional[int], optional): number of columns. Defaults to 2.
        figsize (Optional[int], optional): size of result plt figure. Defaults to (16, 12).
    """
    ewma = lambda x, span: pd.DataFrame({"x": x})["x"].ewm(span=span).mean().values
    nrows = math.ceil(len(metrics) / ncols)
    fig, _axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axs = [item for sublist in _axs for item in sublist] if nrows > 1 else _axs
    for name, ax in zip(metrics.keys(), axs):
        ax.scatter(
            range(len(metrics[name])),
            metrics[name],
            alpha=0.1,
        )
        ax.plot(ewma(metrics[name], span=10) if ewm else metrics[name])
        ax.set_title(name)

    for ax in axs[len(metrics) :]:
        ax.axis("off")

    fig.show()


def fit(
    dataset_passes: int,
    train_epoch_steps: int,
    val_epoch_steps: int,
    batch_szie: int,
    triplet_margin: float,
    model: torch.nn.Module,
    optimizer: Any,
    train_generator: Callable,
    val_generator: Callable,
    train_target_indexes_dict: Dict[str, int],
    train_token_ids_array: torch.Tensor,
    train_attention_mask_array: torch.Tensor,
):
    train_loss_history_array, train_silouhette_history_array = [], []
    val_loss_history_array, val_recall_history_array = [], []

    try:
        for epoch in range(dataset_passes):
            torch.cuda.empty_cache()

            # training
            epoch_train_loss = 0
            model.enable_bert_layers_training()

            pbar = tqdm(
                total=train_epoch_steps,
                desc=f"Training, epoch {epoch}/{dataset_passes}",
            )
            for _ in range(train_epoch_steps):
                batch_sample = train_generator(batch_szie)
                optimizer.zero_grad()
                train_anchor_proj, train_pos_proj, train_neg_proj = model.forward(
                    batch_sample["anchor"],
                    batch_sample["positive"],
                    batch_sample["negative"],
                )

                batch_train_loss = hard_triplet_loss(
                    anchor_sample=train_anchor_proj,
                    pos_sample=train_pos_proj,
                    neg_sample=train_neg_proj,
                    distance_metric=l2_distance,
                    margin=triplet_margin,
                    mining_type="hard",
                )
                batch_train_loss.backward()
                optimizer.step()
                epoch_train_loss += batch_train_loss.item()
                pbar.update(1)

            # Average Loss per sample
            train_loss_history_array.append(
                epoch_train_loss / (train_epoch_steps * batch_szie)
            )
            epoch_train_silouhette = compute_silhouette_score(
                10,
                train_target_indexes_dict,
                train_token_ids_array,
                train_attention_mask_array,
                model.bert,
            )
            train_silouhette_history_array.append(epoch_train_silouhette)

            # check gradient flow
            layers_list, average_grads_list = [], []
            for layer_name, params in model.named_parameters():
                if (params.requires_grad) and ("bias" not in layer_name):
                    layers_list.append(layer_name)
                    average_grads_list.append(params.grad.abs().mean().item())

            plot_metrics(
                {
                    "train triplet loss": train_loss_history_array,
                    "train silouhette metric": train_silouhette_history_array,
                    "training average grads per sample pass": average_grads_list,
                },
                figsize=(10, 8),
            )

            # disable bert training for all layers
            model.disable_bert_training()
            model.eval()

            epoch_val_loss = 0
            epoch_val_recall = 0
            pbar.close()
            pbar = tqdm(total=val_epoch_steps, desc="Validation")
            for _ in range(val_epoch_steps):
                batch_sample = val_generator(2)
                with torch.no_grad():
                    val_anchor_proj, val_pos_proj, val_neg_proj = model.forward(
                        batch_sample["anchor"],
                        batch_sample["positive"],
                        batch_sample["negative"],
                    )
                    batch_val_loss = hard_triplet_loss(
                        anchor_sample=val_anchor_proj,
                        pos_sample=val_pos_proj,
                        neg_sample=val_neg_proj,
                        distance_metric=l2_distance,
                        margin=triplet_margin,
                        mining_type="hard",
                    )
                    epoch_val_loss += batch_val_loss.item()
                    pbar.update(1)

                # calculate recall - our anchor is closer to pos than to neg
                anchor2pos_distance = l2_distance(val_anchor_proj, val_pos_proj)
                anchor2neg_distance = l2_distance(val_anchor_proj, val_neg_proj)

                if (
                    torch.sum(anchor2neg_distance)
                    > torch.sum(anchor2pos_distance) + DELTA
                ):
                    epoch_val_recall += 1
            pbar.close()

            # Average loss per sample
            val_loss_history_array.append(
                epoch_val_loss / (val_epoch_steps * batch_szie)
            )
            # Recall per one epoch pass
            val_recall_history_array.append(epoch_val_recall / val_epoch_steps)

            clear_output(True)
            plot_metrics(
                {
                    "val triplet loss": val_loss_history_array,
                    "Val Recall(one point - recall per epoch)": val_recall_history_array,
                },
                figsize=(8, 3),
            )
            plt.show()

    except KeyboardInterrupt:
        pass
