import torch
from typing import Callable, Any, Dict
import pandas as pd
from tqdm import tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np


from src.losses.losses import l2_distance
from src.losses.losses import compute_triplet_loss
from src.losses.losses import compute_silhouette_score

DELTA = 0.1
LAYERS2FREEZE = 9
EPOCH_STEPS = 256
VAL_STEPS = 32
BATCH_SIZE = 8


def fit(
    dataset_passes: int,
    device: str,
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
    ewma = lambda x, span: pd.DataFrame({"x": x})["x"].ewm(span=span).mean().values

    try:
        for epoch in range(dataset_passes):
            torch.cuda.empty_cache()

            # training
            epoch_train_loss = 0
            model.enable_some_bert_layers_training(layers2freeze=LAYERS2FREEZE)

            pbar = tqdm(
                total=EPOCH_STEPS, desc=f"Training, epoch {epoch}/{dataset_passes}"
            )
            for _ in range(EPOCH_STEPS):
                batch_sample = train_generator(BATCH_SIZE)
                optimizer.zero_grad()
                train_anchor_proj, train_pos_proj, train_neg_proj = model.forward(
                    batch_sample["anchor"],
                    batch_sample["positive"],
                    batch_sample["negative"],
                )
                batch_train_loss = compute_triplet_loss(
                    anchor_sample=train_anchor_proj,
                    pos_sample=train_pos_proj,
                    neg_sample=train_neg_proj,
                    distance_metric=l2_distance,
                    device=device,
                )
                batch_train_loss.backward()
                optimizer.step()
                epoch_train_loss += batch_train_loss.item()
                pbar.update(1)

            # Average Loss per sample
            train_loss_history_array.append(
                epoch_train_loss / (EPOCH_STEPS * BATCH_SIZE)
            )
            epoch_train_silouhette = compute_silhouette_score(
                10,
                train_target_indexes_dict,
                train_token_ids_array,
                train_attention_mask_array,
                model.bert,
            )
            train_silouhette_history_array.append(epoch_train_silouhette)

            clear_output(True)
            plt.figure(figsize=[28, 10])

            # loss plot
            plt.subplot(2, 2, 1), plt.title("train triplet loss"), plt.grid()
            plt.scatter(
                np.arange(len(train_loss_history_array)),
                train_loss_history_array,
                alpha=0.1,
            )
            plt.plot(ewma(train_loss_history_array, span=10))

            # silouhette plot
            plt.subplot(2, 2, 2), plt.title("train silouhette metric"), plt.grid()
            plt.scatter(
                np.arange(len(train_silouhette_history_array)),
                train_silouhette_history_array,
                alpha=0.1,
            )
            plt.plot(ewma(train_silouhette_history_array, span=10))

            # check gradient flow
            layers_list, average_grads_list = [], []
            for layer_name, params in model.named_parameters():
                if (params.requires_grad) and ("bias" not in layer_name):
                    layers_list.append(layer_name)
                    average_grads_list.append(params.grad.abs().mean().item())

            plt.subplot(2, 2, 3)
            plt.title("training average grads per sample pass")
            plt.grid()
            plt.scatter(layers_list, average_grads_list, alpha=0.1)
            plt.xticks(
                range(0, len(average_grads_list), 1), layers_list, rotation="vertical"
            )
            plt.plot(ewma(average_grads_list, span=10))
            plt.show()
            print("Train silouhette value: ", train_silouhette_history_array[-1])

            # disable bert training for all layers
            model.disable_bert_training()
            model.eval()

            epoch_val_loss = 0
            epoch_val_recall = 0
            pbar.close()
            pbar = tqdm(total=VAL_STEPS, desc="Validation")
            for _ in range(VAL_STEPS):
                batch_sample = val_generator(2)
                with torch.no_grad():
                    val_anchor_proj, val_pos_proj, val_neg_proj = model.forward(
                        batch_sample["anchor"],
                        batch_sample["positive"],
                        batch_sample["negative"],
                    )
                    batch_val_loss = compute_triplet_loss(
                        anchor_sample=val_anchor_proj,
                        pos_sample=val_pos_proj,
                        neg_sample=val_neg_proj,
                        distance_metric=l2_distance,
                        device=device,
                    )
                    epoch_val_loss += batch_val_loss.item()
                    pbar.update(1)

                # calculate recall - our anchor is closer to pos than to neg
                anchor2pos_distance = l2_distance(val_anchor_proj, val_pos_proj)
                anchor2neg_distance = l2_distance(val_anchor_proj, val_neg_proj)

                if anchor2neg_distance > anchor2pos_distance + DELTA:
                    epoch_val_recall += 1
            pbar.close()

            # Average loss per sample
            val_loss_history_array.append(epoch_val_loss / (VAL_STEPS * BATCH_SIZE))
            # Recall per one epoch pass
            val_recall_history_array.append(epoch_val_recall / VAL_STEPS)

            clear_output(True)
            plt.figure(figsize=[20, 6])
            plt.subplot(1, 2, 1), plt.title("val triplet loss"), plt.grid()
            plt.scatter(
                np.arange(len(val_loss_history_array)),
                val_loss_history_array,
                alpha=0.1,
            )
            plt.plot(ewma(val_loss_history_array, span=10))

            # how many samples did we correctly arranged in space currently
            plt.subplot(1, 2, 2), plt.title(
                "Val Recall(one point - recall per epoch)"
            ), plt.grid()
            dev_time = np.arange(1, len(val_recall_history_array) + 1)
            plt.scatter(dev_time, val_recall_history_array, alpha=0.1)
            plt.plot(dev_time, ewma(val_recall_history_array, span=10))
            plt.show()

    except KeyboardInterrupt:
        pass
