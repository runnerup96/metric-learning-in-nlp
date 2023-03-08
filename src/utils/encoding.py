import torch
from tqdm import tqdm
from typing import Any, List, Tuple


def create_tranformer_input(
    sample_list: List[str], tokenizer: Any, device: str, max_length: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare samples list for Bert

    Args:
        sample_list (List[str]): _description_
        tokenizer (Any): tokenizer with API from transformers lib
        device (str): device where to put tensors
        max_length (int): the length to which all samples will be
            padded or truncated

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (tokens_ids, attention_mask)
    """
    tokens_ids_list, attention_masks_list = [], []
    for sample in sample_list:
        tokenized_sample = tokenizer.tokenize(sample)
        tokenized_sample = ["[CLS]"] + tokenized_sample + ["[SEP]"]
        if len(tokenized_sample) < max_length:
            tokenized_sample = tokenized_sample + ["[PAD]"] * (
                max_length - len(tokenized_sample)
            )
        else:
            tokenized_sample = tokenized_sample[: max_length - 1] + ["[PAD]"]

        tokens_ids = tokenizer.convert_tokens_to_ids(tokenized_sample)
        tokens_ids = torch.Tensor(tokens_ids).long().resize_((1, max_length)).to(device)
        attention_mask = (tokens_ids != 0).long().resize_((1, max_length)).to(device)
        tokens_ids_list.append(tokens_ids)
        attention_masks_list.append(attention_mask)

    tokens_ids = torch.cat(tokens_ids_list, axis=0)
    attention_masks = torch.cat(attention_masks_list, axis=0)
    return tokens_ids, attention_masks


def encode_data(
    text_samples: List[str],
    bert_encoder: torch.nn.Module,
    tokenizer: Any,
    device: str,
    max_length: int,
    batch_size: int = 256,
    verbose=True,
) -> torch.Tensor:
    """Get embeddings for text samples using bert_encoder and
    corresponding tokenizer.

    Args:
        text_samples (List[str]): text samples
        bert_encoder (torch.nn.Module): encoder
        tokenizer (Any): tokenizer with wich encoder was trained
        device (str): _description_
        max_length (int): the length to which all samples will be
            padded or truncated
        batch_size (int, optional): Defaults to 256
        verbose (bool, optional): Whether to display progress bar. Defaults to True.

    Returns:
        torch.Tensor: concatenated embeddings; axis=0 corresponds to each sample from
            text_samples
    """
    tokens_ids, attention_mask = create_tranformer_input(
        text_samples, tokenizer, device, max_length
    )
    encoded_data = []
    bert_encoder.eval()
    with torch.no_grad():
        if verbose:
            pbar = tqdm(total=len(tokens_ids) // batch_size, desc="Encoding data")

        for batch_number in range(0, tokens_ids.shape[0], batch_size):
            samples_vector = tokens_ids[batch_number : batch_number + batch_size]
            attention_vector = attention_mask[batch_number : batch_number + batch_size]
            outputs = bert_encoder(samples_vector, attention_mask=attention_vector)
            pooler_output = outputs["pooler_output"]
            encoded_data.append(pooler_output)

            if verbose:
                pbar.update(1)

    if verbose:
        pbar.close()

    encoded_data = torch.cat(encoded_data, dim=0)
    return encoded_data
