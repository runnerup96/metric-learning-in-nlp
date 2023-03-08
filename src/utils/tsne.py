from typing import List, Dict


def get_indexes_of_samples(samples: List[str]) -> Dict[str, List[int]]:
    """Group samples by value evaluate their indexes of
    occurrence in the sequence.
    Example:
        >>> get_indexes_of_samples(['a', 'b', 'c', 'c'])
        >>> {'c': [2, 3], 'a': [0], 'b': [1]}

    Args:
        samples (List[str]): in practice , this sequence corresponds
            to the classes

    Returns:
        Dict[str, List[int]]: mapping from unique sample to array of its
            occurence indexes in the input sequence
    """
    return {
        label: [i for i, val in enumerate(samples) if val == label]
        for label in set(samples)
    }
