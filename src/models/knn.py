from typing import List, Union
import faiss
import numpy as np


class KnnIndex:
    def __init__(
        self, embeddings: np.ndarray, classes_array: List[str], to_retrieve: int = 1
    ):
        """KNN Index Searcher

        Args:
            embeddings (np.ndarray): embeddings correspond to classes array
            classes_array (List[str]): classes array corresponds to embeddings
            to_retrieve (int, optional): Number of closest classes to retrieve
                when searching neighbours. Defaults to 1.
        """
        self.to_retrieve = to_retrieve
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.train(embeddings)
        self.index.add(embeddings)
        self.classes_array = classes_array

    def search_index(self, vectorized_query: np.ndarray) -> Union[str, List[str]]:
        """Search closest class to given vectorized query

        Args:
            vectorized_query (np.ndarray): encoded embedding of shape [1, D],
                , where D is the dimension of KNN

        Returns:
            Union[str, List[str]]: closest class to given vector
        """
        assert vectorized_query.shape[0] == 1, "Query vector must be flat"
        dense_search_results = self.index.search(vectorized_query, self.to_retrieve)
        dense_index = dense_search_results[1][0][0]
        closest_class = self.classes_array[dense_index]
        return closest_class
