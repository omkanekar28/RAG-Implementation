import os
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, FAISS_STORE_DIR


class MovieRecommender:
    """
    A class to recommend movies based on a query using FAISS and Sentence Transformers.
    """

    def __init__(self, embedding_model: str, faiss_store_dir: str) -> None:
        """
        Initializes the embedding model, faiss index and metadeta.
        """
        print(f"Loading embedding model {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"Loading FAISS index and metadeta from {faiss_store_dir}...")
        self.index = faiss.read_index(os.path.join(faiss_store_dir, "movies.index"))
        self.metadata = pd.read_pickle(os.path.join(faiss_store_dir, "movies_metadata.pkl"))

    def get_recommendations(self, query: str, top_n: int) -> pd.DataFrame:
        """
        Returns the top N movie recommendations based on the query.
        """
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_embedding, top_n)
        results = self.metadata.iloc[indices[0]].copy()
        results["distance"] = distances[0]
        results.sort_values(by="distance",ascending=False , inplace=True)
        return results


if __name__ == "__main__":
    recommender = MovieRecommender(embedding_model=EMBEDDING_MODEL, faiss_store_dir=FAISS_STORE_DIR)
    query = input("Enter a movie description for recommendations: ")
    number_of_recommendations = int(input("Enter the number of recommendations you want: "))
    recommendations = recommender.get_recommendations(query, top_n=number_of_recommendations)
    print(recommendations[["title", "distance"]])