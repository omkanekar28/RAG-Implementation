import os
import json
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, INPUT_EXCEL_PATH, FAISS_STORE_DIR


class FaissDatabaseCreator:
    """
    A class to create the Faiss database for the recommendation system.
    """

    def __init__(self, input_excel_path: str, embedding_model: str) -> None:
        """
        Initializes the input data and embedding model.
        """
        self.input_df = pd.read_csv(input_excel_path)
        print(f"Dataset having {len(self.input_df)} rows loaded successfully")
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"Embedding model {embedding_model} loaded successfully")

    def create_faiss_database(self, store_dir: str) -> None:
        """
        Creates the Faiss database for the recommendation system.
        """
        input_rows = []
        faiss_rows = []
        print(f"Performing text formatting on {len(self.input_df)} rows...")
        for index, row in self.input_df.iterrows():
            try:
                faiss_row = self.get_faiss_row_template(row)
                input_rows.append(row)
                faiss_rows.append(faiss_row)
            except Exception as e:
                print(f"Error processing row {index}: {e}")
                continue
        print(f"Generating embeddings for {len(faiss_rows)} rows...")
        embeddings = self.embedding_model.encode(faiss_rows, show_progress_bar=True)
        
        print("Initializing FAISS index...")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)  # L2 distance; can also use IndexFlatIP for cosine similarity
        index.add(embeddings)

        print("Saving FAISS index and metadata...")
        os.makedirs(store_dir, exist_ok=True)
        faiss.write_index(index, os.path.join(store_dir, "movies.index"))

        # Save the corresponding metadata (e.g., movie titles or IDs)
        pd.DataFrame(input_rows).to_pickle(os.path.join(store_dir, "movies_metadata.pkl"))

        print("FAISS database creation complete.")
        
    def get_faiss_row_template(self, df_row: pd.DataFrame) -> str:
        """
        Returns a formatted string for the FAISS row template.
        """
        return f"""{df_row['title']}

{df_row['tagline']}

{df_row['overview']}

Genres:- {df_row['genres']}
Keywords:- {df_row['keywords']}
Production Companies:- {", ".join(company["name"] for company in json.loads(df_row['production_companies']))}
Production Countries:- {", ".join(country["name"] for country in json.loads(df_row['production_countries']))}
Spoken Languages:- {", ".join(language["name"] for language in json.loads(df_row['spoken_languages']))}
Cast:- {df_row['cast']}
Director:- {df_row['director']}
Release Date:- {df_row['release_date']}
Runtime:- {round(int(df_row['runtime']) / 60, 1)} hours
"""


if __name__ == "__main__":
    database_creator = FaissDatabaseCreator(
        input_excel_path=INPUT_EXCEL_PATH,
        embedding_model=EMBEDDING_MODEL
    )
    database_creator.create_faiss_database(store_dir=FAISS_STORE_DIR)