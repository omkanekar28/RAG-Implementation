import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import faiss


class FaissDatabaseCreator:
    """
    A class to create the Faiss database for the recommendation system.
    """

    def __init__(self, input_excel_path: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """
        """
        self.input_df = pd.read_csv(input_excel_path)
        print(f"Dataset having {len(self.input_df)} rows loaded successfully")
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"Embedding model {embedding_model} loaded successfully")

    def create_faiss_database(self) -> None:
        """
        Creates the Faiss database for the recommendation system.
        """
        faiss_rows = []
        print(f"Performing text formatting on {len(self.input_df)} rows...")
        for index, row in self.input_df.iterrows():
            try:
                faiss_row = self.get_faiss_row_template(row)
                faiss_rows.append(faiss_row)
            except Exception as e:
                print(f"Error processing row {index}: {e}")
                continue
        print(f"Generating embeddings for {len(faiss_rows)} rows...")
        embeddings = self.embedding_model.encode(faiss_rows, show_progress_bar=True)
        
        # TODO: Save the embeddings in a FIASS index
        
    def get_faiss_row_template(self, df_row: pd.DataFrame) -> None:
        """
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
        input_excel_path="/home/omkanekar28/code/RAG-Implementation/data/cleaned_movie_dataset.csv"
    )
    database_creator.create_faiss_database()