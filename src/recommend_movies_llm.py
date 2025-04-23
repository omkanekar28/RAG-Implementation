import time
import json
import pandas as pd
from recommend_movies import MovieRecommender
from models import GGUFModelHandler
from prompts import SYSTEM_PROMPT, get_llm_prompt
from config import EMBEDDING_MODEL, FAISS_STORE_DIR, LLM_PATH


class MovieRecommenderLLM(MovieRecommender):
    """
    """
    def __init__(self, embedding_model: str, faiss_store_dir: str, model_path: str):
        """
        """
        super().__init__(embedding_model=embedding_model, faiss_store_dir=faiss_store_dir)
        self.model_path = model_path

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

Distance from the query: {df_row['distance']}
"""
    
    def get_recommendations_llm(self, query: str, top_n: int) -> str:
        """
        """
        movies_df = self.get_recommendations(query, top_n)
        movies = ""

        for _, row in movies_df.iterrows():
            movies += "\n\n" + self.get_faiss_row_template(row)

        model_loading_start_time = time.time()
        print("Loading the model...")
        llm = GGUFModelHandler(
            llm_checkpoint=self.model_path,
            context_window_size=1000 + (500 * len(movies_df)),    # 500 tokens for each movie and 1000 for the prompt
            max_tokens=100 * len(movies_df)    # 100 tokens for each movie
        )
        print(f"Model loaded in {time.time() - model_loading_start_time:.2f} seconds.\n")
        instruction_prompt = get_llm_prompt(
            user_description=query,
            movies=movies,
            n=top_n
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction_prompt}
        ]
        inference_start_time = time.time()
        print("Performing inference...")
        output = llm(messages)
        print(f"LLM inference completed in {time.time() - inference_start_time:.2f} seconds.\n")
        return output


if __name__ == "__main__":
    recommender = MovieRecommenderLLM(
        embedding_model=EMBEDDING_MODEL,
        faiss_store_dir=FAISS_STORE_DIR,
        model_path=LLM_PATH
    )

    query = input("Enter a movie description for recommendations: ")
    number_of_recommendations = int(input("Enter the number of recommendations you want: "))
    response = recommender.get_recommendations_llm(query, top_n=number_of_recommendations)
    print(response)
