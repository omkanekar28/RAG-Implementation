from flask import Flask, render_template, request
from recommend_movies import MovieRecommender
from recommend_movies_llm import MovieRecommenderLLM
from config import EMBEDDING_MODEL, FAISS_STORE_DIR, LLM_PATH

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home() -> str:
    """
    Renders the home page of the application.
    """
    return render_template("index.html")

@app.route("/recommend_movies", methods=["GET", "POST"])
def recommend_movies() -> str:
    """
    Handles the recommendation of movies based on user input.
    """
    if request.method == "GET":
        return render_template("recommend_movies.html")
    
    if request.method == "POST":
        description = request.form.get("description")
        num_recommendations = int(request.form.get("num_recommendations"))
        recommender = MovieRecommender(embedding_model=EMBEDDING_MODEL, faiss_store_dir=FAISS_STORE_DIR)
        recommendations = recommender.get_recommendations(description, top_n=num_recommendations)
        recommendations_title = recommendations["title"].tolist()
        recommendations_distance = recommendations["distance"].tolist()
        for index, distance in enumerate(recommendations_distance):
            recommendations_distance[index] = f"{distance:.2f}"
        return render_template("display_results.html", description=description, recommendations=zip(recommendations_title, recommendations_distance))

@app.route("/recommend_movies_llm", methods=["GET", "POST"])
def recommend_movies_llm() -> str:
    """
    Handles the recommendation of movies based on user input using LLM.
    """
    if request.method == "GET":
        return render_template("recommend_movies_llm.html")
    
    if request.method == "POST":
        description = request.form.get("description")
        num_recommendations = int(request.form.get("num_recommendations"))
        return render_template("display_results_llm.html", description=description)

if __name__ == "__main__":
    app.run(debug=True)