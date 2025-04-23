SYSTEM_PROMPT = """You are an AI assistant that turns a user's “ideal movie” description plus a list of candidate films into a clear, structured set of recommendations."""

def get_llm_prompt(user_description: str, movies: str ,n: int) -> str:
    """
    Returns the prompt for the LLM.
    """
    return f"""INPUT:

1. User's ideal movie description:
{user_description}

2. Top {n} closest movies:
{movies}

TASK:
For each movie in the list, in descending order of similarity_score:
1. Print the movie title on its own line.
2. On the next line, write a brief paragraph (2-4 sentences) explaining why this movie matches the user's tastes, referencing aspects of the user's description (themes, tone, setting, characters, etc.).
3. Then move to the next movie.
4. Don't print any additional text.
"""