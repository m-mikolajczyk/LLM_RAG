import config
import parser
import random
import numpy as np
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process


def unify_rating(value, source):
    if source == "Rotten_Tomatoes":
        return value / 10               # Convert % to 0-10 scale
    elif source == "Metacritic":
        return value / 10               # Convert 0-100 to 0-10 scale
    return value                        # IMDb has correct format

def filter_movies(query):
    filters = parser.query_parser.parse_filters(query) # Parsing user query

    print("Filters after parsing:", filters)  # Debugging print

    if not filters:
        print("No filters parsed, returning empty list.")
        return []

    # Retrieve all data from collection
    all_results = config.collection.get()

    if not all_results or "metadatas" not in all_results or not all_results["metadatas"]:
        return "No movies found in the database."

    filtered_movies = []

    # Runtime filtering
    if filters.get("runtime_threshold"):
        operator = filters["runtime_threshold"]["operator"]
        value = filters["runtime_threshold"]["value"]

        try:
            value = int(value)  # Ensure runtime is in minutes
        except ValueError:
            value = 60  # Default to 60 min if something goes wrong
        
        # Create new and delete obsolete
        filters["runtime"] = (operator, value)
        del filters["runtime_threshold"]

    # Loop for all movies
    for metadata in all_results["metadatas"]:
        # GENRE filter
        if filters.get("genre"):
            requested_genre = filters["genre"].lower()
            movie_genres = [g.strip().lower() for g in metadata["Genre"].split(",")]

            if not any(requested_genre in genre for genre in movie_genres):
                continue  # Allow partial matching

        # RUNTIME filter
        if "runtime" in filters:
            operator, runtime_threshold = filters["runtime"]
            runtime_str = metadata.get("Runtime", "").replace("min", "").strip()

            try:
                runtime = int(runtime_str)
                if operator == "<" and runtime >= runtime_threshold:
                    continue  # Skip if runtime is too long
                elif operator == ">" and runtime <= runtime_threshold:
                    continue  # Skip if runtime is too short
            except ValueError:
                continue  # Skip if runtime is invalid

        # RATING filter
        if filters.get("rating_source") and filters.get("rating_threshold") is not None:
            rating_source = filters["rating_source"]
            rating_key = f"{rating_source}_Rating"
            rating = metadata.get(rating_key)

            if rating is None or rating == "":
                continue  # Skip movies without rating

            try:
                rating = float(rating)
                rating = unify_rating(rating, rating_source)
                if rating < filters["rating_threshold"]:
                    continue  # Skip movies below the threshold
            except ValueError:
                continue  # If rating cannot be converted, skip the movie

        # Collect ALL movies that meet the user requirements 
        filtered_movies.append(metadata)
        
    # Ranomization
    random.shuffle(filtered_movies)

    # Apply limit based on user requirement
    return filtered_movies[:filters.get("limit", 3)]

# Load the NLP model
nlp = spacy.load("en_core_web_trf")

def extract_entities(query):
    doc = nlp(query)
    named_entities = {ent.label_: ent.text for ent in doc.ents}
    print("Extracted Entities:", named_entities)  # Debugging print
    return named_entities

def get_movie_title_from_entities(entities):
    # Preferred entity types for movie titles
    preferred_labels = ["WORK_OF_ART", "EVENT", "ORG", "GPE"]

    # First, check for a preferred entity type
    for label in preferred_labels:
        if label in entities:
            return entities[label]

    # As a fallback, pick any entity EXCEPT numerical ones
    for label, value in entities.items():
        if label not in ["CARDINAL", "ORDINAL", "DATE", "MONEY", "PERCENT", "QUANTITY", "TIME"]:
            return value  # Return the first valid non-numerical entity

    # If no suitable entity is found, return None
    return None


def guess_movie_title(query):
    formatted_prompt = f"""
    Identify the most likely movie title from the following query:
    {query}
    Return only the title as plain text, without any extra explanation.
    Do not include phrases like 'most likely title is'—just return the title alone.
    """
    response = config.llm.invoke(formatted_prompt).strip()
    guessed_title = response.split("\n")[-1].strip()  # Take the last line (title)
    return guessed_title if guessed_title.lower() != "unknown" else None

def search_movies(query):
    entities = extract_entities(query)  # Extract named entities
    query_type = parser.query_parser.parse(query)
    
    if query_type == "filter":
        movies = filter_movies(query)
        return movies if movies else "No movies found matching your criteria."
    
    if query_type == "metadata":
        print("\nMetadata search \n")
        search_field = parser.query_parser.parse_metadata(query)  # "title" or "actor"
        print(f"Decided search field for retrieval: {search_field}") # Debugging print
        
        all_results = config.collection.get()
        collection_titles = [metadata["Title"] for metadata in all_results["metadatas"] if "Title" in metadata]

        if search_field == "title":
            movie_title = get_movie_title_from_entities(entities)
            
            if not movie_title:
                print("No entity detected, attempting to infer movie title with LLM...")
                movie_title = guess_movie_title(query)
                
            if not movie_title:
                return "No movie title detected in the query."
            
            # Use fuzzy matching to find the best title match
            best_match, score = process.extractOne(movie_title, collection_titles)
            if score >= 80:  
                print(f"Best fuzzy match: {best_match} (Score: {score})")
                movie_title = best_match
            else:
                return f"No close match found for {movie_title}."

            matching_results = [
                metadata for metadata in all_results["metadatas"]
                if metadata["Title"].lower() == movie_title.lower()
            ]
            
            return matching_results[0] if matching_results else f"No metadata found for {movie_title}."

        elif search_field == "actor":
            actor_name = entities.get("PERSON")
            
            if not actor_name:
                return "No actor name detected in the query."

            matching_results = [
                metadata for metadata in all_results["metadatas"]
                if "Actors" in metadata and actor_name.lower() in metadata["Actors"].lower()
            ]
            
            return matching_results if matching_results else f"No metadata found for {actor_name}."

    
    elif query_type == "similarity":
        print("\nEmbedding-based Similarity Search \n")  # Debugging print
    
        # Extract movie title if mentioned
        movie_title = get_movie_title_from_entities(entities)
        
        if movie_title:
            print(f"Detected movie title for similarity search: {movie_title}")
        
            # Try to retrieve the movie's metadata to get its embedding
            all_results = config.collection.get(include=["embeddings", "metadatas"])
            collection_titles = [metadata["Title"] for metadata in all_results["metadatas"] if "Title" in metadata]
        
            # Fuzzy match to find the closest title in the database
            best_match, score = process.extractOne(movie_title, collection_titles)
            if score >= 80:
                print(f"Best fuzzy match: {best_match} (Score: {score})")
                movie_title = best_match
            else:
                return f"No close match found for {movie_title}."
        
            matching_movie = next(
                (metadata for metadata in all_results["metadatas"] if metadata["Title"].lower() == movie_title.lower()), None
            )
        
            if matching_movie:
                print(f"Found movie in database: {matching_movie['Title']}")
            
                # Retrieve embeddings
                movie_embedding = config.collection.get(where={"Title": matching_movie["Title"]}, include=["embeddings"])
                embeddings = movie_embedding.get("embeddings", [])
            
                if isinstance(embeddings, np.ndarray) and embeddings.size > 0:
                    query_embedding = embeddings[0]
                elif isinstance(embeddings, list) and len(embeddings) > 0:
                    query_embedding = embeddings[0]
                else:
                    return f"Could not retrieve embeddings for {movie_title}."
            else:
                return f"No matching movie found for {movie_title}."
        else:
            print("No specific movie detected. Using query embedding for search.")
            query_embedding = np.array(config.embedding_model.embed_query(query)).reshape(1, -1)
    
        # Retrieve all stored embeddings and metadata
        stored_data = config.collection.get(include=["embeddings", "metadatas"])
        stored_embeddings = np.array(stored_data["embeddings"])
        movie_metadata = stored_data["metadatas"]
    
        # Calculate cosine similarity
        similarity_scores = cosine_similarity(query_embedding.reshape(1, -1), stored_embeddings).flatten()
    
        # Sort results based on similarity
        sorted_indices = np.argsort(similarity_scores)[::-1]  # Sort in descending order
        sorted_movies = [(movie_metadata[i], similarity_scores[i]) for i in sorted_indices]
    
        # Return top 6 results
        top_results = sorted_movies[:6]
    
        if top_results:
            print("\nRetrieved Similar Movies:\n")
            for i, (movie, score) in enumerate(top_results[1:], start=2):  # Skip first movie, start numbering from 2
                print(f"{i-1}. {movie['Title']} - Similarity: {score:.4f}")
        
            return [movie for movie, _ in top_results]
        else:
            return "No similar movies found."
        
    elif query_type == "others":
        # Answer using LLM general knowledge
        return config.llm.invoke(f"You are a helpful assistant. Answer this question:\n\n{query}")

    else:
        return "Sorry, I couldn't understand your request. Please ask about movies, actors, genres, or similar films."


def ask_llm_about_movie(movie_metadata, user_query):
    if isinstance(movie_metadata, str):  # If no movie was found
        return movie_metadata

    formatted_prompt = f""" 
    Here is information about the movie:
    {movie_metadata}

    Based on this information, answer the following question in a full sentence:
    {user_query}
    """

    llm_response = config.llm.invoke(formatted_prompt)
    return llm_response.strip()  # Return LLM's answer


# Adding a second LLM to act as the critic
def validate_response(original_response, user_query):
    """Uses the critic LLM to validate and correct mistakes in the generated response."""
    
    validation_prompt = f"""
    You are a fact-checking AI. Your task is to verify the accuracy of the response below 
    and correct any mistakes in it.

    User Query: {user_query}

    Original Response: {original_response}

    **Validation Instructions:**
    - Ensure numerical facts are correct (e.g., 88 min is greater than 1 hour, 66.0 is greater than 6.0).
    - Verify that movies listed match the requested filters.
    - If you find mistakes, return the corrected response.
    - If everything is correct, return "VALID".

    Provide only the corrected response or "VALID" without extra explanations.
    """
    
    validation_result = config.critic_llm.invoke(validation_prompt).strip()

    if validation_result == "VALID":
        return original_response  # Return the original response if it's correct

    return validation_result  # Return corrected response if validation finds issues


def search_and_answer(query):
    """Runs movie search and validates the output before returning it."""
    
    # Retrieve movie data
    movie_data = search_movies(query)
    
    # Generate initial response
    initial_response = ask_llm_about_movie(movie_data, query)
    
    # Validate and correct mistakes
    final_response = validate_response(initial_response, query)

    return final_response  # Return the corrected response or the original if valid