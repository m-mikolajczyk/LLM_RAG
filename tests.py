import utils
import csv
import random
import chromadb
import pandas as pd

### CONFIGURATION ###
MODE = "actors"  # Options: "actors", "titles", ,"filter", "similarity", "embeddings"
SAMPLE_SIZE = 2
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "movies3"
OUTPUT_PATH = f"random_{SAMPLE_SIZE}_{MODE}_with_responses.csv"
#####################

# ChromaDB
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

# Retrieve movie metadata
all_movies = collection.get()

# Actor data
all_actors_raw = [meta["Actors"] for meta in all_movies["metadatas"] if "Actors" in meta and meta["Actors"]]
actor_set = set()
for actor_line in all_actors_raw:
    actor_list = [actor.strip() for actor in actor_line.split(",") if actor.strip()]
    actor_set.update(actor_list)
all_actors = list(actor_set)

# Titles data
movie_titles = [meta["Title"] for meta in all_movies["metadatas"] if "Title" in meta]

# Filter data
genres = ["action", "comedy", "drama", "horror", "sci-fi", "romance", "thriller", "animation"]
rating_types = ["IMDb", "Rotten Tomatoes", "Metacritic"]
directions = ["above", "below"]
runtime_directions = ["greater", "less"]

def generate_filter_inputs(n):
    inputs = []
    for _ in range(n):
        item = {
            "n": random.randint(1, 5),
            "genre": random.choice(genres),
            "rating_type": random.choice(rating_types),
            "direction": random.choice(directions),
            "rating_value": random.randint(2, 8),
            "runtime": random.randint(1, 179),
            "runtime_direction": random.choice(runtime_directions),
        }
        inputs.append(item)
    return inputs

# Queries
queries_actors = [
    "What are some films featuring {}",
    "In what movies did {} star?",
    "Which movies feature {}?",
    "Can you list some movies with {} in the cast?",
    "Name some movies that {} acted in.",
    "What movies has {} been in?",
    "Which roles has {} played in movies?"
    ]

queries_titles = [
    "Who is in the cast of {}?",
    "How many minutes does {} run for?",
    "Which genre does {} belong to?",
    "Who directed {}?",
    "What is the plot of {}?",
    "On which date was {} released?",
    ]

queries_similarity = [
    "Recommend a few movies similar to {}.",
    "What are some movies like {}?",
    "Can you list a few movies similar to {} ?",
    "What movies are comparable to {}?",
    "Which films resemble {} in style or theme?",
    "If I liked {}, what else should I watch?",
    "Find movies similar to {}.",
]

queries_filter = [
    "Recommend me {n} {genre} movies with {rating_type} rating {direction} {rating_value}.",
    "Can you suggest {n} {genre} films  with {rating_type} above {rating_value}?",
    "List {n} {genre} movies with a runtime {runtime_direction} than {runtime} minutes and {rating_type} {direction} {rating_value}.",
    "Give me {n} {genre} movies that have an {rating_type} rating {direction} {rating_value} and runtime {runtime_direction} than {runtime} minutes.",
    "I want {n} {genre} movies with {rating_type} score {direction} {rating_value}.",
]

queries_embeddings = [
    "Give me 5 movies about love.",
    "Name 3 movies with superheroes.",
    "Provide me 3 movies about serial killers.",
    "List 4 films involving time travel.",
    "Find 2 movies set during World War II.",
    "Suggest 5 films about artificial intelligence.",
    "Name 3 mystery thrillers involving detectives.",
    "Recommend 4 animated movies about animals.",
    "Give me 2 movies about surviving in the wilderness.",
    "List 3 romantic comedies with unusual couples.",
]

# Sample input data
if MODE == "actors":
    input_items = random.sample(all_actors, min(SAMPLE_SIZE, len(all_actors)))
    query_templates = queries_actors
elif MODE == "titles":
    input_items = random.sample(movie_titles, min(SAMPLE_SIZE, len(movie_titles)))
    query_templates = queries_titles
elif MODE == "similarity":
    input_items = random.sample(movie_titles, min(SAMPLE_SIZE, len(movie_titles)))
    query_templates = queries_similarity
elif MODE == "filter":
    input_items = generate_filter_inputs(SAMPLE_SIZE)
    query_templates = queries_filter
elif MODE == "embeddings":
    input_items = queries_embeddings  # each is already a full query
    query_templates = None  

# Generate responses
results = []
for item in input_items:
    if MODE == "filter":
        query_template = random.choice(query_templates)
        query = query_template.format(**item)
        display_item = f"{item['genre']} | {item['rating_type']} {item['direction']} {item['rating_value']} | {item['runtime']} {item['runtime_direction']}"
    elif MODE == "embeddings":
        query = item  
        display_item = "Freeform"
    else:
        query_template = random.choice(query_templates)
        query = query_template.format(item)
        display_item = item
        
    response = utils.search_and_answer(query)
    results.append((display_item, query, response))

# Save 
df_results = pd.DataFrame(results, columns=["Item", "LLM_Query", "LLM_Response"])
df_results.to_csv(f"{OUTPUT_PATH}", index=False)
print(f"Results saved to {OUTPUT_PATH}")
