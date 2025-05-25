import json
import chromadb
from sentence_transformers import SentenceTransformer

# load data from JSON
with open("movies_data.json", "r", encoding = "utf-8") as file:
    movies = json.load(file)
    
# chromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="movies")

# embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# string to float conversion (rating field)
def extract_numeric(value):
    if not value:
        return ""
    if "/" in value:
        return float(value.split("/")[0])
    if "%" in value:
        return float(value.replace("%",""))
    return ""

# unique imdbID to avoid duplicates
unique_imdbID = set()

# movies metadata
for movie in movies:
    imdbID = movie.get("imdbID", "")
    
    # skip of duplicates
    if imdbID in unique_imdbID:
        print(f"[X] Skipping of duplicate: {movie.get('Title', 'Unknown Title')} ({imdbID})")
        continue
    
    unique_imdbID.add(imdbID)
    
    # split of nested "Ratings" field
    ratings_dict = {rating["Source"]: rating["Value"] for rating in movie.get("Ratings", [])}
    
    metadata = {
        "Title": movie.get("Title", ""),
        "Year": movie.get("Year", ""),
        "Released": movie.get("Released", ""),
        "Runtime": movie.get("Runtime", ""),
        "Genre": movie.get("Genre", ""),
        "Director": movie.get("Director", ""),
        "Actors": movie.get("Actors", ""),
        "Language": movie.get("Language", ""),
        "Country": movie.get("Country", ""),
        "Type": movie.get("Type", ""),
        "Number of seasons": movie.get("totalSeasons", ""),
        "Plot": movie.get("Plot", ""),
        "IMDb_Rating": extract_numeric(ratings_dict.get("Internet Movie Database", "")),
        "Rotten_Tomatoes_Rating": extract_numeric(ratings_dict.get("Rotten Tomatoes", "")),
        "Metacritic_Rating": extract_numeric(ratings_dict.get("Metacritic", ""))
    }
    
    # plot embedding    
    plot_text = movie.get("Plot", "").strip()
    plot_embedding = embedding_model.encode(plot_text).tolist() if plot_text else None
    
    # if no embeddings Chroma requires documents/images/uris
    document_text = plot_text if plot_text else "No description aviable"

    # dodanie do ChromaDB
    collection.add(
        ids=[imdbID],
        embeddings=[plot_embedding] if plot_embedding is not None else None,
        metadatas=[metadata],
        documents=[document_text]
    )
    
    print(f"[OK] Added to database: {movie.get('Title', 'Unknown Title')} ({imdbID})")

print("Database created")