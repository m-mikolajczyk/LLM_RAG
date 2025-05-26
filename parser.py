import config
import json
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


class QueryParser:
    def __init__(self, llm):
        self.llm = llm
        self.parser = StructuredOutputParser.from_response_schemas([
            ResponseSchema(name="classification", description="The type of query, either 'similarity', 'metadata', 'filter', or 'others'.")
        ])
        self.metadata_parser = StructuredOutputParser.from_response_schemas([
            ResponseSchema(name="search_field", description="Decide whether to search for 'title' or 'actor'.")
        ])
        self.filter_parser = StructuredOutputParser.from_response_schemas([
            ResponseSchema(name="genre", description="Genre filter if specified, else null."),
            ResponseSchema(name="runtime_threshold", description="A dictionary with 'operator' (< or >) and 'value' (runtime in minutes or hours)."),
            ResponseSchema(name="rating_source", description="Which rating source is mentioned (IMDb, Rotten_Tomatoes, Metacritic), if none of there are mentioned return 'IMDb'."),
            ResponseSchema(name="rating_threshold", description="Rating value threshold if specified, else null."),
            ResponseSchema(name="limit", description="Number of movies requested if specified, default 3."),
        ])
    
    def parse(self, query):
        formatted_prompt = f"""
        Classify this query into one of four categories:

        - 'similarity': Asking for similar movies, OR for movies based on **themes, topics, settings, or abstract concepts** — even if a number is mentioned (e.g., "5 movies about love").
         Examples:
        - "Recommend me movies like Inception"
        - "Find similar movies to Gladiator II"
        - "Give me 5 movies about love"
        - "Name 3 movies with superheroes"
        - "Provide 3 movies about serial killers"
        - "List 4 films involving time travel"
        - "Suggest 5 films about artificial intelligence"
    
        - 'metadata': Asking for **movie details**, such as actors, release dates, or genres.
          Examples:
          - "Who directed The Dark Knight?"
          - "When was Interstellar released?"
          - "List the actors in Titanic"

        - 'filter': Requesting a **list of movies with specific conditions**.
          Examples:
          - "Top 3 action movies with a rating above 8"
          - "Show me horror movies from the 90s"
          - "Find comedies with a Rotten Tomatoes score over 70%"

        - 'others': If the query does not fit these categories.

        Respond in JSON without any whitespaces or addidionaly things, just raw JSON response:
        {{ "classification": "similarity" | "metadata" | "filter" | "others" }}
        
        Query: {query}
        """
        response = self.llm.invoke(formatted_prompt).strip()
        
        print("Raw response:", repr(response))
        
        # Trivial fix of incomplete JSON
        if not response.endswith("}"):
            print("Incomplete JSON format, fixing with trivial fix")
            response += "}"

        # Parse the response
        return self.parser.parse(response)["classification"]


    def parse_metadata(self, query):
        formatted_prompt = f"""
        You need to decide whether the search should be based on:
        - 'title' (if the query is looking for a movie by name and contains a movie title)
        - 'actor' (if the query is looking for movies with a specific actor and contains a person's full name)

        The decision is about **how to search the database**, NOT what the question ultimately asks.  
        If the query includes a movie title, **always** return "title".  
        If the query includes an actor's full name, return "actor".  
        Do not assume the question meaning—just classify the search type.

        Examples:
        - "Who played in Sanak?" → "title" !! As well as similar ones questions asking about certain things about movies!(first, search the database for 'Sanak', then answer)
        - ""Who is in the cast of Gladiator II?" → "title"
        - "What movies did Leonardo DiCaprio star in?" → "actor" (search by actor name)
        - "When was The Matrix released?" → "title" (search by title first)
    
        Respond in **EXACTLY** this JSON format:
        {{
            "search_field": "title" | "actor"
        }}

        Query: {query}
        """
        response = self.llm.invoke(formatted_prompt)
        return self.metadata_parser.parse(response)["search_field"]

    def parse_filters(self, query):
        formatted_prompt = f"""
        Extract filtering criteria from the query.
    
        **Your response must be a valid JSON object, without extra text.**  
        **Do not explain, only return JSON.**  

        Example:
        Query: "Give me 5 action movies with Rotten Tomatoes rating above 5/10"
        Response:
        {{
            "genre": "Action",
            "runtime_threshold": {{"operator": ">", "value": 5}},
            "rating_source": "Rotten_Tomatoes",
            "rating_threshold": 5.0,
            "limit": 5
        }}

        If no rating_threshold is mentioned, return "rating_threshold": null.  
        If no rating_source is mentioned, return "rating_source": "IMDb".  

        Your response must be strictly formatted like this:
        {{
            "genre": <string> or null,
            "runtime_threshold": {{"operator": "<" or ">", "value": <int>}} or null,
            "rating_source": <string> or "IMDb",
            "rating_threshold": <float> or null,
            "limit": <int>
        }}

        Query: {query}
        """
    
        response = self.llm.invoke(formatted_prompt).strip()

        print("LLM raw response:", response)  # Debugging print
        
        # Trivial fix of incomplete JSON
        if not response.endswith("}"):
            print("Incomplete JSON format, fixing with trivial fix")
            response += "}"  

        try:
            # Ensure valid JSON response
            parsed_response = json.loads(response)

            # Handling missing values 
            parsed_response.setdefault("rating_threshold", None)
            parsed_response.setdefault("rating_source", "IMDb")

            return parsed_response
        except json.JSONDecodeError as e:
            print(f"Error parsing response: {e}")
            return {}

query_parser = QueryParser(config.llm)