# Description: This script demonstrates how to use ChromaDB for semantic search and OpenAI Chat API for RAG. It uses
# the same Semantic Search code from simple_semantic_search.py, but adds a RAG response using OpenAI Chat API. It
# supports multiple embedding functions and OpenAI models to demonstrate the impact on quality and relevance of the
# response.

import sys
from os import environ

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# embedding_function = embedding_functions.DefaultEmbeddingFunction()
embedding_function = embedding_functions.OpenAIEmbeddingFunction(
    model_name="text-embedding-3-small",
    api_key=environ.get("OPENAI_API_KEY"),
)

GPT35_TURBO = "gpt-3.5-turbo"
GPT4_TURBO = "gpt-4-0125-preview"

if __name__ == '__main__':
    # -- Semantic Search using ChromaDB -- #
    # Create a persistent ChromaDB instance and get the client
    chroma_client = chromadb.PersistentClient()

    # Create or retrieve an existing collection in the database
    collection = chroma_client.get_or_create_collection(
        # name="semantic-search-demo",
        name="semantic-search-demo-oai",
        embedding_function=embedding_function
    )

    # If the collection is empty, insert the rules from the text file
    if collection.count() == 0:
        # Open the text file and insert each line into the collection
        with open("../../data/tsa_rules.txt", "r") as file:
            lines = file.readlines()
            print(f"Inserting {len(lines)} lines into the collection")
            for i, line in enumerate(lines):
                # Remove the newline character from the line
                line = line.strip()
                item = line.split(".")[0]
                # Insert each line into the collection - this automatically create an embedding for each line
                collection.upsert(ids=str(i + 1), documents=line, metadatas={"line_number": i + 1, "item": item})
            print(f"Inserted {collection.count()} lines into the collection")

    # Define the search query
    # search_query = "I'm traveling with my newborn for the first time. What do I need to know?"
    search_query = "I'm going hunting and camping with my rifle in the forest. What do I need to know?"
    # search_query = "I'm traveling with a school band for a college football game. What do I need to know?"

    # Print the search query
    print(f"Search query: {search_query}")

    # Search the collection for the most relevant rules
    db_result = collection.query(query_texts=search_query, n_results=20)

    # Print the items and the distance
    print(f"\nRetrieved rules:")
    print(f"     Dist.:  | Item:")
    for i, document in enumerate(db_result['documents'][0]):
        distance = db_result['distances'][0][i]
        item = db_result['metadatas'][0][i].get("item")
        # Format the index as i+1 with a fixed width of 2
        print(f"{i + 1:2} | {distance:.5f} | {item}")

    # -- RAG using OpenAI Chat API -- #
    # Create a newline separated list of relevant rules
    rules = "\n".join(db_result['documents'][0])

    # Create a system message to provide context to the AI model, such as the relevant rules
    system_message = f"""
You are a helpful airline travel assistant specializing in TSA baggage rules.

These rules might be relevant to the users query:
---
{rules}
---
Do ONLY answer the users query based on the rules provided.
Do ONLY include rules that are relevant to the users query.
Do NOT include irrelevant rules.

Answer politely, professionally, and concise.
""".strip()

    # Get the chat response to the user query, providing the system message as context
    client = OpenAI()
    completion = client.chat.completions.create(
        model=GPT35_TURBO,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": search_query}
        ],
        stream=True
    )
    # Stream the response to the console
    print("\nChat response:")
    for chunk in completion:
        content = chunk.choices[0].delta.content
        sys.stdout.write(content or "")  # Write the content to the console

    # Close the stream
    sys.stdout.flush()
