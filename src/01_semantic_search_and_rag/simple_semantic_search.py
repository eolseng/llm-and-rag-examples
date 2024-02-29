import chromadb

if __name__ == '__main__':
    # -- Semantic Search using ChromaDB -- #

    # Create a ChromaDB instance and get the client
    chroma_client = chromadb.Client()

    # Create a collection in the database
    collection = chroma_client.create_collection(name="semantic-search-demo")

    # Open the text file and insert each line into the collection
    with open("../../data/tsa_rules.txt", "r") as file:
        lines = file.readlines()

        for i, line in enumerate(lines):
            # Remove the newline character from the line
            line = line.strip()
            item = line.split(".")[0]
            # Insert each line into the collection - this automatically create an embedding for each line
            collection.upsert(ids=str(i + 1), documents=line, metadatas={"line_number": i + 1, "item": item})

    # Search the collection
    # search_query = "I'm traveling with my new born for the first time. What do I need to know?"
    # search_query = "I'm going hunting and camping with my rifle in the forest. What do I need to know?"
    search_query = "I want to bring batteries on my flight. What do I need to know?"
    db_result = collection.query(query_texts=search_query, n_results=5)

    # Print the search results
    for i, document in enumerate(db_result['documents'][0]):
        distance = db_result['distances'][0][i]
        item = db_result['metadatas'][0][i].get("item")
        line_number = db_result['metadatas'][0][i].get("line_number")

        print(
            f"#: {i + 1:2} | Distance: {distance:.5f} | Line: {line_number:3} | Item: {item:16} | Document: {document}")
