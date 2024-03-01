# LLMs, Vector Databases, and RAG

## Dependencies and Poetry

This repository uses [Poetry](https://python-poetry.org/) for dependency management. To install the dependencies, run:

```bash
poetry install
```

In case you don't want to use Poetry, you can install the dependencies in your environment manually using `pip`:

```bash
pip install -r requirements.txt
```

Note to self: When adding new dependencies, update the `requirements.txt` file by running:

```bash
poetry export -f requirements.txt --output requirements.txt
```

## `.env` file

Run the following command to create a `.env` file from the [.env.example](.env.example) file:

```bash
cp .env.example .env
```

Then, fill in the values for the environment variables.

## Part 1: Semantic Search and Simple RAG

The first part introduces simple implementations of Semantic Search and RAG.  
It uses [ChromaDB](https://www.trychroma.com/) for vector storage and search, and
the [OpenAI package](https://platform.openai.com/docs/api-reference/chat) for embeddings and LLM chat completions. 

## Part 2: TBD

I will update this repository with more code as I have time.
