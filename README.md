## Automated Test Case Generation for User Stories using LLMs, LanceDB, and PostgreSQL

## Overview

A Python project designed to automate the generation of test cases for user stories by leveraging Large Language Models (LLMs), vector search via LanceDB, and persistent storage in PostgreSQL. The system processes user stories, retrieves similar contexts, and generates relevant test cases using LLMs—currently supporting Google's Gemini model.

## Features

- **Automated Test Case Generation**  
  Uses LLMs to generate test cases for user stories.
- **Vector Search**  
  Leverages LanceDB for efficient similarity search over story embeddings.
- **Contextual Prompts**  
  Incorporates context from similar stories to enrich test case generation.
- **PostgreSQL Integration**  
  Persists generated test cases and user story metadata in a relational database.
- **Batch Processing**  
  Processes all user stories without existing test cases in a single run.
- **RAG-based Chat Interface**  
  Supports retrieval-augmented querying for test cases related to user queries.


## Quick Start

1. **Clone the repository:**
```
git clone https://github.com/srinipalli/delta-llm
cd delta-llm
```


2. **Install dependencies:**  
(Python 3.9+ recommended)

```
pip install -r requirements.txt
```


3. **Set up environment variables:**  
Add your Google API key to a `.env` file:

```
GOOGLE_API_KEY=your_api_key_here
```


4. **Prepare your data:**  
- Make sure to have a uploaded_docs folder with all the input requirement docs and then run the data pipeline code.
- Configure PostgreSQL connection in `postgres_writer.py` and in the .env file.

5. **Run the pipeline:**
```
python main.py
```


6. **Use the Chat RAG interface:**
```
results = Chat_RAG("What is the test case for login?")
```

## Project Structure

- **main.py**  
Main script for batch processing user stories and generating test cases.
- **postgres_writer.py**  
Handles PostgreSQL integration for test case storage.
- **test_case_prompt.txt**  
Prompt template for LLM-based test case generation. Can be changed according to use case.
- **new_data_pipeline.py**  
To convert the uploaded_docs to vectors and store in LanceDB along with metadata. Precursor to main.py.
- **uploaded_docs**  
Folder consisting of the developer requirement docs - .txt, .pdf, .docx file types are supported. 
- **.env**  
Environment variables (API keys, etc.).


## How It Works

1. **Retrieve user stories** from LanceDB.
2. **Filter out stories** that already have test cases in PostgreSQL.
3. **Generate test cases** for each new story using LLM prompts enriched by similar context from other stories.
4. **Store results** in PostgreSQL for persistence and easy querying.
5. **Chat RAG interface** allows querying for test cases by user story or natural language description.


## Dependencies

- **Python 3.9+**
- **LanceDB**
- **PostgreSQL**
- **langchain-google-genai**
- **sentence-transformers**
- **python-dotenv**

