# main.py

import os
import lancedb
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from postgres_writer import get_all_generated_story_ids
from generate import generate_test_case_for_story
from rag import Chat_RAG

load_dotenv()

# Constants
LANCE_DB_PATH = "my_lance_db"
TABLE_NAME = "user_stories"

# Connect DB and LLM
db = lancedb.connect(LANCE_DB_PATH)
table = db.open_table(TABLE_NAME)

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",
    temperature=0.3,
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

if __name__ == "__main__":
    # Run generation
    generated_ids = set(get_all_generated_story_ids())   
    records = [row for row in table.search().to_list() if row["storyID"] not in generated_ids] # gets stories for which test cases have not been generated yet

    print(f"🟡 Found {len(records)} entries to process.\n")

    for row in records:
        generate_test_case_for_story(row["storyID"], table, llm)

    # Sample RAG query
    # test_query = "What is the test case for login?"
    # results = Chat_RAG(test_query, table)

    # print("\n🎯 Results from Chat_RAG:")
    # for res in results:
    #     print(f"\nStory ID: {res['story_id']}")
    #     print(f"Similarity Score: {res['similarity_score']:.4f}")
    #     print(f"Test Case JSON:\n{res['test_case_json']}\n")
