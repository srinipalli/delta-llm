import os
import json
import lancedb
import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from llm_utils.postgres_writer import (
    insert_test_case,
    get_test_case_json_by_story_id,
    get_all_generated_story_ids
)
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

load_dotenv()

# === CONFIGURATION ===
LANCE_DB_PATH = "my_lance_db"
TABLE_NAME = "user_stories"
TOP_K = 3
MAX_MAIN_TEXT_CHARS = 5000

# Load prompt
with open("test_case_prompt.txt", "r", encoding="utf-8") as f:
    INSTRUCTIONS = f.read()

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",
    temperature=0.3,
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# Connect to LanceDB
db = lancedb.connect(LANCE_DB_PATH)
table = db.open_table(TABLE_NAME)

'''
For frontend integration, just import and call:

from your_module import generate_test_case_for_story
generate_test_case_for_story(story_id, table, llm)
'''

def generate_test_case_for_story(story_id, table_ref=table, llm_ref=llm):
    all_rows = table_ref.search().to_list()
    row = next((r for r in all_rows if r["storyID"] == story_id), None)

    if not row:
        print(f"❌ Story ID '{story_id}' not found in LanceDB.")
        return

    story_description = row["storyDescription"]
    vector = np.array(row["vector"])
    main_text = row.get("doc_content_text", "").strip()

    if not main_text:
        print(f"❌ Skipping {story_id} — missing doc_content_text.")
        return

    print(f"🔍 Generating test case for: {story_id}")

    # Get similar docs
    similar_docs = (
        table_ref.search(vector)
        .distance_type("cosine")
        .limit(TOP_K + 5)
        .to_list()
    )

    context_parts = []
    for doc in similar_docs:
        ctx_story_id = doc["storyID"]
        if ctx_story_id == story_id:
            continue

        test_case_json = get_test_case_json_by_story_id(ctx_story_id)
        if test_case_json:
            context_parts.append(f"--- Context from {ctx_story_id} ---\n{json.dumps(test_case_json, indent=2)}")

        if len(context_parts) >= TOP_K:
            break

    context_text = "\n\n".join(context_parts)

    full_prompt = (
        f"{INSTRUCTIONS.strip()}\n\n"
        f"=======================\n"
        f"📄 USER STORY DOCUMENT ({story_id}):\n"
        f"{main_text}\n\n"
        f"=======================\n"
        f"📚 RELEVANT CONTEXT TEST CASES:\n"
        f"{context_text if context_text else '[No similar context found]'}"
    )

    #print(f"\n🧪 Prompt being sent for {story_id}:\n{'='*60}\n{full_prompt[:1000]}\n{'='*60}\n")

    try:
        response = llm_ref.invoke(full_prompt)
        content = response.content.strip()

        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]

        try:
            testcases = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode failed for {story_id}: {e}")
            print("Raw response:\n", content[:1000])
            return

        insert_test_case(
            story_id=story_id,
            story_description=story_description,
            test_case_json=testcases
        )
        print(f"✅ Inserted test cases for {story_id} into Postgres.\n")

    except Exception as e:
        print(f"❌ LLM error for {story_id}: {e}")

# === Run for all unprocessed
generated_ids = set(get_all_generated_story_ids())
records = [row for row in table.search().to_list() if row["storyID"] not in generated_ids]

print(f"🟡 Found {len(records)} entries to process.\n")

for row in records:
    generate_test_case_for_story(row["storyID"])

def Chat_RAG(user_query, table_ref=table, top_k=3):
    query_vector = embedding_model.encode(user_query).tolist()

    results = (
        table_ref.search(query_vector)
        .distance_type("cosine")
        .limit(top_k)
        .to_list()
    )

    if not results:
        return {"error": "No relevant stories found."}

    response = []
    for result in results:
        story_id = result["storyID"]
        test_case_json = get_test_case_json_by_story_id(story_id)
        response.append({
            "story_id": story_id,
            "similarity_score": result["_distance"],
            "test_case_json": test_case_json or "[Test case not found]"
        })

    return response

if __name__ == "__main__":
    # Run RAG search on a sample query
    test_query = "What is the test case for login?"
    results = Chat_RAG(test_query)

    print("\n🎯 Results from Chat_RAG:")
    for res in results:
        print(f"\nStory ID: {res['story_id']}")
        print(f"Similarity Score: {res['similarity_score']:.4f}")
        print(f"Test Case JSON:\n{json.dumps(res['test_case_json'], indent=2)}\n")
