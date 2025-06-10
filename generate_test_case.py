import os
import json
import lancedb
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
from docx import Document
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# === CONFIGURATION ===
LANCE_DB_PATH = "my_lance_db"
TABLE_NAME = "user_stories"
SUCCESS_FOLDER = "success"
TESTCASE_OUTPUT_FOLDER = "testcases"
TOP_K = 3

# Ensure output folder exists
os.makedirs(TESTCASE_OUTPUT_FOLDER, exist_ok=True)

# Load prompt template
with open("test_case_prompt.txt", "r", encoding="utf-8") as f:
    INSTRUCTIONS = f.read()

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",
    temperature=0.3,
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# Connect to LanceDB and load table
db = lancedb.connect(LANCE_DB_PATH)
table = db.open_table(TABLE_NAME)

# Load DataFrame
df = table.to_pandas()

# Filter only rows for which test case file doesn't exist
def should_generate(row):
    story_id = row["storyID"]
    json_file_path = os.path.join(TESTCASE_OUTPUT_FOLDER, f"{story_id}_testcases.json")
    return not os.path.exists(json_file_path)

df_pending = df[df.apply(should_generate, axis=1)]
print("🟡 Found", len(df_pending), "entries to process.\n")

def extract_text(file_path):
    try:
        if file_path.endswith(".pdf"):
            with fitz.open(file_path) as doc:
                return "\n".join(page.get_text() for page in doc)
        elif file_path.endswith(".docx"):
            doc = Document(file_path)
            return "\n".join(p.text for p in doc.paragraphs)
        elif file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            return None
    except Exception as e:
        print(f"❌ Failed to extract {file_path}: {e}")
        return None

# Loop through all rows
for _, row in df_pending.iterrows():
    story_id = row["storyID"]
    vector = np.array(row["vector"])
    filename = row["filename"]
    file_path = os.path.join(SUCCESS_FOLDER, filename)

    print(f"🔍 Processing test case generation for: {story_id}")

    document_text = extract_text(file_path)
    if not document_text:
        print(f"❌ Skipping {story_id} — failed to read original document.")
        continue

    # Similar docs for context
    similar_docs = (
        table.search(vector)
        .distance_type("cosine")
        .limit(TOP_K + 1)
        .to_list()
    )

    context_parts = []
    for doc in similar_docs:
        if doc["storyID"] == story_id:
            continue
        ctx_path = os.path.join(SUCCESS_FOLDER, doc["filename"])
        ctx_text = extract_text(ctx_path)
        if ctx_text:
            context_parts.append(f"--- Context: {doc['storyID']} ---\n{ctx_text.strip()}")

    context_text = "\n\n".join(context_parts)

    # Build prompt
    full_prompt = (
        f"{INSTRUCTIONS.strip()}\n\n"
        f"=======================\n"
        f"📄 USER STORY DOCUMENT ({story_id}):\n"
        f"{document_text.strip()}\n\n"
        f"=======================\n"
        f"📚 RELEVANT CONTEXT DOCUMENTS:\n"
        f"{context_text.strip() if context_text else '[No similar documents found]'}"
    )

    # LLM call
    print(f"💬 Sending prompt to LLM for {story_id}...")
    try:
        response = llm.invoke(full_prompt)
        content = response.content.strip()

        # Strip ```json blocks if present
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]

        try:
            testcases = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"❌ JSON decoding failed for {story_id}: {e}")
            print("📝 Raw cleaned content:\n", content[:1000])
            continue

        # Save test cases
        output_path = os.path.join(TESTCASE_OUTPUT_FOLDER, f"{story_id}_testcases.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(testcases, f, indent=2)

        print(f"✅ Test cases saved to: {output_path}")

        # ✅ Update metadata in LanceDB
        try:
            table.update(
                where=f'"storyID" == \'{story_id}\'',
                values={
                    "test_case_generated": "YES",
                    "test_case_content": output_path
                }
            )
            print(f"✅ Metadata updated for {story_id} in LanceDB.\n")
        except Exception as e:
            print(f"❌ Failed to update LanceDB for {story_id}: {e}")

    except Exception as e:
        print(f"❌ Error generating test cases for {story_id}: {e}")
