import os
import shutil
import json
import lancedb
import fitz
import pyarrow as pa
from docx import Document
from datetime import datetime
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

UPLOAD_FOLDER = config.get("uploaded_docs", "uploaded_docs")
SUCCESS_FOLDER = "success"
FAILURE_FOLDER = "failure"
LANCE_DB_PATH = "my_lance_db"
TABLE_NAME = "user_stories"

os.makedirs(SUCCESS_FOLDER, exist_ok=True)
os.makedirs(FAILURE_FOLDER, exist_ok=True)

embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
db = lancedb.connect(LANCE_DB_PATH)

# Create table if needed
schema = pa.schema([
    ("vector", pa.list_(pa.float32(), 768)),
    ("storyID", pa.string()),
    ("storyDescription", pa.string()),
    ("test_case_content", pa.string()),
    ("filename", pa.string()),
    ("original_path", pa.string()),
    ("doc_content_text", pa.string())
])

table = db.create_table(TABLE_NAME, schema=schema, exist_ok=True)
print(f"Table '{TABLE_NAME}' is ready.")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",
    temperature=0.3,
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

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
        print(f"❌ Error reading {file_path}: {e}")
        return None

def summarize_in_chunks(text, chunk_size=4000):
    try:
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        summaries = []
        for chunk in chunks[:3]:  # Limit to 3 chunks for efficiency
            prompt = (
                "Summarize the following document section in 1 sentence:\n\n" + chunk
            )
            try:
                response = llm.invoke(prompt)
                summaries.append(response.content.strip())
            except Exception as e:
                summaries.append("[Summary failed for a chunk]")
                print(f"❌ LLM failed on a chunk: {e}")
        return " ".join(summaries)
    except Exception as e:
        print(f"❌ LLM summary failed: {e}")
        return "Summary could not be generated."

def story_id_exists(table, story_id):
    try:
        result = table.to_pandas().query("storyID == @story_id")
        return not result.empty
    except Exception:
        return False

for file in os.listdir(UPLOAD_FOLDER):
    file_path = os.path.join(UPLOAD_FOLDER, file)

    if os.path.isdir(file_path):
        continue

    print(f"📄 Processing {file}...")

    text = extract_text(file_path)

    if not text:
        print(f"❌ Skipping {file} — couldn't extract text.")
        shutil.move(file_path, os.path.join(FAILURE_FOLDER, file))
        continue

    try:
        story_id = os.path.splitext(file)[0]

        if story_id_exists(table, story_id):
            print(f"⚠️ Skipping {file} — storyID '{story_id}' already exists.")
            shutil.move(file_path, os.path.join(FAILURE_FOLDER, file))
            continue

        story_description = summarize_in_chunks(text)

        try:
            embedding = embedding_model.encode(text).tolist()
        except Exception as e:
            print(f"❌ Embedding generation failed for {file}: {e}")
            shutil.move(file_path, os.path.join(FAILURE_FOLDER, file))
            continue

        print(f"🔢 Vector length: {len(embedding)} for {file}")

        table.add([{
            "vector": embedding,
            "storyID": story_id,
            "storyDescription": story_description,
            "test_case_content": "",
            "filename": file,
            "original_path": file_path,
            "doc_content_text": text
        }])

        shutil.move(file_path, os.path.join(SUCCESS_FOLDER, file))
        print(f"✅ Stored {file} in LanceDB and moved to success.")
    except Exception as e:
        print(f"❌ Error storing {file}: {e}")
        shutil.move(file_path, os.path.join(FAILURE_FOLDER, file))
