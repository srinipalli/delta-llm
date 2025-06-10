import os
import shutil
import json
import lancedb
import fitz
import pyarrow as pa
from docx import Document
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

# ✅ Use guaranteed 768-dim model
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
db = lancedb.connect(LANCE_DB_PATH)

# ✅ Create table with test_case_content added
try:
    table = db.open_table(TABLE_NAME)
    print(f"Table '{TABLE_NAME}' already exists.")
except Exception:
    schema = pa.schema([
        ("vector", pa.list_(pa.float32(), 768)),
        ("storyID", pa.string()),
        ("storyDescription", pa.string()),
        ("test_case_generated", pa.string()),
        ("test_case_content", pa.string()),  # ✅ New field
        ("filename", pa.string()),
        ("original_path", pa.string())
    ])
    table = db.create_table(TABLE_NAME, schema=schema)
    print(f"Table '{TABLE_NAME}' created.")

# LLM for summarizing
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

def describe_user_story(filename, text):
    try:
        prompt = (
            "Summarize the following document in 1-2 concise sentences. "
            "Focus only on what the document is about and avoid unnecessary details:\n\n"
            + text[:4000]
        )
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"❌ LLM summary failed for {filename}: {e}")
        return f"Summary could not be generated for {filename}"

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
        story_description = describe_user_story(file, text)

        # ✅ Use SentenceTransformer and convert to list
        embedding = embedding_model.encode(text).tolist()
        print(f"🔢 Vector length: {len(embedding)} for {file}")

        table.add([{
            "vector": embedding,
            "storyID": story_id,
            "storyDescription": story_description,
            "test_case_generated": "NO",
            "test_case_content": "",  # ✅ initialize as blank
            "filename": file,
            "original_path": file_path
        }])

        shutil.move(file_path, os.path.join(SUCCESS_FOLDER, file))
        print(f"✅ Stored {file} in LanceDB and moved to success.")
    except Exception as e:
        print(f"❌ Error storing {file}: {e}")
        shutil.move(file_path, os.path.join(FAILURE_FOLDER, file))
