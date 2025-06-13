# generate.py

import json
import numpy as np
from llm_utils import load_prompt
from postgres_writer import insert_test_case, get_test_case_json_by_story_id

INSTRUCTIONS = load_prompt("test_case_prompt.txt")
TOP_K = 3

def generate_test_case_for_story(story_id, table_ref, llm_ref):
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
