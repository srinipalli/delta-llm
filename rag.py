# rag.py

from sentence_transformers import SentenceTransformer
from postgres_writer import get_test_case_json_by_story_id

embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def Chat_RAG(user_query, table_ref, top_k=3):
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
