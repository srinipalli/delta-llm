import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
import os
import uuid
import datetime
import json

load_dotenv()

def get_postgres_connection():
    return psycopg2.connect(
        dbname=os.getenv("PG_DB"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        host=os.getenv("PG_HOST", "localhost"),
        port=os.getenv("PG_PORT", 5432)
    )

def get_test_case_json_by_story_id(story_id):
    """Get test_case_json for a single story_id (used for context)."""
    try:
        conn = get_postgres_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            query = """
                SELECT test_case_json
                FROM test_cases
                WHERE story_id = %s
                AND test_case_json IS NOT NULL
                LIMIT 1
            """
            cur.execute(query, (story_id,))
            result = cur.fetchone()
            return result["test_case_json"] if result else None
    except Exception as e:
        print(f"❌ Error fetching test case for {story_id}: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_all_generated_story_ids():
    """Fetch list of story_ids that already have test cases generated."""
    try:
        conn = get_postgres_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT story_id FROM test_cases
                WHERE test_case_generated = TRUE
            """)
            return [row[0] for row in cur.fetchall()]
    except Exception as e:
        print(f"❌ Error fetching generated story IDs: {e}")
        return []
    finally:
        if conn:
            conn.close()

def insert_test_case(story_id, story_description, test_case_json):
    """Insert or update generated test case JSON into PostgreSQL."""
    try:
        conn = get_postgres_connection()
        with conn.cursor() as cur:
            query = """
                INSERT INTO test_cases (
                    run_id,
                    story_id,
                    story_description,
                    created_on,
                    test_case_json,
                    total_test_cases,
                    test_case_generated
                ) VALUES (%s, %s, %s, %s, %s, %s, TRUE)
                ON CONFLICT (story_id)
                DO UPDATE SET
                    test_case_json = EXCLUDED.test_case_json,
                    total_test_cases = EXCLUDED.total_test_cases,
                    test_case_generated = TRUE
            """
            run_id = str(uuid.uuid4())
            created_on = datetime.datetime.now()
            total_test_cases = len(test_case_json.get("test_cases", []))

            cur.execute(query, (
                run_id,
                story_id,
                story_description,
                created_on,
                json.dumps(test_case_json),
                total_test_cases
            ))
            conn.commit()
    except Exception as e:
        print(f"❌ Failed to insert test case for {story_id}: {e}")
    finally:
        if conn:
            conn.close()
