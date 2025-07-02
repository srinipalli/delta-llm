import os
import json
import lancedb
import numpy as np
from dotenv import load_dotenv
from app.config import Config
from app.models.postgress_writer import (
    insert_test_case,
    get_test_case_json_by_story_id,
    get_all_generated_story_ids
)
import pandas as pd
from .impact_analyzer import analyze_test_case_impacts
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
import re
import random  # Add import for randomization

load_dotenv()

# === CONFIGURATION ===
LANCE_DB_PATH = Config.LANCE_DB_PATH
TABLE_NAME = Config.TABLE_NAME_LANCE
TOP_K = 3
MAX_MAIN_TEXT_CHARS = 5000
BATCH_SIZE = 10  # Increased batch size for more efficient generation
MAX_RETRIES = 3

# Load prompt
with open("Backend/app/LLM/test_case_prompt.txt", "r", encoding="utf-8") as f:
    INSTRUCTIONS = f.read()

class JSONResponseHandler:
    @staticmethod
    def extract_json_from_text(text: str) -> str:
        """Extract JSON from text that might contain other content"""
        # Remove code block markers
        text = text.replace("```json", "").replace("```", "").strip()
        
        # Find the first { and last }
        start_idx = text.find("{")
        end_idx = text.rfind("}") + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON object found in response")
            
        return text[start_idx:end_idx]

    @staticmethod
    def clean_malformed_json(json_str: str) -> str:
        """Clean common JSON formatting issues"""
        # Fix common formatting issues
        json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
        json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
        
        # Fix escaped quotes
        json_str = json_str.replace('\\"', '"')
        json_str = json_str.replace('""', '"')
        
        # Fix newlines in strings
        json_str = re.sub(r'(?<!\\)\n', ' ', json_str)
        
        return json_str

    @staticmethod
    def validate_test_case_structure(test_case: Dict) -> Dict:
        """Validate and fix test case structure"""
        required_fields = {
            "id": str,
            "title": str,
            "steps": list,
            "expected_result": str,
            "priority": str
        }
        
        # Ensure all required fields exist with correct types
        for field, field_type in required_fields.items():
            if field not in test_case:
                if field == "steps":
                    test_case[field] = ["Step 1: No steps provided"]
                elif field == "priority":
                    test_case[field] = "Medium"
                else:
                    test_case[field] = f"Missing {field}"
            elif not isinstance(test_case[field], field_type):
                if field == "steps" and isinstance(test_case[field], str):
                    test_case[field] = [test_case[field]]
                elif field_type == str:
                    test_case[field] = str(test_case[field])

        # Fix step formatting
        test_case["steps"] = [
            f"Step {i+1}: {step.split(': ')[-1]}" if not step.startswith(f"Step {i+1}: ") else step
            for i, step in enumerate(test_case["steps"])
        ]
        
        # Validate priority
        if test_case["priority"] not in ["High", "Medium", "Low"]:
            test_case["priority"] = "Medium"
            
        return test_case

    @staticmethod
    def parse_and_validate_response(response_text: str, story_id: str, current_count: int) -> Dict:
        """Parse and validate the complete response"""
        try:
            # Extract JSON
            json_str = JSONResponseHandler.extract_json_from_text(response_text)
            
            # Clean JSON
            json_str = JSONResponseHandler.clean_malformed_json(json_str)
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Validate basic structure
            if not isinstance(data, dict) or "test_cases" not in data:
                raise ValueError("Invalid response structure")
                
            # Validate and fix each test case
            for i, tc in enumerate(data["test_cases"]):
                # Ensure correct ID
                tc["id"] = f"{story_id}-TC{current_count + i + 1}"
                data["test_cases"][i] = JSONResponseHandler.validate_test_case_structure(tc)
                
            return data
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print("Attempting to fix malformed JSON...")
            # Create a minimal valid response
            return {
                "test_cases": [{
                    "id": f"{story_id}-TC{current_count + 1}",
                    "title": "Error in test case generation",
                    "steps": ["Step 1: Verify basic functionality"],
                    "expected_result": "System functions as expected",
                    "priority": "Medium"
                }]
            }
        except Exception as e:
            print(f"Error in response parsing: {str(e)}")
            raise

class TestCaseGenerator:
    def __init__(self):
        self.prompt_file = os.path.join(os.path.dirname(__file__), "test_case_prompt.txt")
        with open(self.prompt_file, "r") as f:
            self.base_prompt = f.read()

    def validate_test_cases(self, test_cases: List[Dict]) -> bool:
        """Validate the generated test cases meet requirements"""
        if not test_cases:
            print("No test cases provided")
            return False

        for tc in test_cases:
            # Validate required fields
            if not all(key in tc for key in ["id", "title", "steps", "expected_result", "priority"]):
                print(f"Missing required fields in test case: {tc.get('id', 'unknown')}")
                return False

            # Validate steps format
            for i, step in enumerate(tc["steps"], 1):
                if not step.startswith(f"Step {i}: "):
                    print(f"Invalid step format in test case {tc['id']}, step {i}: {step}")
                    print("Steps must start with 'Step N: '")
                    return False

            # Validate priority
            if tc["priority"] not in ["High", "Medium", "Low"]:
                print(f"Invalid priority in test case {tc['id']}: {tc['priority']}")
                return False

        return True

    def get_story_complexity(self, story_description: str, main_text: str) -> str:
        """Determine story complexity based on content length and keywords"""
        # Combine all text
        full_text = f"{story_description} {main_text}".lower()
        
        # Count complexity indicators
        integration_keywords = ["integrate", "api", "database", "external", "system", "interface"]
        security_keywords = ["auth", "security", "permission", "role", "access"]
        complexity_keywords = ["complex", "multiple", "various", "different", "workflow"]
        
        # Calculate scores
        text_length_score = len(full_text) / 500  # Normalize by typical story length
        integration_score = sum(1 for word in integration_keywords if word in full_text)
        security_score = sum(1 for word in security_keywords if word in full_text)
        complexity_score = sum(1 for word in complexity_keywords if word in full_text)
        
        total_score = text_length_score + integration_score + security_score + complexity_score
        
        # Determine complexity level
        if total_score > 10:
            return "high"
        elif total_score > 5:
            return "medium"
        else:
            return "low"

    def get_category_counts(self, complexity: str) -> Dict[str, int]:
        """Get test case counts based on story complexity"""
        base_counts = {
            "low": {
                "positive": 8,
                "negative": 6,
                "boundary": 4,
                "security": 3,
                "performance": 2
            },
            "medium": {
                "positive": 15,
                "negative": 10,
                "boundary": 6,
                "security": 5,
                "performance": 4
            },
            "high": {
                "positive": 25,
                "negative": 15,
                "boundary": 8,
                "security": 7,
                "performance": 5
            }
        }
        
        # Add some randomization (±20%)
        counts = base_counts[complexity]
        return {
            category: max(1, int(count * random.uniform(0.8, 1.2)))
            for category, count in counts.items()
        }

    def generate_test_cases(self, story_id: str, story_description: str) -> Dict:
        """Generate test cases for a given user story"""
        try:
            # Initialize the final result
            final_test_cases = {
                "storyID": story_id,
                "storyDescription": story_description,
                "generated_on": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "total_test_cases": 0,
                "test_cases": []
            }
            
            # Get story complexity and determine test case counts
            complexity = self.get_story_complexity(story_description, story_description)
            category_counts = self.get_category_counts(complexity)
            
            # Define test case categories with dynamic counts
            categories = [
                {
                    "type": "positive",
                    "count": category_counts["positive"],
                    "focus": ["Core functionality", "Happy path scenarios", "Main user workflows"]
                },
                {
                    "type": "negative",
                    "count": category_counts["negative"],
                    "focus": ["Error handling", "Invalid inputs", "Exception scenarios"]
                },
                {
                    "type": "boundary",
                    "count": category_counts["boundary"],
                    "focus": ["Edge cases", "Limit testing", "Boundary conditions"]
                },
                {
                    "type": "security",
                    "count": category_counts["security"],
                    "focus": ["Authentication", "Authorization", "Data protection", "Security vulnerabilities"]
                },
                {
                    "type": "performance",
                    "count": category_counts["performance"],
                    "focus": ["Response time", "Load handling", "Resource usage", "Scalability"]
                }
            ]
            
            # Generate test cases in batches for each category
            for category in categories:
                remaining = category["count"]
                while remaining > 0:
                    batch_count = min(BATCH_SIZE, remaining)
                    batch = self.generate_test_cases_batch(
                        story_id,
                        story_description,
                        category["type"],
                        category["focus"],
                        len(final_test_cases["test_cases"]),
                        batch_count
                    )
                    
                    if "test_cases" in batch:
                        final_test_cases["test_cases"].extend(batch["test_cases"])
                        remaining -= len(batch["test_cases"])
                    else:
                        print(f"Warning: No test cases generated for {category['type']} batch")
                        break
            
            # Update total count
            final_test_cases["total_test_cases"] = len(final_test_cases["test_cases"])
            
            return final_test_cases
            
        except Exception as e:
            print(f"Error generating test cases: {e}")
            raise

    def generate_test_cases_batch(self, story_id: str, story_description: str, test_type: str, focus_areas: List[str], current_count: int, batch_size: int) -> Dict:
        """Generate a batch of test cases for a given category"""
        for attempt in range(MAX_RETRIES):
            try:
                # Randomize the actual batch size slightly
                actual_batch_size = random.randint(max(1, batch_size - 2), batch_size + 2)
                
                # Create a focused prompt for this specific batch
                batch_prompt = f"""You are a Senior QA Architect with 15+ years of experience in enterprise software testing.
Your task is to generate exactly {actual_batch_size} {test_type} test cases for this user story.

Story ID: {story_id}
Description: {story_description}

Focus Areas for this batch:
{chr(10).join(f"- {area}" for area in focus_areas)}

Requirements:
1. Generate ONLY {test_type} test cases
2. Each test case must be unique and detailed
3. Focus on the areas listed above
4. Include specific validation points
5. Consider error scenarios and edge cases

CRITICAL: Your response MUST be a valid JSON object with this exact structure:
{{
    "test_cases": [
        {{
            "id": "{story_id}-TC{current_count + 1}",
            "title": "Detailed description of what is being tested",
            "steps": [
                "Step 1: Detailed step with specific actions",
                "Step 2: Detailed step with validation points"
            ],
            "expected_result": "Comprehensive description of expected outcomes",
            "priority": "High" | "Medium" | "Low"
        }}
    ]
}}

Remember:
- Be extremely detailed and specific
- Include all necessary validation points
- Make steps clear and actionable
- Include specific test data
- Return ONLY the JSON object, no other text
- Do not use markdown code blocks"""

                # Call LLM
                response = Config.llm.invoke(batch_prompt)
                response_text = response.content.strip()
                
                # Use the new JSON handler to parse and validate the response
                batch_test_cases = JSONResponseHandler.parse_and_validate_response(
                    response_text,
                    story_id,
                    current_count
                )
                
                if batch_test_cases and batch_test_cases.get("test_cases"):
                    print(f"✅ Successfully generated {len(batch_test_cases['test_cases'])} {test_type} test cases")
                    return batch_test_cases
                    
            except Exception as e:
                print(f"Attempt {attempt + 1}/{MAX_RETRIES} failed: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    print("Retrying...")
                    continue
                else:
                    print("All retries failed, returning minimal valid response")
                    return {
                        "test_cases": [{
                            "id": f"{story_id}-TC{current_count + 1}",
                            "title": f"Basic {test_type} test case",
                            "steps": ["Step 1: Verify basic functionality"],
                            "expected_result": "System functions as expected",
                            "priority": "Medium"
                        }]
                    }

    def format_test_cases(self, test_cases: Dict) -> Dict:
        """Format and validate the test cases"""
        formatted = {
            "storyID": test_cases["storyID"],
            "storyDescription": test_cases["storyDescription"],
            "generated_on": datetime.now().isoformat(),
            "total_test_cases": len(test_cases["test_cases"]),
            "test_cases": test_cases["test_cases"]
        }
        
        return formatted

def generate_test_case_for_story(story_id, llm_ref=None):
    """Synchronous wrapper for async function"""
    if llm_ref is None:
        llm_ref = Config.llm
    return asyncio.run(_generate_test_case_for_story(story_id, llm_ref))

async def _generate_test_case_for_story(story_id, llm_ref=None):
    """Async implementation of test case generation"""
    if llm_ref is None:
        llm_ref = Config.llm
    
    try:
        # Initialize test case generator
        generator = TestCaseGenerator()
        
        # Get story data from LanceDB
        db = lancedb.connect(Config.LANCE_DB_PATH)
        table = db.open_table(Config.TABLE_NAME_LANCE)
        all_rows = table.to_pandas()
        row_data = all_rows[all_rows['storyID'] == story_id]
        
        if row_data.empty:
            print(f"❌ Story ID '{story_id}' not found in LanceDB.")
            return
        
        row = row_data.iloc[0].to_dict()
        project_id = row.get("project_id", "")
        story_description = row["storyDescription"]
        main_text = row.get("doc_content_text", "").strip()
        
        if not main_text:
            print(f"❌ Skipping {story_id} — missing doc_content_text.")
            return
        
        print(f"🔍 Generating test case for: {story_id} (Project: {project_id})")
        
        # Use the dynamic test case generation
        test_cases = generator.generate_test_cases(story_id, story_description)
        
        if test_cases and test_cases.get("test_cases"):
            # Insert test cases into database
            insert_test_case(
                story_id=story_id,
                story_description=story_description,
                test_case_json=test_cases,
                project_id=project_id,
                source='llm',
                inputs={
                    "story_description": story_description,
                    "main_text": main_text,
                    "project_id": project_id
                }
            )
            print(f"✅ Inserted test cases for {story_id} into Postgres.\n")
            
            # Trigger impact analysis after storing test cases
            print(f"🔄 Triggering impact analysis for {story_id}")
            analyze_test_case_impacts(story_id, project_id)
            print(f"✅ Impact analysis completed for {story_id}\n")
            
            return test_cases
        else:
            print(f"❌ Failed to generate test cases for {story_id}")
            return None
    except Exception as e:
        print(f"❌ Error in test case generation for {story_id}: {e}")
        return None

# === Run for all unprocessed
def generate_test_cases_for_all_stories():
    """Synchronous wrapper for async function"""
    return asyncio.run(_generate_test_cases_for_all_stories())

async def _generate_test_cases_for_all_stories():
    """Async implementation of batch test case generation"""
    db = lancedb.connect(Config.LANCE_DB_PATH)
    table = db.open_table(Config.TABLE_NAME_LANCE)

    generated_ids = set(get_all_generated_story_ids())
    all_rows = table.to_pandas()
    
    print(f"📊 Total stories in LanceDB: {len(all_rows)}")
    print(f"📊 Already generated stories: {len(generated_ids)}")
    
    # Check for missing vectors
    missing_vector_count = 0
    for index, row in all_rows.iterrows():
        vector_value = row.get("vector")
        if vector_value is None or (hasattr(vector_value, '__len__') and len(vector_value) == 0):
            print(f"Story {row['storyID']} is missing a vector and will not be processed.")
            missing_vector_count += 1
    
    print(f"📊 Stories missing vectors: {missing_vector_count}")
    
    # Get all story IDs and filter out already generated ones
    all_story_ids = all_rows['storyID'].tolist()
    records = [story_id for story_id in all_story_ids if story_id not in generated_ids]
    
    print(f"🟡 Found {len(records)} entries to process.\n")
    
    if len(records) == 0:
        print("✅ All stories with vectors have been processed!")
        return
        
    for story_id in records:
        await _generate_test_case_for_story(story_id)

def Chat_RAG(user_query, top_k=3):
    db = lancedb.connect(Config.LANCE_DB_PATH)
    table = db.open_table(Config.TABLE_NAME_LANCE)
    query_vector = Config.EMBEDDING_MODEL.encode(user_query).tolist()

    results = (
        table.search(query_vector)
        .metric("cosine")
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