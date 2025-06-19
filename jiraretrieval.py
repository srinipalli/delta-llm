 import requests
import pandas as pd
import lancedb


JIRA_BASE_URL = "https://team-delta-innovasolutions.atlassian.net/"
API_KEY = "ATATT3xFfGF0a4KtL9RUftnvMUUAY3JaXH37_pl71TR9i-gOjycj7NoIfmZdVY0N3e3fQJFXZz7FlHToV70yHh8H_Vjpqk5-8ylbzNp6n8LgpK0JsVQTXFFvnvCgU-u_jEwHLQNpfx5os07dRQD2BZ9WLqiBJypvAMQ-yhICh_jQmWrk-fXVbO8=86642574"
EMAIL = "kaushik24062004@gmail.com"
ISSUE_KEY = "TIS"  # Replace with your Jira issue key


 
 

def get_inprogress_stories(jql, max_results=10):
    url = f"{JIRA_BASE_URL}/rest/api/2/search"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    auth = (EMAIL, API_KEY)
    params = {
        "jql": jql,
        "maxResults": max_results
    }
    response = requests.get(url, headers=headers, params=params, auth=auth)
    response.raise_for_status()
    return response.json()

if __name__ == "__main__":
    # JQL to get all stories that are In Progress
    jql_query = 'status = "Selected for Development" AND issuetype = Story'
    stories = get_inprogress_stories(jql_query)

    # Extract relevant fields from each issue
    data = []
    for issue in stories.get("issues", []):
        fields = issue.get("fields", {})
        data.append({
            "key": issue.get("key"),
            "summary": fields.get("summary"),
            "status": fields.get("status", {}).get("name"),
            "priority": fields.get("priority", {}).get("name"),
            "created": fields.get("created"),
            "updated": fields.get("updated"),
            "description": fields.get("description"),
            "reporter": fields.get("reporter", {}).get("displayName"),
            "creator": fields.get("creator", {}).get("displayName"),
            "project": fields.get("project", {}).get("name"),
            "issuetype": fields.get("issuetype", {}).get("name"),
        })

    df = pd.DataFrame(data)
    print(df.head())

    # Connect to LanceDB (replace with your actual DB path)
    db = lancedb.connect("/tmp/lancedb")
    table_name = "jira_stories"

    # Create table if it doesn't exist
    if table_name not in db.table_names():
        db.create_table(
            table_name,
            schema={
                "project_id": lancedb.types.String(),
                "story_id": lancedb.types.String(),
                "story_description": lancedb.types.String(),
            }
        )

    table = db.open_table(table_name)

    # Insert rows into LanceDB
    records = []
    for _, row in df.iterrows():
        records.append({
            "project_id": row["project"],
            "story_id": row["key"],
            "story_description": row["description"]
        })

    if records:
        table.add(records)
    #print(stories)

 
