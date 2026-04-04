import os
import base64
import json

from dotenv import load_dotenv
load_dotenv(".env")

api_key = os.getenv("API_KEY")
if not api_key:
    api_key = "your-secret-api-key-here"

from fastapi.testclient import TestClient
from app.main import app

def main():
    file_path = "sample3.jpg"
    print(f"Reading image from {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"File {file_path} not found!")
        return
        
    with open(file_path, "rb") as image_file:
        file_bytes = image_file.read()
        b64_content = base64.b64encode(file_bytes).decode('utf-8')
    
    print("Processing document with NLP Pipeline...")
    with TestClient(app) as client:
        response = client.post(
            "/document/analyze",
            headers={"x-api-key": api_key},
            json={
                "fileName": "sample3.jpg",
                "fileType": "jpg",
                "fileBase64": b64_content
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n================== API RESPONSE ==================")
            print(json.dumps(result, indent=2))
            print("==================================================")
        else:
            print(f"\nError ({response.status_code}):")
            print(response.text)

if __name__ == "__main__":
    main()
