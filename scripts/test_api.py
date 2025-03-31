import requests
import json

def test_api():
    base_url = "http://localhost:7860"
    
    # Test health check
    print("\nTesting health check endpoint...")
    response = requests.get(f"{base_url}/")
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    
    # Test model info
    print("\nTesting model info endpoint...")
    response = requests.get(f"{base_url}/info")
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    
    # Test query endpoint
    print("\nTesting query endpoint...")
    query_data = {
        "text": "请介绍一下溪水旁杂志",
        "language": "simplified"
    }
    response = requests.post(
        f"{base_url}/query",
        json=query_data,
        headers={"Content-Type": "application/json; charset=utf-8"}
    )
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    test_api() 