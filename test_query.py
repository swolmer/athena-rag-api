import requests

response = requests.post("http://127.0.0.1:8000/query", json={
    "question": "What are the anatomical considerations for nasolabial filler injection?"
})
print(response.json())
