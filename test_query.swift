import requests

response = requests.post("http://127.0.0.1:8000/query", json={
    "question": "How is a DIEP flap performed?"
})

print(response.json())
