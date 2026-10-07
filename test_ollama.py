import httpx

print("Sending request...")

response = httpx.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "qwen3:4b",
        "messages": [
            {
                "role": "user",
                "content": "Say hello."
            }
        ],
        "stream": False
    },
    timeout=120
)

print("Response received!")
print(response.status_code)
print(response.json())