import httpx
import json
def generate_response(message: list[dict]):
    with httpx.stream(
        "POST",
        "http://localhost:11434/api/chat",
        json = {
            "model": "qwen3:4b",
            "messages":message,
            "stream": True
        },
        timeout = 120
    ) as response:

      response.raise_for_status()
      for line in response.iter_lines():
         if not line:
            continue
         data = json.loads(line)
         content = data["message"]["content"]
         if content:
            yield content

