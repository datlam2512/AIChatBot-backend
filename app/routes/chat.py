from pydantic import BaseModel
from app.services.llm_service import generate_response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import BaseModel


class ChatMessage(BaseModel):
    role:str
    content:str
    
class ChatRequest(BaseModel):
    messages: list[ChatMessage]

@app.post("/api/chat")
def chat(request: ChatRequest):
    return StreamingResponse(
        generate_response(
            [message.model_dump() for message in request.messages]
        ),
        media_type="text/plain",
    )