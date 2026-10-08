from fastapi import FastAPI
from pydantic import BaseModel
from app.services.llm_service import generate_response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from app.routes.conversation import router as conversations_router
from app.routes.message import router as messages_router
app = FastAPI()
app.include_router(conversations_router)
app.include_router(messages_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

class ChatMessage(BaseModel):
    role:str
    content:str
    
class ChatRequest(BaseModel):
    messages: list[ChatMessage]

@app.get("/")
def root():
    return {"Message":"AI backend is running successfully!"}

@app.post("/api/chat")
def chat(request: ChatRequest):
    return StreamingResponse(
        generate_response(
            [message.model_dump() for message in request.messages]
        ),
        media_type="text/plain",
    )
    