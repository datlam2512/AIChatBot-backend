from fastapi import FastAPI
from app.routes import documents
from fastapi.middleware.cors import CORSMiddleware
from app.routes.conversation import router as conversations_router
from app.routes.message import router as messages_router
from app.routes.chat import router as chat_router
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)
app.include_router(conversations_router)
app.include_router(messages_router)
app.include_router(chat_router)
app.include_router(documents.router)
@app.get("/")
def root():
    return {"Message":"AI backend is running successfully!"}

