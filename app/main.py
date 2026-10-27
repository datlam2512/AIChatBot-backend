
import os
import logging

from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Load .env trước khi import LangGraph
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy.engine import make_url

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from langgraph.checkpoint.postgres import PostgresSaver

from app.services.tool_graph import builder

from app.routes import (
    chat,
    conversation,
    message,
    documents,
    auth
)

# ======================================
# 1. LOGGING
# ======================================

logging.basicConfig(level=logging.INFO)

# Hiển thị nguyên nhân lỗi khi Pool
# không thể kết nối PostgreSQL
logging.getLogger("psycopg.pool").setLevel(
    logging.DEBUG
)


# ======================================
# 2. FASTAPI LIFESPAN
# ======================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("[SERVER] Starting FastAPI...")

    # Lấy DATABASE_URL từ .env
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is missing in .env"
        )

    # Chuyển SQLAlchemy URL
    # sang PostgreSQL URI cho psycopg
    url = make_url(database_url)

    db_uri = (
        url.set(drivername="postgresql")
        .render_as_string(hide_password=False)
    )

    # Chỉ in thông tin không nhạy cảm
    print("[DATABASE] Host:", url.host)
    print("[DATABASE] Port:", url.port)
    print("[DATABASE] Database:", url.database)

    # ==================================
    # 3. POSTGRESQL CONNECTION POOL
    # ==================================

    with ConnectionPool(
        conninfo=db_uri,

        min_size=1,
        max_size=5,

        timeout=10,

        kwargs={
            "autocommit": True,
            "row_factory": dict_row,
            "prepare_threshold": 0,
            "connect_timeout": 5,
        },

        open=False,
    ) as pool:

        # Đợi kết nối đầu tiên sẵn sàng
        pool.wait(timeout=10)

        print("[DATABASE] Connection Pool ready")

        # ==================================
        # 4. LANGGRAPH CHECKPOINTER
        # ==================================

        checkpointer = PostgresSaver(pool)

        # Tạo / cập nhật bảng checkpoint
        checkpointer.setup()

        print("[MEMORY] Checkpointer ready")

        # ==================================
        # 5. COMPILE AI AGENT
        # ==================================

        app.state.agent_graph = builder.compile(
            checkpointer=checkpointer
        )

        print("[AGENT] LangGraph Agent ready")

        # FastAPI bắt đầu phục vụ request
        yield

        # ==================================
        # 6. SHUTDOWN
        # ==================================

        print("[SERVER] Shutting down...")

        del app.state.agent_graph

    # Pool được đóng khi thoát khỏi with
    print("[DATABASE] Connection Pool closed")


# ======================================
# 7. FASTAPI APPLICATION
# ======================================

app = FastAPI(
    title="AI Chatbot API",
    description=(
        "AI Chatbot with Ollama, RAG, "
        "LangGraph, Tool Calling and Memory"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ======================================
# 8. CORS
# ======================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],

    allow_credentials=True,

    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================
# 9. API ROUTERS
# ======================================

app.include_router(chat.router)
app.include_router(conversation.router)
app.include_router(message.router)
app.include_router(documents.router)
app.include_router(auth.router)

# ======================================
# 10. HEALTH CHECK
# ======================================

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "AI Chatbot Backend is running",
    }
