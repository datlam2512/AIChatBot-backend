from datetime import datetime

from app.database import SessionLocal
from app.models import Conversation, Message


db = SessionLocal()

try:
    now = datetime.now()

    conversation = Conversation(
        title="Learning Python",
        created_at=now,
        updated_at=now,
    )

    db.add(conversation)
    db.flush()

    message1 = Message(
        conversation_id=conversation.id,
        role="user",
        content="Explain Python",
        created_at=now,
    )

    message2 = Message(
        conversation_id=conversation.id,
        role="assistant",
        content="Python is a programming language...",
        created_at=now,
    )

    message3 = Message(
        conversation_id=conversation.id,
        role="user",
        content="What is FastAPI?",
        created_at=now,
    )

    db.add_all([message1, message2, message3])

    db.commit()

    print("Seed data inserted successfully!")
    print("Conversation ID:", conversation.id)

except Exception as e:
    db.rollback()
    print("Failed to insert seed data:")
    print(e)

finally:
    db.close()