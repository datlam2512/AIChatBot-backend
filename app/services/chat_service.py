from app.database import SessionLocal
from app.services.llm_service import generate_response
from app.services.message_service import save_message

def stream_and_save(message: list[dict], conversation_id: int):
    full_response = []
    for chunk in generate_response(message):
        full_response.append(chunk)
        yield chunk

    if full_response:
        with SessionLocal() as db:
            save_message(
                db=db,
                conversation_id = conversation_id,
                role = "assistant",
                content = "".join(full_response)
            )