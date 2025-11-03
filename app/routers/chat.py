from fastapi import APIRouter, Depends, HTTPException
from supabase import Client
from app.database.supabase import get_supabase_client
from app.services.chat_service import ChatService
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

router = APIRouter()
chatService = ChatService()

class ChatRequest(BaseModel):
    message: str

@router.post("/stream")
async def chat_api(chat_request: ChatRequest):
    """챗 api"""
    async def event_stream():
        async for chunk in chatService.stream_chat_data(chat_request.message):
            yield chunk

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"}
    )

    # response_data = ""
    # async for chunk in chatService.stream_chat_data(chat_request.message):
    #     response_data += chunk

    # return {"response": response_data}

@router.get("/ping")
async def ping():
    return {"message": "pong"}
