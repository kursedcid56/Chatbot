from dotenv import load_dotenv
import os
import certifi
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUEST_CA_BUNDLE"] = certifi.where()

import json
import uuid
import base64
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    ToolMessage
)

from agent import get_agent, ALLOWED_MODELS, DEFAULT_MODEL
from database import (
    init_db,
    save_chat_message,
    get_chat_history,
    create_or_update_conversation,
    list_conversation,
    delete_conversation,
    update_conversation_title
)

from rag import add_document_to_rag
from tools import set_current_thread_id
from security import SecurityPipeline
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Jarvis Agentic Chatbot", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

security_pipeline = SecurityPipeline()

templates = Jinja2Templates(directory="templates")

Path("uploads").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker container orchestration and service monitoring."""
    return {
        "status": "healthy",
        "service": "Jarvis Agentic Chatbot",
        "guardrails": "enabled"
    }

MAX_FILE_SIZE = 15 * 1024 * 1024

@app.get("/uploads/{filename}")
async def get_uploaded_image(filename: str):
    """Serve uploaded chat images while strictly denying access to non-image formats."""
    allowed_image_exts = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    safe_filename = Path(filename).name
    file_path = Path("uploads") / safe_filename
    if not file_path.exists() or file_path.suffix.lower() not in allowed_image_exts:
        return JSONResponse({"error": "File not found or access denied"}, status_code=404)
    return FileResponse(file_path)

init_db()

@app.get("/")
async def home(request: Request):
    """Render primary conversational interface."""
    return templates.TemplateResponse(
        request=request,
        name='index.html',
        context={}
    )

@app.get("/conversations")
async def conversations():
    """List historical conversation threads for sidebar navigation."""
    items = list_conversation()
    return {
        "conversations": [
            {
                "thread_id": item.thread_id,
                "title": item.title,
                "created_at": item.created_at.isoformat(),
                "update_at": item.updated_at.isoformat()
            }
            for item in items
        ]
    }

@app.get("/history/{thread_id}")
async def history(thread_id: str):
    """Hydrate chat window with historical messages for the selected thread."""
    messages = get_chat_history(thread_id=thread_id)
    return { 
        'messages': [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
        ]
    }

@app.delete("/conversation/{thread_id}")
async def remove_conversation(thread_id: str):
    """Delete thread and purge all associated conversation history."""
    delete_conversation(thread_id)
    return {"success": True, "message": "Conversation deleted successfully"}

@app.patch("/conversation/{thread_id}/title")
async def rename_conversation(thread_id: str, request: Request):
    """Update title for a specific conversation thread."""
    data = await request.json()
    new_title = data.get("title", "").strip()
    if not new_title:
        return JSONResponse({"error": "Title cannot be empty"}, status_code=400)
    update_conversation_title(thread_id, new_title)
    return {"success": True, "message": "Conversation renamed successfully"}

@app.post("/upload")
@limiter.limit("10/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    thread_id: str = Form(...)
):
    """Stage document, extract textual payload, and index into thread-scoped RAG."""
    try:
        allowed_extensions = [".pdf", ".docx", ".txt", ".md", ".py", ".csv"]

        filename = file.filename or "uploaded_file"
        suffix = Path(filename).suffix.lower()

        if suffix not in allowed_extensions:
            return JSONResponse(
                {
                    "success": False,
                    "message": "Unsupported file type"
                }
            )
        file_id = str(uuid.uuid4())
        safe_filename = filename.replace(" ", "_")
        file_path = f"uploads/{file_id}_{safe_filename}"
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            return JSONResponse(
                {
                    "success": False,
                    "message": "File exceeds maximum allowed size (15MB)"
                },
                status_code=413
            )

        with open(file_path, 'wb') as f:
            f.write(content)

        create_or_update_conversation(thread_id, "Uploaded document")    
        result = add_document_to_rag(
            file_path=file_path,
            thread_id=thread_id,
            original_filename=safe_filename
        )

        return JSONResponse({
            "success": True,
            "message": f"Uploaded {result['filename']} and created {result['chunks']} chunks "
        })

    except Exception as e:
        return JSONResponse(
            {
                "success": False,
                "message": str(e)
            },
            status_code=500
        )

def sse_data(payload: dict) -> str:
    """Format dictionary into standard Server-Sent Events (SSE) data wire format."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

def should_stream_chunk(chunk, metadata) -> bool:
    """Filter out tool execution nodes, tool calls, and raw ToolMessages from client stream."""
    metadata = metadata or {}
    node_name = str(metadata.get("langgraph_node")).lower()

    if 'tool' in node_name:
        return False
    if isinstance(chunk, ToolMessage):
        return False
    if not isinstance(chunk, (AIMessage, AIMessageChunk)):
        return False
    if getattr(chunk, "tool_calls", None):
        return False
    if getattr(chunk, "invalid_tool_calls", None):
        return False

    additional_kwargs = getattr(chunk, "additional_kwargs", {}) or {}
    if additional_kwargs.get("tool_calls"):
        return False

    return True

def extract_text_from_chunk(chunk) -> str:
    """Unpack raw string, list parts, or structured dict payloads into plain text."""
    content = getattr(chunk, "content", "")
    if not content:
        return ""
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    text_parts.append(item['text'])        
                elif isinstance(item.get("text"), str):
                    text_parts.append(item['text'])        
                elif isinstance(item.get("content"), str):
                    text_parts.append(item['content'])        
        return "".join(text_parts)            
    return str(content)

@app.post("/chat/stream")
@limiter.limit("30/minute")
async def chat_stream(request: Request):
    """
    Primary conversational stream endpoint.
    Orchestrates LangGraph agent loop, broadcasts tool call events, and yields token deltas via SSE.
    """
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(
            {"error": "Invalid JSON body"},
            status_code=400
        )

    user_message = data.get("message", "")
    thread_id = data.get("thread_id", "default")
    selected_model = data.get("model", DEFAULT_MODEL)

    if selected_model not in ALLOWED_MODELS:
        return JSONResponse(
            {"error": f"Invalid model '{selected_model}'. Allowed models: {list(ALLOWED_MODELS.keys())}"},
            status_code=400
        )

    image_data = data.get("image", None)
    image_name = data.get("image_name", "image.png")

    if not user_message.strip() and not image_data:
        return JSONResponse(
            {"error": "message or image is required"},
            status_code=400
        )

    # Guardrail Step 1: Input screening & PII sanitization
    if user_message.strip():
        is_safe, sanitized_message, audit_notes = security_pipeline.check_input(user_message)
        if not is_safe:
            rejection_reason = audit_notes[0] if audit_notes else "Blocked: Prompt injection or security violation detected."
            def rejection_generator():
                yield sse_data({"token": f"⚠️ {rejection_reason}"})
                yield sse_data({"done": True})
            return StreamingResponse(
                rejection_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        user_message = sanitized_message

    agent = get_agent(selected_model)

    # Persist uploaded image to disk and generate lightweight URL reference for chat log
    saved_image_path = None
    if image_data and isinstance(image_data, str) and image_data.startswith("data:image"):
        try:
            header, encoded = image_data.split(",", 1)
            ext = header.split(";")[0].split("/")[1]
            if ext == "jpeg":
                ext = "jpg"
            img_id = str(uuid.uuid4())[:8]
            safe_name = f"img_{img_id}.{ext}"
            saved_file_path = Path("uploads") / safe_name
            with open(saved_file_path, "wb") as f:
                f.write(base64.b64decode(encoded))
            saved_image_path = f"/uploads/{safe_name}"
        except Exception:
            pass

    if saved_image_path:
        display_message = f"![{image_name}]({saved_image_path})\n\n{user_message}" if user_message.strip() else f"![{image_name}]({saved_image_path})"
        title_summary = user_message.strip() or f"Image: {image_name}"
    else:
        display_message = user_message
        title_summary = user_message

    create_or_update_conversation(thread_id, title_summary)
    save_chat_message(thread_id, "user", display_message)

    # Bind thread identifier to tool runtime context
    set_current_thread_id(thread_id)

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    def event_generator():
        raw_answer = ""

        try:
            # Construct multimodal or textual HumanMessage payload
            if image_data:
                prompt_text = user_message.strip() or "Please analyze and describe this image in detail."
                input_message = HumanMessage(content=[
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": image_data}
                ])
            else:
                input_message = HumanMessage(content=user_message)

            inputs = {
                "messages": [input_message]
            }

            # Broadcast tool invocation metadata in real-time while accumulating raw tokens.
            # Tradeoff: Buffering full response before streaming introduces higher time-to-first-token (TTFT)
            # compared to immediate raw streaming, but guarantees that the output guardrail cannot be bypassed
            # by streaming unvalidated or harmful tokens directly to the client.
            for chunk, metadata in agent.stream(
                inputs,
                config=config,
                stream_mode="messages"
            ):
                tool_calls = getattr(chunk, "tool_calls", None)
                if tool_calls:
                    for tc in tool_calls:
                        tool_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                        if tool_name:
                            yield sse_data({"tool": tool_name})

                if not should_stream_chunk(chunk, metadata):
                    continue
                token = extract_text_from_chunk(chunk)

                if token:
                    raw_answer += token

            # Post-generation Output Guardrail validation
            validated_answer, _ = security_pipeline.check_output(raw_answer)

            # Stream validated answer in 60-character slices to client
            slice_size = 60
            for i in range(0, len(validated_answer), slice_size):
                yield sse_data({"token": validated_answer[i:i + slice_size]})

            # Record final synthesized response in database after output safety checks
            if validated_answer.strip():
                save_chat_message(thread_id, "assistant", validated_answer)
            yield sse_data({"done": True})

        except Exception as e:
            yield sse_data({"error": str(e)})            
            yield sse_data({"done": True})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no_cache",
            "Connection": "keep-alive",
            "x-Accel-Buffering": "no"
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )