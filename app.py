from dotenv import load_dotenv
import os
import certifi
from fastapi.responses import JSONResponse, StreamingResponse

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUEST_CA_BUNDLE"] = certifi.where()

import json
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI,Request,UploadFile,File,Form
from fastapi.templating import Jinja2Templates

from langchain_core.messages import(
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    ToolMessage
)

from agent import get_agent
from database import(
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

app = FastAPI()

templates  = Jinja2Templates(directory="templates")

Path("uploads").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

init_db()

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name='index.html',
        context={}
    )

@app.get("/conversations")
async def conversations():
    items = list_conversation()

    return {
        "conversations":[
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
    messages = get_chat_history(thread_id=thread_id)

    return { 
        'messages':[
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
        ]
        
    }

@app.delete("/conversation/{thread_id}")
async def remove_conversation(thread_id: str):
    delete_conversation(thread_id)
    return {"success": True, "message": "Conversation deleted successfully"}

@app.patch("/conversation/{thread_id}/title")
async def rename_conversation(thread_id: str, request: Request):
    data = await request.json()
    new_title = data.get("title", "").strip()
    if not new_title:
        return JSONResponse({"error": "Title cannot be empty"}, status_code=400)
    update_conversation_title(thread_id, new_title)
    return {"success": True, "message": "Conversation renamed successfully"}

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    thread_id: str = Form(...)
):
    try:
        allowed_extensions = [".pdf",".docx",".txt",".md",".py",".csv"]

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
        safe_filename = filename.replace(" ","_")
        file_path = f"uploads/{file_id}_{safe_filename}"

        with open(file_path,'wb') as f:
            f.write(await file.read())

        create_or_update_conversation(thread_id,"Uploaded document")    
        result = add_document_to_rag(
            file_path=file_path,
            thread_id=thread_id,
            original_filename=safe_filename
        )

        return JSONResponse({
            "success":True,
            "message":f"Uploaded {result['filename']} and created {result['chunks']} chunks "

        })

    except Exception as e:
        return JSONResponse(
            {
                "success":False,
                "message": str(e)
            },
            status_code=500
        )

def sse_data(payload: dict)->str:
    return f"data: {json.dumps(payload,ensure_ascii=False)}\n\n"

def should_stream_chunk(chunk, metadata) -> bool:
    metadata = metadata or {}

    node_name = str(metadata.get("langgraph_node")).lower()

    if 'tool' in node_name:
        return False
    if isinstance(chunk,ToolMessage):
        return False
    if not isinstance(chunk,(AIMessage,AIMessageChunk)):
        return False
    if getattr(chunk,"tool_calls",None):
        return False
    if getattr(chunk,"invalid_tool_calls",None):
        return False

    additional_kwargs = getattr(chunk,"additional_kwargs",{}) or {}

    if additional_kwargs.get("tool_calls"):
        return False

    return True

def extract_text_from_chunk(chunk)->str:
    content = getattr(chunk,"content", "")

    if not content:
        return ""
    if isinstance(content,list):
        text_parts = []

        for item in content:
            if isinstance(item,str):
                text_parts.append(item)

            elif isinstance(item,dict):
                if item.get("type") =="text" and isinstance(item.get("text"),str):
                    text_parts.append(item['text'])        
                elif isinstance(item.get("text"),str):
                    text_parts.append(item['text'])        
                elif isinstance(item.get("content"),str):
                    text_parts.append(item['content'])        

        return "".join(text_parts)            
    return str(content)

@app.post("/chat/stream")
async def chat_stream(request:Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(
            {"error":"Invalid JSON body"},
            status_code=400
        )

    user_message = data.get("message","")
    thread_id = data.get("thread_id","default")
    selected_model = data.get("model", "gemini-3.5-flash-lite")

    if not user_message.strip():
        return JSONResponse(
                {"error":"message is required"},
                status_code=400
        )
    agent = get_agent(selected_model)

    create_or_update_conversation(thread_id,user_message)
    save_chat_message(thread_id,"user",user_message)

    set_current_thread_id(thread_id)

    config = {
        "configurable":{
            "thread_id":thread_id
        }
    }
    def event_generator():
        final_answer = ""

        try:
            inputs = {
                "messages":[
                    HumanMessage(content=user_message)
                ]
            }
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
                    final_answer += token
                    yield sse_data({"token": token})
            if final_answer.strip():
                save_chat_message(thread_id, "assistant", final_answer)
            yield sse_data({"done": True})

        except Exception as e:
            yield sse_data({"error":str(e)})            
            yield sse_data({"done":True})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":"no_cache",
            "Connection":"keep-alive",
            "x-Accel-Buffering":"no"
        }
    )




if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )