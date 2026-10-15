from datetime import datetime,timedelta,timezone
from pathlib import Path

from sqlalchemy import String, create_engine, Column,Integer,Text,DateTime
from sqlalchemy.orm import declarative_base,sessionmaker


Path("data").mkdir(exist_ok=True)

DATABASE_URL = "sqlite:///data/chatbot_memory.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread":False}
)

SessionLocal = sessionmaker(bind=engine,autoflush=False,autocommit=False)
Base = declarative_base()

VN_TZ = timezone(timedelta(hours=7))
def get_vn_time():
    return datetime.now(VN_TZ).replace(microsecond=0)

class Conversation(Base):
    __tablename__ = "conversation"

    id = Column(Integer,primary_key=True,index=True)
    thread_id = Column(String,unique=True,index=True)
    title = Column(String,default="New chat")
    created_at = Column(DateTime, default=get_vn_time)
    updated_at = Column(DateTime, default=get_vn_time)

class ChatMessage(Base):
    __tablename__ = "chat_messages"    
    
    id = Column(Integer,primary_key=True,index=True)
    thread_id = Column(String,index=True)
    role = Column(String)
    content = Column(Text)
    created_at = Column(DateTime, default=get_vn_time)

class LongTermMemory(Base):
    __tablename__ = "long_term_memory"    

    id = Column(Integer,primary_key=True,index=True)
    thread_id = Column(String,index=True)
    memory = Column(Text)
    created_at = Column(DateTime, default=get_vn_time)

def init_db():
    Base.metadata.create_all(bind=engine)

def create_or_update_conversation(thread_id:str, first_message:str | None=None):
    db = SessionLocal()

    try:
        conversation = (
            db.query(Conversation)
            .filter(Conversation.thread_id==thread_id)
            .first()
        )

        def clean_title(msg: str) -> str:
            msg = msg.strip()
            if msg.startswith("[Attached document:"):
                parts = msg.split("]\n\n", 1)
                if len(parts) > 1 and parts[1].strip():
                    msg = parts[1].strip()
                else:
                    doc_name = msg.replace("[Attached document:", "").split("]")[0].strip()
                    msg = f"Doc: {doc_name}"
            title = msg[:40]
            if len(msg) > 40:
                title += "..."
            return title

        if not conversation:
            title = clean_title(first_message) if first_message else "New chat"

            conversation = Conversation(
                thread_id=thread_id,
                title=title,
                created_at=get_vn_time(),
                updated_at=get_vn_time(),
            )        
            db.add(conversation)
        else:
            if conversation.title.lower() in ["new chat", "uploaded document"] and first_message:
                conversation.title = clean_title(first_message)
            conversation.updated_at = get_vn_time()
        db.commit()    

    finally:
        db.close()    

def list_conversation():
    db=SessionLocal()

    try:
        return(
            db.query(Conversation)
            .order_by(Conversation.updated_at.desc())
            .all()
        )
    finally:
        db.close()

def delete_conversation(thread_id: str):
    db = SessionLocal()
    try:
        db.query(ChatMessage).filter(ChatMessage.thread_id == thread_id).delete()
        db.query(Conversation).filter(Conversation.thread_id == thread_id).delete()
        db.commit()
    finally:
        db.close()

def update_conversation_title(thread_id: str, new_title: str):
    db = SessionLocal()
    try:
        conv = db.query(Conversation).filter(Conversation.thread_id == thread_id).first()
        if conv:
            conv.title = new_title.strip()
            conv.updated_at = get_vn_time()
            db.commit()
    finally:
        db.close()

def save_chat_message(thread_id: str, role:str, content:str):
    db = SessionLocal()

    try:
        msg = ChatMessage(
            thread_id=thread_id,
            role=role,
            content=content,
            created_at=get_vn_time()
        )

        db.add(msg)

        conversation = (
            db.query(Conversation).filter(Conversation.thread_id==thread_id).first()
        )
        if conversation:
            conversation.updated_at=get_vn_time()

        db.commit()

    finally:
        db.close()       

def get_chat_history(thread_id:str):
    db = SessionLocal()
    try:
        return(
            db.query(ChatMessage).filter(ChatMessage.thread_id==thread_id)
            .order_by(ChatMessage.created_at.asc())
            .all()
        )
    finally:
        db.close()
def save_memory(thread_id:str,memory: str):
    db= SessionLocal()

    try:
        item =LongTermMemory(
            thread_id=thread_id,
            memory=memory,
            created_at=get_vn_time()
        )
        db.add(item)
        db.commit()

        return "Memory saved successfully"
    finally:
        db.close()

def search_memory(thread_id: str, query:str):
    db = SessionLocal()
    try:
        memories = (
            db.query(LongTermMemory)
            .filter(LongTermMemory.thread_id==thread_id)
            .order_by(LongTermMemory.created_at.desc())
            .limit(20)
            .all()
        )
        if not memories:
            return "No saved memory found"
        return "\n".join([f"- {m.memory}"for m in memories])
    finally:
        db.close()