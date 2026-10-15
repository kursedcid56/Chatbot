import os
from pathlib import Path
import certifi
from tools import tools
from dotenv import load_dotenv
import sqlite3

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUEST_CA_BUNDLE"] = certifi.where()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

Path('data').mkdir(exist_ok=True)

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

ALLOWED_MODELS = {
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.6-flash",
}

SYSTEM_PROMPT = """
You are a helpful Agentic AI assistant named JarvisGPT similar to ChatGPT.

You can:
1. Answer normal questions.
2. Use tools when needed.
3. Search uploaded documents using the RAG tool.
4. Search the web for latest/current information using Tavily Search.
5. Remember important user information using the memory tool.
6. Recall memory when useful.
7. Use calculator for math.

Rules:
- If the user asks about latest news, current events, recent updates, today's information, current prices, etc., use Tavily Search.
- If the user asks about an uploaded document, use search_uploaded_documents.
- If the user asks you to remember something, use remember_this.
- If the user asks about previous preferences or saved facts, use recall_memory.
- Use calculator for math questions.
- When using web search, summarize clearly and mention that the answer is based on web search results.
- When answering from uploaded documents, always cite the source file and specific section/article clearly at the end under a 'Sources:' header in a human-friendly format (e.g. '- [File Name] — [Section / Article / Heading]'). Never display raw UUID hashes or technical chunk numbers.
- Be clear, helpful, and concise.
"""

def normalize_model_name(model_name: str | None) -> str:
    if not model_name:
        return DEFAULT_MODEL
    model_name = model_name.strip()
    if model_name not in ALLOWED_MODELS:
        return DEFAULT_MODEL
    return model_name

def build_agent(model_name: str):
    selected_model = normalize_model_name(model_name)

    llm = ChatGoogleGenerativeAI(
        model=selected_model,
        temperature=0.3,
        streaming=True,
        max_retries=3
    )

    llm_with_tools = llm.bind_tools(tools)

    def chatbot_node(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state['messages']
        response = llm_with_tools.invoke(messages)
        return {
            "messages": [response]
        }

    tool_node = ToolNode(tools)

    workflow = StateGraph(MessagesState)

    workflow.add_node('chatbot', chatbot_node)
    workflow.add_node('tools', tool_node)

    workflow.add_edge(START, 'chatbot')
    workflow.add_conditional_edges('chatbot', tools_condition)
    workflow.add_edge("tools", 'chatbot')

    conn = sqlite3.connect(
        'data/langraph_checkpoints.sqlite',
        check_same_thread=False
    )

    checkpointer = SqliteSaver(conn)

    return workflow.compile(checkpointer=checkpointer)

_AGENT_CACHE = {}

def get_agent(model_name: str | None = None):
    selected_model = normalize_model_name(model_name)

    if selected_model not in _AGENT_CACHE:
        _AGENT_CACHE[selected_model] = build_agent(selected_model)

    return _AGENT_CACHE[selected_model]