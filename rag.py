import os
import certifi
from pathlib import Path
from typing import List, Optional
import re
from dotenv import load_dotenv

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUEST_CA_BUNDLE"] = certifi.where()

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from pypdf import PdfReader
import docx2txt

Path('uploads').mkdir(exist_ok=True)
Path('chroma_db').mkdir(exist_ok=True)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 150

embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

vectorstore = Chroma(
    collection_name='agentic_chatbot_docs',
    embedding_function=embeddings,
    persist_directory='chroma_db'
)

def clean_filename(raw_name: str) -> str:
    return re.sub(r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}_', '', raw_name)

def extract_section_title(text: str) -> str:
    patterns = [
        r'(Điều\s+\d+[^:\n]*[:\.\-][^\n]+)',
        r'(Mục\s+\d+[^:\n]*[:\.\-][^\n]+)',
        r'(Phần\s+\d+[^:\n]*[:\.\-][^\n]+)',
        r'(Chương\s+[IVXLCDM\d]+[^\n]+)',
        r'(#+\s+[^\n]+)',
        r'(BÊN\s+[AB][^\n]*)'
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip('#').strip()[:60]
    return ""

def read_file_text(file_path: str) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == '.pdf':
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
            text += "\n"
        return text
    elif suffix == '.docx':
        return docx2txt.process(file_path)
    elif suffix in [".txt", ".md", ".py", ".csv"]:
        return path.read_text(encoding='utf-8', errors="ignore")

    raise ValueError("Unsupported file type")

def add_document_to_rag(file_path: str, thread_id: str, original_filename: str | None = None, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
    text = read_file_text(file_path=file_path)

    if not text.strip():
        raise ValueError("No text could be extracted from this file")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_text(text)
    clean_name = original_filename or clean_filename(Path(file_path).name)

    docs: List[Document] = []
    current_section = ""

    for idx, chunk in enumerate(chunks):
        found_section = extract_section_title(chunk)
        if found_section:
            current_section = found_section

        docs.append(
            Document(
                page_content=chunk,
                metadata={
                    "thread_id": thread_id,
                    "source": clean_name,
                    "section": current_section or f"Đoạn {idx + 1}",
                    "chunk_id": idx + 1
                }
            )
        )

    vectorstore.add_documents(docs)

    return {
        "filename": clean_name,
        "chunks": len(docs)
    }

class BM25Retriever:
    def __init__(self, docs: List[Document], k: int = 4):
        self.docs = docs
        self.k = k
        self.corpus = [re.findall(r'\w+', d.page_content.lower()) for d in docs]
        self.bm25 = BM25Okapi(self.corpus) if self.corpus else None

    def invoke(self, query: str) -> List[Document]:
        if not self.bm25 or not self.docs:
            return []
        query_tokens = re.findall(r'\w+', query.lower())
        if not query_tokens:
            return self.docs[:self.k]
        scores = self.bm25.get_scores(query_tokens)
        scored = sorted(zip(self.docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored[:self.k]]

def hybrid_retrieve(query: str, retrievers: List, weights: List[float], k: int = 4, rrf_k: int = 60) -> List[Document]:
    doc_scores = {}

    for retriever, weight in zip(retrievers, weights):
        results = retriever.invoke(query)
        for rank, doc in enumerate(results):
            key = doc.page_content
            rrf_score = weight * (1.0 / (rank + rrf_k))
            if key in doc_scores:
                doc_scores[key] = (doc_scores[key][0] + rrf_score, doc)
            else:
                doc_scores[key] = (rrf_score, doc)

    sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in sorted_docs[:k]]

def retrieve_from_rag(query: str, thread_id: str, k: int = 4, weights: Optional[List[float]] = None) -> str:
    if weights is None:
        weights = [0.5, 0.5]

    vector_retriever = vectorstore.as_retriever(
        search_kwargs={
            "filter": {"thread_id": thread_id},
            "k": k
        }
    )

    all_data = vectorstore.get(where={"thread_id": thread_id})
    all_raw_docs = all_data.get("documents", [])
    all_raw_metas = all_data.get("metadatas", [])

    sparse_docs: List[Document] = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(all_raw_docs, all_raw_metas)
    ]

    bm25_retriever = BM25Retriever(sparse_docs, k=k)

    retrievers = [vector_retriever, bm25_retriever]

    matched_docs = hybrid_retrieve(
        query=query,
        retrievers=retrievers,
        weights=weights,
        k=k
    )

    if not matched_docs:
        return "No relevant uploaded docs content found"

    results = []
    for i, doc in enumerate(matched_docs, start=1):
        source = clean_filename(doc.metadata.get("source", "uploaded document"))
        section = doc.metadata.get("section", f"Đoạn {doc.metadata.get('chunk_id', i)}")
        results.append(
            f"[Nguồn {i}: {source} | Vị trí: {section}]\n{doc.page_content}"
        )

    return "\n\n".join(results)