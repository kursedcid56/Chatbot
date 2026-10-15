# Jarvis

Jarvis là ứng dụng AI Chatbot dạng Agentic được xây dựng bằng FastAPI, LangGraph và Google Gemini. Hệ thống hỗ trợ điều phối công cụ, ghi nhớ ngữ cảnh hội thoại, streaming thời gian thực qua SSE, và trích xuất tài liệu bằng cơ chế Hybrid Search kết hợp ChromaDB và BM25 (xếp hạng RRF).

[English](README.md) | [Tiếng Việt](README_vi.md)

## Quick Start

### Cách 1: Chạy bằng Docker (Khuyến nghị)
```bash
# 1. Cấu hình biến môi trường
cp .env.example .env   # điền GOOGLE_API_KEY và TAVILY_API_KEY của bạn

# 2. Build image và khởi chạy container
docker build -t jarvis .
docker run -d -p 8080:8080 --env-file .env jarvis
```
Mở trình duyệt và truy cập: `http://localhost:8080`.

### Cách 2: Chạy trực tiếp trên máy cá nhân (Local)
```bash
# 1. Cài đặt thư viện phụ thuộc
pip install -r requirements.txt

# 2. Cấu hình biến môi trường
cp .env.example .env   # điền GOOGLE_API_KEY và TAVILY_API_KEY của bạn

# 3. Khởi chạy ứng dụng
python app.py
```
Mở trình duyệt và truy cập: `http://localhost:8080`.

## Demo

Video demo: [https://drive.google.com/file/d/1-bW6XUzkcUqfO1DndhW0Ava5fuyrm1o3/view?usp=sharing](https://drive.google.com/file/d/1-bW6XUzkcUqfO1DndhW0Ava5fuyrm1o3/view?usp=sharing)

## Tính năng chính

- Điều phối công cụ linh hoạt (Tìm kiếm web với Tavily, máy tính, lưu trữ bộ nhớ) dựa trên LangGraph
- Tìm kiếm tài liệu Hybrid Search kết hợp vector ngữ nghĩa (ChromaDB) và từ khóa chính xác (BM25) qua thuật toán RRF
- Hệ thống Guardrails hai lớp (Input/Output): phát hiện prompt injection (Anh/Việt), che giấu thông tin cá nhân (PII, mật khẩu/credentials), và ngăn chặn rò rỉ secret ở output
- Giới hạn tần suất request (Rate limiting) theo IP bảo vệ tài nguyên hệ thống (slowapi)
- Streaming phản hồi thời gian thực qua Server-Sent Events (SSE) kèm thông báo trạng thái công cụ
- Lưu trữ lịch sử trò chuyện và quản lý nhiều phiên chat riêng biệt với SQLite
- Giao diện web trực quan, hỗ trợ tải lên file PDF, DOCX, TXT, CSV và phân tích hình ảnh
- Hỗ trợ đóng gói Docker chạy local và cấu hình sẵn luồng CI với GitHub Actions

## Công nghệ sử dụng

- Backend: FastAPI, Uvicorn
- Agent & Mô hình: LangGraph, LangChain, Google Gemini
- Bảo mật & Guardrails: Pipeline bảo mật tùy biến, slowapi
- Tìm kiếm & RAG: ChromaDB, BM25 (rank-bm25), PyPDF, docx2txt
- Công cụ & Dữ liệu: Tavily Search, SQLite
- DevOps & CI: Docker, GitHub Actions

## Hướng dẫn cài đặt chi tiết

### Yêu cầu hệ thống

- Python 3.11 trở lên
- Git

### Cài đặt từng bước

1. Clone repository về máy:
```bash
git clone https://github.com/kursedcid56/Chatbot.git
cd Chatbot
```

2. Tạo và kích hoạt môi trường ảo:
```bash
python -m venv .venv

# Trên Windows:
.venv\Scripts\activate

# Trên Linux/macOS:
source .venv/bin/activate
```

3. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

4. Cấu hình biến môi trường:
Tạo file `.env` tại thư mục gốc (hoặc sao chép từ `.env.example`):
```env
GOOGLE_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.5-flash-lite
TAVILY_API_KEY=your_tavily_api_key
```

5. Khởi chạy ứng dụng:
```bash
python app.py
```
Truy cập ứng dụng tại `http://localhost:8080`.

## Chạy bằng Docker (Local)

```bash
docker build -t jarvis .
docker run -d -p 8080:8080 --env-file .env jarvis
```

## Giấy phép

Phát hành theo giấy phép Apache 2.0.
