# 🎥 MultiModal RAG with Videos

A multimodal Retrieval-Augmented Generation (RAG) application that enables users to interact with YouTube videos through natural language questions.

The system processes video content by combining visual information extracted from video frames with audio transcripts. It uses multimodal embeddings and vector search to retrieve relevant content, allowing an LLM to generate context-aware answers grounded in the video.

## ✨ Features

- **YouTube Video Ingestion** — Download videos directly from YouTube.
- **Video Frame Extraction** — Extract representative frames for visual analysis.
- **Audio Transcription** — Convert speech into text using OpenAI Whisper.
- **Visual Embeddings** — Generate image embeddings using CLIP.
- **Multimodal Retrieval** — Retrieve relevant video frames and transcript segments using semantic search.
- **AI-Powered Q&A** — Generate answers based on retrieved visual and textual context.
- **Interactive Chat Interface** — Ask questions about video content through a Streamlit application.
- **Video-Based Caching** — Reuse processed data for previously ingested videos.

## 🏗️ Architecture

The application follows a multimodal RAG pipeline:

```text
          YouTube URL
               │
               ▼
        Video Download
               │
       ┌───────┴────────┐
       ▼                ▼
  Frame Extraction  Audio Extraction
       │                │
       ▼                ▼
  CLIP Embeddings   Whisper Transcription
       │                │
       └───────┬────────┘
               ▼
       Multimodal Indexing
               │
               ▼
          LanceDB
               │
               ▼
       User Question
               │
               ▼
       Semantic Retrieval
               │
               ▼
     Retrieved Frames + Text
               │
               ▼
        OpenAI LLM
               │
               ▼
          Final Answer
```

### Processing Pipeline

1. **Ingestion:** The application downloads a YouTube video and extracts its metadata.
2. **Preprocessing:** Video frames and audio are extracted for further processing.
3. **Transcription:** Whisper converts the audio track into text.
4. **Embedding Generation:** CLIP generates visual embeddings for extracted frames.
5. **Indexing:** Processed multimodal data is stored in LanceDB.
6. **Retrieval:** Relevant visual and textual information is retrieved based on the user's question.
7. **Generation:** The retrieved context is passed to an OpenAI model to generate a relevant response.

## 🛠️ Tech Stack

| Category           | Technologies             |
| ------------------ | ------------------------ |
| Language           | Python                   |
| LLM                | OpenAI API (GPT-4o-mini) |
| RAG Framework      | LlamaIndex               |
| Computer Vision    | OpenAI CLIP, OpenCV      |
| Speech Recognition | OpenAI Whisper           |
| Vector Database    | LanceDB                  |
| Video Processing   | yt-dlp, MoviePy, FFmpeg  |
| User Interface     | Streamlit                |
| Data Processing    | NumPy, Pillow            |

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd MultiModal-RAG-with-Videos
```

### 2. Create a virtual environment

Python 3.12 is recommended.

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root using `.env.example` as a template.

Add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key
```

## 🚀 Usage

Start the Streamlit application:

```bash
streamlit run app.py
```

Then:

1. Open the application in your browser.
2. Enter a publicly accessible YouTube video URL.
3. Start video processing.
4. Wait for transcription, frame extraction, and indexing to complete.
5. Ask questions about the video content through the chat interface.

The application retrieves relevant visual and textual information to provide context-aware answers.

## 📂 Project Structure

```text
MultiModal-RAG-with-Videos/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
├── .gitignore             # Git ignore configuration
├── README.md              # Project documentation
│
├── video_data/            # Downloaded videos and metadata
├── mixed_data/            # Extracted frames and processed data
└── lancedb/                # Local vector database
```

*Generated data directories are created during application execution and are not required to be committed to the repository.*

## ⚠️ Notes & Limitations

- An OpenAI API key is required for LLM-based response generation.
- Internet access is required to download YouTube videos and communicate with the OpenAI API.
- Processing time depends on video duration, resolution, and available computing resources.
- The quality of generated answers depends on the accuracy of transcription and the relevance of retrieved multimodal context.

## 📌 Future Improvements

- More advanced temporal video retrieval.
- Improved synchronization between video frames and transcript segments.
- More efficient multimodal indexing and retrieval.
- Support for additional video sources and formats.

---