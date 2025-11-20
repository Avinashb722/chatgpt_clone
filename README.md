# ChatGPT Clone – Multi-Provider AI Assistant

![ChatGPT Clone UI](ui.png)

A full-stack AI chat application built with **Flask** that lets you talk to multiple AI providers (Groq, Gemini, Local GPT4All), run code in different languages, upload & analyze files, generate **PDF/DOCX/PPT** documents from natural language, and even use your app as an **API** for other clients.

---

## ✨ Features

### 🧠 Multi-Model AI Chat
- Support for:
  - **Groq** (Llama 3.1, Mixtral, etc.)
  - **Google Gemini** (2.0 Flash, 1.5 Pro)
  - **Local models via GPT4All** (Llama, TinyLlama, Mistral, etc.)
- **Auto provider** mode: automatically selects the best available backend.
- Per-chat **conversation memory** saved as JSON files in `chat_history/`.

### 💬 Modern Chat UI
- Clean, ChatGPT-style interface built with **HTML/CSS/JS**.
- Sidebar with:
  - **New chat**
  - **Chat history list**
  - **Clear all history**
  - **Download current chat** (PDF / DOCX / TXT / JSON).
- Status indicator, active model display, typing indicator.

### 🎙 Voice & Smart Input
- **Text-to-speech (TTS)** with multiple voices (Microsoft / Google voices where available).
- Speech speed toggle.
- Message **autocomplete/suggestions** while typing.
- Keyboard navigation for suggestions (↑ / ↓ / Enter / Esc).

### 📂 File Upload & OCR
- Upload files from the UI:
  - `.txt`, `.pdf`, `.docx`
  - `.png`, `.jpg`, `.jpeg`, `.gif`
- Backend:
  - Extracts text from PDFs & DOCX.
  - Uses **pytesseract** + Pillow to extract text from images (OCR).
- Extracted content is attached as context for the AI.

### 📄 Document Generator (PDF / DOCX / PPT)
Just ask things like:

> "Create a 7 slide PPT about Artificial Intelligence"  
> "Generate a 5 page PDF document on machine learning basics"  
> "Make a 3 page Word document on cloud computing"

The app will:
- Parse **pages/slides** + **format** from your prompt.
- Generate structured content using the AI.
- Create:
  - Multi-page **PDF** (via `reportlab`)
  - Multi-slide **PPTX** (via `python-pptx`)
  - Multi-page **DOCX** (via `python-docx`)
- Return the file as a download.

### 🖼 Image Generation
- Natural language image generation if your message contains words like:
  - `create image`, `generate image`, `draw`, `make image`
- Uses an external image generation endpoint to return a downloadable `.jpg`.

### 🧪 Built-in Code Runner
- `/run-code` and `/run-interactive` endpoints to execute code.
- Supports:
  - **Python**
  - **JavaScript (Node.js)**
  - **C**
  - **C++**
  - **Java**
- Uses temporary files + `subprocess` under the hood.
- Returns stdout/stderr as JSON so the UI can display execution results.

### 🔑 Simple API Layer
- This project also gives you a ready-made API so other apps can talk to your AI.
- `simple_chatbot.py` shows how to:
  - Send a message to your local API.
  - Read and print AI responses in the terminal.
  - Use an API key style header.
- In short: Your ChatGPT Clone also works as an API, and any app can use it.
---

## 🧱 Tech Stack

- **Backend**
  - Python 3
  - Flask
  - Groq SDK
  - Google Generative AI SDK
  - GPT4All (for local models)
  - PyPDF2, python-docx, python-pptx, reportlab
  - pytesseract, Pillow

- **Frontend**
  - HTML5, CSS3
  - Vanilla JavaScript
  - Font Awesome icons

---

## 📁 Project Structure

```text
chatgpt_clone/
├─ app.py                     # Main Flask application
├─ simple_chatbot.py          # Example CLI client using the API
├─ install_download_feature.bat
├─ requirements.txt
├─ requirements_docs.txt      # Extra doc-related deps (if needed)
├─ templates/
│  └─ index.html              # Main chat UI
├─ static/
│  ├─ style.css               # UI styling
│  └─ script.js               # Frontend logic (chat, voice, uploads, etc.)
├─ chat_history/              # JSON chat logs (auto-created)
└─ uploads/                   # Uploaded files (auto-created)
```

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Avinashb722/chatgpt_clone.git
cd chatgpt_clone
```

### 2. Create & Activate Virtual Environment (Recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

To ensure document download features work properly, you can also run:
```bash
install_download_feature.bat  # on Windows
```
(or just manually make sure reportlab, python-docx, and python-pptx are installed).

### 4. Configure API Keys
In `app.py`, there is an `AI_CONFIGS` dictionary:

```python
AI_CONFIGS = {
    'groq': {
        'api_key': 'gsk_YOUR_GROQ_API_KEY_HERE',
        'models': ['llama-3.1-8b-instant', 'llama-3.1-70b-versatile', 'mixtral-8x7b-32768']
    },
    'gemini': {
        'api_key': 'YOUR_GEMINI_API_KEY_HERE',
        'models': ['gemini-2.0-flash', 'gemini-1.5-pro']
    },
    'local': {
        'models': ['Llama-3.2-1B-Instruct-Q4_0.gguf']
    }
}
```

**Important:**
- Replace the placeholder values with your own API keys.
- For safety, it's better to load them from environment variables instead of hard-coding them.

## 🚀 Running the App

From inside the project directory:
```bash
python app.py
```

By default, the server runs on:
```
http://127.0.0.1:5000
```

Open that in your browser to use the chat UI.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙌 Credits

- Built with Flask, Groq, Google Gemini, GPT4All, and a lot of glue code.
- Frontend inspired by modern AI chat UIs like ChatGPT.
- Developed by **Team Avinash, Anand, Akshay, and Mudhura**.
