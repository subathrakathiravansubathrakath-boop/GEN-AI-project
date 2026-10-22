# GenAI Capstone — Study & Career Assistant

A beginner-friendly Generative AI application demonstrating LLM API integration, prompt engineering, Retrieval-Augmented Generation (RAG), embeddings, tool calling, and session state.

## Features
- Study, coding, interview, and career Q&A
- Chat history during the current session
- Prompt Engineering Lab for structured prompts
- Upload PDF, DOCX, TXT, or Markdown files and ask questions
- Text extraction and overlapping chunking
- Embeddings and cosine-similarity retrieval
- LLM answers grounded in retrieved document passages
- Function calling tools: calculator, current date, and word/character count
- Retrieved passages and tool-call traces for transparency
- Basic error handling and safety instructions

## Architecture

```text
General chat:
User → Prompt/Instructions → LLM API → Optional Tool Call → Tool Result → LLM → Answer

Document Q&A (RAG):
Document → Text Extraction → Chunking → Embeddings → Similarity Retrieval
         → Relevant Context + Question → LLM → Answer
```

## Run locally
1. Install Python 3.10 or later.
2. Open a terminal in this project folder.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the application:
   ```bash
   streamlit run app.py
   ```
5. Enter your OpenAI API key in the sidebar.

## GitHub upload
1. Create a new repository on GitHub.
2. Upload `app.py`, `requirements.txt`, `README.md`, and `.gitignore`.
3. Never upload your API key or `.env` file.

## Live deployment
1. Push the project to GitHub.
2. Open Streamlit Community Cloud: https://share.streamlit.io/
3. Connect GitHub, create a new app, and select `app.py`.
4. In app settings, add a secret named `OPENAI_API_KEY` if you adapt the code to read it from `st.secrets`.
5. Deploy and test the live app.

## Demo prompts
- Explain recursion in Python with a simple example.
- Create a 7-day study plan for a Python interview.
- Count the words in: "Generative AI can support learning."
- Calculate `(245 * 18) / 3`.
- Upload your course notes and ask: "Summarize the main concepts."

## Cost and privacy
LLM and embedding API calls may incur charges. Avoid uploading sensitive/private documents. Keep API keys private. This app is a learning prototype and does not guarantee factual correctness.
