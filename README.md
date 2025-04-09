# SHL Gen AI Assessment Recommender

An intelligent recommendation system fOR SHL Product catalog that takes a natural language job description or query and returns the most relevant SHL assessments using LLM-powered parsing, semantic search, and post-filtering logic.

---
## 🚀 Features

- ✅ Natural language input (job descriptions or queries)
- 🤖 LLM-enhanced query understanding using Groq (LLaMA 3.3 70B)
- 🔍 Semantic search with SentenceTransformers + FAISS
- 🎯 Structured post-filtering based on duration, remote support, deduplication
- 🖥️ FastAPI backend + Streamlit frontend
- ☁️ Deployed on Streamlit Cloud and Render

---
## 📸 Demo Screenshots

![Working Backend](https://github.com/user-attachments/assets/72bf99a6-de44-4080-bb5e-9b720d067b62)
![Screenshot 2025-04-10 001406](https://github.com/user-attachments/assets/2147540c-aac7-47b0-a52c-688ae676740e)
![Screenshot 2025-04-10 001509](https://github.com/user-attachments/assets/869f64a9-a14e-43b5-aa31-3148efe528dd)
![Screenshot 2025-04-10 011745](https://github.com/user-attachments/assets/8c16aece-38ca-451c-b181-9effcc37dd9d)
![Screenshot 2025-04-10 011823](https://github.com/user-attachments/assets/1f9e8fa9-89af-4403-aff3-e6070770f28a)
---

## 🧠 Approach

### 1. Data Extraction
- Scraped SHL's assessment catalog using `scraper_shl.py` with Selenium + BeautifulSoup.
- Structured metadata into `shl_assessments_with_ids.csv`.

### 2. Embedding + Indexing
- Encoded assessment data using `SentenceTransformer ('all-MiniLM-L6-v2')`
- Built FAISS vector index using `embedding_index.py` → `shl_index.faiss`

### 3. LLM Query Parsing
- Parsed traits, skills, duration, remote requirement via `llm_query_parser.py`
- LLM used: Groq API with LLaMA 3.3 70B Versatile

### 4. Recommendation Logic
- Main logic inside `main.py` and `recommender_api.py`
- Steps:
  - Parse query (LLM)
  - Semantic search (FAISS)
  - Post-filter (duration, remote, duplicates)

### 5. Interfaces
- **API**: FastAPI (`/recommend`) in `recommender_api.py`
- **Frontend**: Streamlit UI (`main.py`)

---

## 🛠️ Tech Stack

| Category        | Tools / Libraries                      |
|-----------------|-----------------------------------------|
| Scraping        | Selenium, BeautifulSoup                |
| Embedding       | SentenceTransformers, FAISS            |
| LLM Parsing     | Groq API (LLaMA 3.3 70B Versatile)     |
| Backend         | FastAPI                                |
| Frontend        | Streamlit                              |
| Deployment      | Render, Streamlit Cloud                |
| Others          | Pandas, difflib, Pydantic              |

---

## 📁 Project Structure

```bash
├── .devcontainer/                 # Dev container setup
├── models/all-MiniLM-L6-v2/      # Locally downloaded embedding model
├── data_preprocessor.py          # Preprocess scraped data
├── scraper_shl.py                # Scraper for SHL catalog
├── embedding_index.py            # Create embeddings + FAISS index
├── build_faiss_index.py          # CLI wrapper for building index
├── llm_query_parser.py           # Structured query generation using Groq LLM
├── recommender_api.py            # FastAPI backend
├── main.py                       # Streamlit frontend
├── requirements.txt              # Python dependencies
├── .gitignore                    # Ignore rules
├── build.sh                      # Build script for deployment
├── embeddings.npy                # Numpy array of assessment embeddings
├── shl_index.faiss               # FAISS vector index for search
├── shl_assessments_full.csv      # Full scraped assessment data
├── shl_assessments_with_ids.csv  # Processed data with unique IDs
├── shl_indexed_data.csv          # Indexed data with vectors
``` 
## 🌐 Live Links

- **Frontend Demo:** [https://gen-ai-shl-task.streamlit.app](https://gen-ai-shl-task.streamlit.app)  
- **API Endpoint:** [https://gen-ai-shl-task.onrender.com/recommend](https://gen-ai-shl-task.onrender.com/recommend)  
- **GitHub Repository:** [https://github.com/akansh30/SHL_Gen_AI_Task](https://github.com/akansh30/SHL_Gen_AI_Task)

## 📦 How to Run Locally

### 1. Clone Repo
```bash
git clone https://github.com/akansh30/GEN_AI_SHL_TASK.git
cd GEN_AI_SHL_TASK
```
## 📦 How to Run Locally

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Run Backend (FastAPI)
```bash
uvicorn recommender_api:app --reload
```
### 4. Run Frontend (Streamlit)
```bash
streamlit run main.py
```



