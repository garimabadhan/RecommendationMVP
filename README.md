# 🎓 MTSS Recommendation MVP

This project provides an intelligent intervention suggestion system for Multi-Tiered System of Supports (MTSS) using AI embeddings and semantic similarity. Built with Streamlit, Google Gemini, and Sentence Transformers.

## 🚀 Features

- PDF chunking from MTSS guides (on-the-fly during deployment)
- SentenceTransformer-based embedding of content chunks
- Embedding of student profiles from BigQuery
- Cosine similarity to find top 5 relevant MTSS documents
- Generative AI (Gemini) to suggest classroom-friendly interventions

## 🛠️ Tech Stack

- **Python 3.10**
- [Streamlit](https://streamlit.io/) – UI
- [Google Generative AI](https://ai.google.dev/) – Gemini content generation
- [Sentence Transformers](https://www.sbert.net/) – Semantic similarity
- [Google BigQuery](https://cloud.google.com/bigquery) – Student data store
- [Docker](https://www.docker.com/) – Deployment containerization

## 🧠 How it works

1. MTSS PDFs are chunked by page using `pdfplumber`
2. Each chunk is embedded using `all-MiniLM-L6-v2` (from HuggingFace)
3. Student embeddings are retrieved from BigQuery and padded to 384D
4. Cosine similarity identifies top 5 similar MTSS chunks per student
5. Gemini generates 3 tailored intervention suggestions from those chunks

## 📂 Project Structure

```
.
├── app.py                  # Main Streamlit app logic
├── Dockerfile              # Deployment config for Cloud Run
├── requirements.txt        # Dependencies (Streamlit, Gemini, BigQuery, etc.)
├── models/                 # Local cache of SentenceTransformer
├── MTSS_PDFs/              # Folder containing MTSS documents (PDF)
├── .env                    # Contains GEMINI_API_KEY (excluded via .gitignore)
├── .gitignore
└── README.md
```

## 🔐 Setup

1. Create a `.env` file:

```env
GEMINI_API_KEY=your-actual-api-key
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run locally:

```bash
streamlit run app.py
```

## ☁️ Deployment

To deploy to Google Cloud Run:

```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/mtss-ui
gcloud run deploy mtss-ui --image gcr.io/YOUR_PROJECT_ID/mtss-ui --platform managed --region us-central1 --memory 1Gi --port 8080
```

## 🤖 Models Used

- **SentenceTransformer**: `all-MiniLM-L6-v2` (local)
- **Gemini**: `models/gemini-1.5-pro` (Generative content model)
- **BigQuery**: For student profile embeddings

## 📌 Notes
- Ensure that the model all-MiniLM-L6-v2 is cached locally and included under /models/.

- MTSS PDFs must be present in MTSS_PDFs/ folder.

- You must have access to the relevant BigQuery tables (student_embeddings) in the specified project.

## 📝 Example Output

- Student ID: 12345
- Profile: GPA: 2.5, Attendance: 3.0, Math: 1.5, Reading: 2.0
- 🔍 System finds 5 most relevant MTSS chunks
- ✨ Gemini suggests 3 actionable interventions

## 🙋‍♀️ Author

Garima Badhan – [@garimabadhan](https://github.com/garimabadhan)