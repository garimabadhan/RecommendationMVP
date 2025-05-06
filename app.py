import os
import pandas as pd
import numpy as np
import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google.cloud import bigquery
import google.generativeai as genai


from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model_gemini = genai.GenerativeModel("models/gemini-1.5-pro")  




# Load SentenceTransformer embedding model
embedding_model = SentenceTransformer('./models/all-MiniLM-L6-v2')




# Load and embed MTSS PDF chunks
@st.cache_resource
def load_mtss_chunks(folder_path="MTSS_PDFs"):
    mtss_chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            with pdfplumber.open(os.path.join(folder_path, filename)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and len(text) > 100:
                        mtss_chunks.append({
                            'doc_name': filename,
                            'citation': f"{filename} - Page {page_num+1}",
                            'chunk_text': text
                        })
    df = pd.DataFrame(mtss_chunks)
    df['embedding_vector'] = df['chunk_text'].apply(lambda x: embedding_model.encode(x).tolist())
    return df

# Load student embeddings from BigQuery
@st.cache_data(show_spinner=False)
def load_student_embeddings():
    client = bigquery.Client(project="innosights")
    query = """
        SELECT Student_ID, embedding_vector AS student_embedding
        FROM `innosights.CPS_IL_Empower_Fake_Data.student_embeddings`
    """
    df = client.query(query).to_dataframe()
    df['student_embedding'] = df['student_embedding'].apply(lambda x: np.array(x))
    df['projected_embedding'] = df['student_embedding'].apply(lambda x: np.pad(x, (0, 384 - len(x)), 'constant'))
    return df

# Similarity scoring
def get_top_mtss_chunks(student_vector, chunks_df, top_n=5):
    chunk_vectors = np.vstack(chunks_df['embedding_vector'].values)
    similarities = cosine_similarity(student_vector.reshape(1, -1), chunk_vectors)[0]
    chunks_df['similarity'] = similarities
    return chunks_df.sort_values(by='similarity', ascending=False).head(top_n)

# Gemini-based intervention generator
def generate_interventions(student_id, student_vec, top_chunks, original_vec):
    characteristics = ['GPA', 'Attendance', 'Math Level', 'Reading Level']
    performance_context = "\n".join([
        f"- {char}: {round(score, 2)} (scaled 0–4)"
        for char, score in zip(characteristics, original_vec)
    ])
    combined_chunks = "\n\n".join([
        f"Source: {row['citation']}\nContent: {row['chunk_text'][:500]}..."
        for _, row in top_chunks.iterrows()
    ])
    prompt = f"""
You are an AI assistant designed to help teachers implement MTSS strategies.

Student Performance Profile:
{performance_context}

Here are 5 relevant MTSS resource excerpts:
\"\"\"
{combined_chunks}
\"\"\"

Instructions:
- Suggest 3 practical, classroom-friendly interventions based ONLY on the provided MTSS content.
- Focus on the student's weaker areas.
- If no interventions are found, reply exactly: "I don't find any intervention in my knowledge."
"""
    response = model_gemini.generate_content(prompt)
    return response.text

# Streamlit UI
st.set_page_config(page_title="MTSS Recommendations", layout="centered")
st.title("🎓 MTSS Recommendation System")

mtss_df = load_mtss_chunks()
students_df = load_student_embeddings()

student_id = st.selectbox("Select Student ID:", students_df['Student_ID'].tolist())
selected_student = students_df[students_df['Student_ID'] == student_id].iloc[0]

if st.button("Generate Interventions"):
    student_vec = selected_student['projected_embedding']
    original_vec = selected_student['student_embedding']
    top_chunks = get_top_mtss_chunks(student_vec, mtss_df, top_n=5)
    interventions = generate_interventions(student_id, student_vec, top_chunks, original_vec)

    st.markdown(f"### 📌 Student ID: {student_id}")
    st.markdown("### 🎯 Student Performance Profile:")
    for char, score in zip(['GPA', 'Attendance', 'Math Level', 'Reading Level'], original_vec):
        st.markdown(f"- **{char}**: {round(score, 2)} (scaled 0–4)")
    st.markdown("### 📄 Top 5 MTSS Chunks & Citations:")
    st.dataframe(top_chunks[['citation', 'similarity']])
    st.markdown("### 📝 Generated Interventions:")
    st.write(interventions)
