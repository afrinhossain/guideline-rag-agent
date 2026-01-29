from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np 

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

#Small "knowledge base"
DOCS =[
    {"id" : "doc1", "text" : "Flu symptoms include fever, cough, sore throat, and fatigue."},
    {"id": "doc2", "text": "Migraine is often a severe headache with nausea and sensitivity to light."},
    {"id": "doc3", "text": "Symptoms of food poisoning depend on the type of germ you swallowed. The most common symptoms include diarrhea, stomach pain or cramps, nausea, vomiting, and fever"}
]

# Precompute embeddings once at startup
DOC_EMB = model.encode([d["text"] for d in DOCS])

class Query(BaseModel):
    text: str

def cosine_sim(a,b):
    a = a / np.linalg.norm(a)
    b = b/ np.linalg.norm(b)
    return float(np.dot(a, b))


"""
@app.get("/")
def read_root():
    return {"status":"ok"}


@app.get("/echo")
def echo(text:str): 
    return {"you sent" : text}

@app.post("/ask")
def ask(query:Query):
    return{"received": query.text}
"""


@app.post("/embed")
def embed_text(text_in: Query):
    vector = model.encode(text_in.text)
    return{
        "vector_length" : len(vector),
        "first_5_values" : vector[:5].tolist()
    }

@app.post("/search")
def search(query : Query):
    q_emb = model.encode(query.text)

    best_doc = None
    best_score = -1.0

    for doc, emb in zip(DOCS, DOC_EMB):
        score = cosine_sim(q_emb, emb)
        if score > best_score:
            best_score = score
            best_doc = doc 

    return{
        "original_query": query.text,
        "best_doc_id": best_doc["id"],
        "best_doc_text" : best_doc["text"],
        "score" : best_score 
    }




