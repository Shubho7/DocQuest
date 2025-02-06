import os
import tempfile
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import streamlit as st
from groq import Groq
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

from dotenv import load_dotenv
load_dotenv()

# System prompt
system_prompt = """ 
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a 
comprehensive, well structured response to the question.

Context will be passed as "Context:"
User question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context and identify key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear and concise language.
2. Write an attention-grabbing introduction and a compelling conclusion.
3. Organize your answer into paragraphs for readability.
4. Use bullet points or numbered lists where appropriate to break down complex information.
5. Well-structured body sections with proper headings and subheadings.
6. Ensure proper english grammar, punctuation, style and spelling throughout your answer.
7. Summarize complex information with clarity and precision.

**VERY IMPORTANT:** Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""

# Process document
def process_document(uploaded_file: UploadedFile) -> list[Document]:
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    temp_file.write(uploaded_file.read())

    loader = PyMuPDFLoader(temp_file.name)
    docs = loader.load()
    os.unlink(temp_file.name) 

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )

    return text_splitter.split_documents(docs)

# Vector DB
def get_vector_collection() -> chromadb.Collection:
    google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=os.getenv("GOOGLE_API_KEY"))

    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=google_ef,
        metadata={"hnsw:space": "cosine"},
    )

# Store data in vector DB
def add_to_vector_collection(all_splits: list[Document], file_name: str):
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    st.success("File added successfully!")

# Process user query
def query_collection(prompt: str, n_results: int = 10):
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)

    return results

# Generate response
def llm(context: str, prompt: str):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        stream=True,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Context: {context}, Question: {prompt}",
            },
        ],
    )

    for chunk in response:
         if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

# Re-rank docs
def re_rank(documents: list[str], query: str) -> tuple[str, list[int]]:
    relevant_text = ""
    relevant_text_ids = []
    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    if not documents:
        return "", []
    
    if documents and isinstance(documents[0], list):
        documents = [
            doc[0] if isinstance(doc, list) and len(doc) > 0 else (doc if not isinstance(doc, list) else "")
            for doc in documents
        ]

    ranks = encoder_model.rank(query, documents, top_k=3)
    for rank in ranks:
        corpus_id = rank["corpus_id"]
        if corpus_id < len(documents):
            relevant_text += documents[corpus_id] + "\n"
            relevant_text_ids.append(corpus_id)
       
    return relevant_text, relevant_text_ids

# Main function
if __name__ == "__main__":
    with st.sidebar:
        st.set_page_config(page_title="DocQuest")
        uploaded_file = st.file_uploader(
            "**📑 Upload PDF files**", type=["pdf"], accept_multiple_files=False
        )

        process = st.button("⚡️Process")
        if uploaded_file and process:
            normalize_uploaded_file_name = uploaded_file.name.translate(
                str.maketrans({"-": "_", ".": "_", " ": "_"})
            )
            all_splits = process_document(uploaded_file)
            add_to_vector_collection(all_splits, normalize_uploaded_file_name)

    st.header("Start your Quest")
    prompt = st.text_area("How can I help you today?")
    ask = st.button("Ask")

    if ask and prompt:
        results = query_collection(prompt)
        context_docs = results.get("documents")
        if context_docs:
            relevant_text, relevant_text_ids = re_rank(context_docs, prompt)
            response = llm(context=relevant_text, prompt=prompt)
            st.write_stream(response)

            with st.expander("See retrieved documents"):
                st.write(results)

            with st.expander("See most relevant document IDs"):
                st.write(relevant_text_ids)
                st.write(relevant_text)