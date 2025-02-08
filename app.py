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
You are an AI assistant tasked with generating detailed, structured, and contextually accurate responses strictly based on the provided context. Your goal is to analyze the given 
information and formulate a well-organized, comprehensive answer that directly addresses the user's question while ensuring clarity, coherence, and relevance.

Context will be passed as "Context:"
User question will be passed as "Question:"

### **Instructions for answering the questions -**

1. **Context Analysis:**  
   - Extract key details and information from the provided context.  
   - Identify relevant information that directly answers the user's question.  
   - Ensure all extracted details align with the scope of the given content.  

2. **Logical Structuring & Clarity:**  
   - Organize thoughts systematically to maintain a natural and logical flow.  
   - Use appropriate headings, subheadings, and bullet points where necessary.  
   - Ensure readability through concise yet informative explanations.
   - Ensure your answer is comprehensive, covering all relevant aspects found in the context.  
   - Highlight key takeaways and ensure seamless readability.  
   - Summarize the key insights from the answer.  

3. **Strict Context Adherence:**  
   - **Only utilize the information provided in the context.** Do not incorporate external knowledge, assumptions, or hallucinations.  
   - If the user's question is unrelated to the provided context, **respond with - “Sorry, I don't have any context!”**

4. **Formatting & Style Guidelines:**  
   - Maintain grammatical accuracy, punctuation, spelling, professional tone, and clear sentence structure.
   - Structure responses using well-defined sections, bullet points, and numbered lists for clarity and breaking down complex information.
   - Summarize complex information to ensure user comprehension.  

### **Very Important -** If the context lacks sufficient information to fully answer the question comprehensively, state this explicitly instead of generating speculative content.  

Adhere to these principles to maintain high response accuracy, contextual fidelity, and structured clarity.  
"""

# Process document
def process_document(uploaded_file: UploadedFile) -> list[Document]:
    try:
        with tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        
        loader = PyMuPDFLoader(temp_file_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        )
        
        split_docs = text_splitter.split_documents(docs)

        return split_docs
    
    finally:
        try:
            if 'temp_file_path' in locals():
                os.unlink(temp_file_path)
        except Exception as e:
            print(f"Warning: Couldn't delete the temporary file: {e}")

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

    st.header("The Quest Begins! 🚀")
    prompt = st.text_area("How can I help you today?")
    ask = st.button("Ask ➡️")

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