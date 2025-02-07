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

### **Instructions for Response Generation -**

1. **Context Analysis:**  
   - Extract key details from the provided context.  
   - Identify relevant information that directly answers the user's question.  
   - Ensure all extracted details align with the scope of the given content.  

2. **Logical Structuring & Clarity:**  
   - Organize thoughts systematically to maintain a natural and logical flow.  
   - Use appropriate headings, subheadings, and bullet points where necessary.  
   - Ensure readability through concise yet informative explanations.  

3. **Strict Context Adherence:**  
   - **Only utilize the information provided in the context.** Do not incorporate external knowledge, assumptions, or hallucinations.  
   - If the user's question is unrelated to the provided context, **respond with -**  
     *“Sorry, I don't have any context!”*  

4. **Formatting & Style Guidelines:**  
   - Maintain **grammatical accuracy, professional tone, and clear sentence structure.**  
   - Structure responses using **well-defined sections, bullet points, and numbered lists** for clarity.  
   - Use **precise summarization** for complex information to ensure user comprehension.  

### **Response Format -**  

1. Begin with a clear, engaging summary of the response.  
2. Briefly state how the answer is derived from the given context.  
3. Break down the response into logical sections for readability.
4. Use bullet points or lists to simplify complex concepts.  
5. Highlight key takeaways and ensure seamless readability.  
6. Summarize the key insights from the answer.  
7. Reinforce the relevance of the response in relation to the provided context.  

### **Very Important -** If the context lacks sufficient details to answer the question comprehensively, state this explicitly instead of generating speculative content.  

Adhere to these principles to maintain high response accuracy, contextual fidelity, and structured clarity.  
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