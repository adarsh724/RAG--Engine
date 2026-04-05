import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA # Updated import for compatibility

st.set_page_config(page_title="Infallible RAG", layout="wide")
st.title("🧪 Chemistry AI Assistant")

# --- 1. CONFIG ---
# Pro-tip: Use an environment variable or a text input for security
hf_token = st.sidebar.text_input("Enter HF Token", type="password")

@st.cache_resource
def init_rag(file_path):
    loader = PyPDFLoader(file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = loader.load_and_split(text_splitter)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)

    repo_id = "mistralai/Mistral-7B-v0.1"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        huggingfacehub_api_token=hf_token,
        task="text-generation",
        temperature=0.1,
        max_new_tokens=200
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 3})
    )

# --- UI LOGIC ---
uploaded_file = st.sidebar.file_uploader("Upload Chemistry PDF", type="pdf")

if not hf_token:
    st.warning("Please enter your Hugging Face Token in the sidebar.")
elif uploaded_file:
    # Save temp file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    if "qa_chain" not in st.session_state:
        with st.spinner("Connecting to AI Engine..."):
            st.session_state.qa_chain = init_rag("temp.pdf")
            st.sidebar.success("Ready!")

    if user_input := st.chat_input("Ask a chemistry question"):
        st.chat_message("user").write(user_input)
        response = st.session_state.qa_chain.invoke({"query": user_input})
        st.chat_message("assistant").write(response["result"])
else:
    st.info("Please upload a PDF to start.")
