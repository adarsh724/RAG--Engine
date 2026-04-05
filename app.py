import streamlit as st
import os
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import FlashrankRerank
from langchain_experimental.text_splitter import SemanticChunker

# --- AGENTIC & CONVERSATIONAL IMPORTS ---
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import nltk
try:
    # This tries to load the component; if it fails, it downloads it
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Suppress warnings
logging.getLogger("transformers.dynamic_module_utils").setLevel(logging.ERROR)

st.set_page_config(page_title="Agentic RAG Engine", layout="wide")
st.title("🤖 Agentic PDF Intelligence — NIT Patna Edition")

# --- SIDEBAR ---
st.sidebar.header("⚙️ System Settings")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

selected_model = st.sidebar.selectbox("Brain (LLM)", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"])

@st.cache_resource
def init_agentic_system(file_path, _api_key, model_name):
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Semantic Chunking
    semantic_splitter = SemanticChunker(
        embeddings_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95
    )
    
    # Persistence Logic
    file_id = os.path.basename(file_path).replace(".pdf", "")
    index_folder = f"faiss_index_{file_id}"
    
    if os.path.exists(index_folder):
        vector_db = FAISS.load_local(index_folder, embeddings_model, allow_dangerous_deserialization=True)
        loader = PyPDFLoader(file_path)
        chunks = loader.load_and_split(semantic_splitter)
    else:
        loader = PyPDFLoader(file_path)
        chunks = loader.load_and_split(semantic_splitter)
        vector_db = FAISS.from_documents(chunks, embeddings_model)
        vector_db.save_local(index_folder)

    # Retrieval Stack
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k = 5
    
    ensemble = EnsembleRetriever(retrievers=[keyword_retriever, vector_retriever], weights=[0.4, 0.6])
    compressor = FlashrankRerank()
    base_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble)

    # --- THE AGENTIC SHIFT ---
    # 1. Create the Tool
    retriever_tool = create_retriever_tool(
        base_retriever,
        "pdf_search",
        "Use this tool to search the uploaded document for specific facts, technical details, or formulas."
    )
    tools = [retriever_tool]

    # 2. LLM Setup
    llm = ChatGroq(model=model_name, api_key=_api_key, temperature=0) # Low temp for accuracy

    # 3. Agent Prompt (The Reasoning Engine)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful academic assistant. "
                   "If the user asks a general question, answer directly. "
                   "If the question requires specific info from the PDF, use the 'pdf_search' tool. "
                   "When you use information from the tool, mention which page it came from if available."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"), # Internal reasoning space
    ])

    # 4. Create Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, return_intermediate_steps=True)

# --- CHAT INTERFACE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if groq_api_key and uploaded_file:
    tmp = f"active_{uploaded_file.name}"
    with open(tmp, "wb") as f: f.write(uploaded_file.getbuffer())

    executor = init_agentic_system(tmp, groq_api_key, selected_model)

    # Display History
    for msg in st.session_state.chat_history:
        st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant").write(msg.content)

    if user_input := st.chat_input("Ask the agent..."):
        st.chat_message("user").write(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = executor.invoke({
                    "input": user_input, 
                    "chat_history": st.session_state.chat_history
                })
                
                st.write(response["output"])

                
                

                st.session_state.chat_history.extend([
                    HumanMessage(content=user_input),
                    AIMessage(content=response["output"])
                ])
else:
    st.info("Upload a PDF and enter API key to start.")


