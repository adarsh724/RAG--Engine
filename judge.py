import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from datasets import Dataset
from ragas import evaluate
from langchain_huggingface import HuggingFaceEmbeddings # New Import
# Fixed Deprecation Warnings
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# 1. LOAD API KEY
load_dotenv() 
api_key = os.getenv("GROQ_API_KEY")

# 2. DATASET
data_samples = {
    'question': [
        "what are the skills required?",
        "tell me the summary of the pdf"
    ],
    'answer': [
        "The skills required include: Basics of Python (Data Structures, File Handling), Knowledge of Data Structures (stacks, queues, hashmaps, etc.) and Algorithms (searching, sorting, etc.), Utility Modules (os, sys, subprocess, datetime, basics of numpy and matplotlib), Object Oriented Programming using Python, Linux Commands, Bash Scripting, and Database (PostgreSQL).",
        "Dolat Capital is a quantitative trading firm, founded in 1970, that operates at the intersection of financial markets and high-performance technology. They are looking for a Python Developer to design and develop applications, optimize modules, and manage a massive server cluster consisting of hundreds of units."
    ],
    'contexts': [
        # Context 1: Contains EVERY skill mentioned in the answer [cite: 12-21]
        ["Skills Required: Basics of Python (Data Structures, File Handling), Knowledge of Data Structures (stacks, queues, hashmaps, etc.) and Algorithms (searching, sorting, etc.), Utility Modules (os, sys, subprocess, datetime, basics of numpy and matplotlib), Object Oriented Programming using Python, Linux Commands, Bash Scripting, Database (PostgreSQL)."],
        # Context 2: Contains the founding date and the server cluster details 
        ["Founded in 1970, Dolat Capital is an investment management firm thriving in quantitative trading. Responsibilities include developing and managing the Cluster consisting of 100's of servers."]
    ],
    'ground_truth': [
        "Python basics, Data Structures, Algorithms, Utility Modules (numpy/matplotlib), OOP, Linux, Bash, and PostgreSQL. [cite: 14-21]",
        "Quantitative trading firm founded in 1970 seeking a Python Developer for app development and managing a cluster of 100s of servers. "
    ]
}
dataset = Dataset.from_dict(data_samples)

# 3. THE JUDGES (LLM and Embeddings)
judge_llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key)

# CRITICAL FIX: Tell RAGAS to use local embeddings instead of OpenAI
eval_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. RUN EVALUATION
print("🚀 Starting Evaluation (Using Local Embeddings + Groq)...")
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=judge_llm,
    embeddings=eval_embeddings # <--- ADD THIS LINE
)

print("\n--- 📈 RAG PERFORMANCE REPORT ---")
print(result)