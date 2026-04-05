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
        "What are the technical skills and responsibilities for the Python Developer role?",
        "Provide a brief overview of Dolat Capital's mission and infrastructure."
    ],
    'answer': [
        "Required skills include Python basics, Data Structures, Algorithms, Linux, Bash, and PostgreSQL. Responsibilities involve designing applications, unit testing, and optimizing modules.",
        "Founded in 1970, Dolat Capital uses scientific principles for quantitative trading and manages a cluster of 100's of servers."
    ],
    'contexts': [
        
        ["Skills: Basics of Python, Data Structures, Algorithms, Linux, Bash, PostgreSQL. Responsibilities: Designing and development of applications, Unit testing, and optimizing modules."],
        
        ["Founded in 1970, Dolat Capital pioneers who use scientific principles for quantitative trading. Responsibilities include managing the Cluster consisting of 100's of servers."]
    ],
    'ground_truth': [
        "Proficiency in Python, DS/Algo, Linux/Bash, and PostgreSQL for app development and optimization. [cite: 14-26]",
        "A quantitative firm founded in 1970 using scientific methods to manage a 100+ server cluster. [cite: 2, 4, 28]"
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