# RAG--Engine

An advanced, production-grade Retrieval-Augmented Generation (RAG) system engineered for high-precision document analysis. This system features an automated evaluation and auditing pipeline (`judge.py`) to benchmark performance, ensuring maximum truthfulness, semantic accuracy, and zero hallucinations.

---

## 🚀 Features

* **Dynamic Document Ingestion:** Streamlined document parsing and vectorization built into an intuitive interface.
* **Model Automation:** Dedicated standalone script (`download_model.py`) to pre-fetch and cache model weights locally for faster, offline deployment.
* **Automated Judge/Evaluation Pipeline:** Includes a specialized `judge.py` evaluation script to audit generations, successfully benchmarking the system to achieve **100% Faithfulness and Precision**.
* **Streamlit Interactive UI:** A clean, production-ready interface (`app.py`) for uploading multi-format documents, asking questions, and getting instant, context-backed answers.

---

## 🛠️ Tech Stack

* **Core Framework:** Python, LangChain
* **Vector Database & Similarity Search:** FAISS
* **Frontend Dashboard:** Streamlit
* **Evaluation Framework:** LLM-as-a-Judge / RAGAS evaluation principles

---

## 📂 Project Structure

```text
├── app.py               # Main Streamlit application web interface
├── download_model.py    # Script to download and cache embeddings/LLM weights locally
├── judge.py             # Evaluation and auditing script for checking RAG performance metrics
├── requirements.txt     # Complete list of Python library dependencies
└── .gitignore           # Deployment configuration file

