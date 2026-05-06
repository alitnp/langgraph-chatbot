from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

load_dotenv()

ollama_llm = ChatOllama(model="qwen2.5:14b", temperature=0)
groq_llm = ChatGroq(model="llama-3-70b-8192")
