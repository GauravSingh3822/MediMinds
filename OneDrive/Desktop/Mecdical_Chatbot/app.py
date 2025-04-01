import streamlit as st
from langchain_community.llms import Ollama
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from src.prompt import *
# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Check Pinecone API Key
if not PINECONE_API_KEY:
    st.error("Pinecone API key is missing. Check your .env file.")
    st.stop()  # Stop execution if API key is missing

# Download embeddings
embeddings = download_hugging_face_embeddings()

# Define Pinecone index
index_name = "medicalbot"

# Create Pinecone vector search
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize LLM
llm = Ollama(model="llama3.1")  # Ensure correct model name

# Define Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create Retrieval-Augmented Generation (RAG) Chain
question_answering_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answering_chain)

# Streamlit UI

st.set_page_config(page_title="MedicalBot - Q&A", layout="wide")
st.title("🤖 MedicalBot: Ask Your Questions")

# User Input
question = st.text_input("💬 Enter your question:", "")

# Submit Button
if st.button("🔍 Get Answer"):
    if question.strip():
        response = rag_chain.invoke({"input": question})
        answer = response["answer"]
        sources = response.get("sources", [])
        
        # Display Answer
        st.subheader("📝 Answer:")
        st.write(answer)
        
        # Display Sources
        if sources:
            st.subheader("📌 Sources:")
            for source in sources:
                st.write(f"- {source}")
    else:
        st.warning("⚠️ Please enter a question before submitting.")







# st.title("Medical Chatbot with Ollama")
# st.write("Ask me anything related to medical topics.")

# # Chat input
# user_input = st.text_input("Enter your question:")

# if user_input:
#     response = rag_chain.invoke({"input": user_input})
#     st.write("*Response:*", response["answer"])
