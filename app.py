import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceBgeEmbeddings
import os
import re

# Get API keys from Streamlit secrets
groq_api_key = st.secrets["GROQ_API_KEY"]

def get_pdf_text(pdf_docs):
    """Extracts text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Splits extracted text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Creates and saves a FAISS vector store from text chunks."""
    try:
        embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return True
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return False

def get_conversational_chain():
    """Sets up a conversational chain using Groq LLM."""
    try:
        prompt_template = """
        You are a specialized assistant that ONLY answers questions based on the provided context.
        If the question cannot be answered using the information in the context, respond with:
        "I cannot answer this question as it's not covered in the provided documents."

        DO NOT use any external knowledge or make assumptions beyond what's in the context.
        If you're unsure about any part of the answer, err on the side of saying the information is not available.

        Keep the font syle of the response text uniform, do not use any variation in font syle

        Context:
        {context}

        Question:
        {question}

        Answer (based STRICTLY on the above context):
        """

        model = ChatGroq(
            temperature=0.1,
            model_name="deepseek-r1-distill-llama-70b",
            groq_api_key=groq_api_key
        )
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error initializing Groq model: {str(e)}")
        return None

def user_input(user_question):
    """Handles user queries by retrieving answers from the vector store."""
    try:
        embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        # Increase k for better context matching
        docs = new_db.similarity_search(user_question, k=4)

        # Check if the similarity score is too low
        if not docs:
            st.markdown("### Reply:\nI cannot answer this question as it's not covered in the provided documents.")
            return

        chain = get_conversational_chain()
        if chain is None:
            return

        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )

        response_text = re.sub(r'<think>.*?</think>', '', response['output_text'], flags=re.DOTALL).strip()
        st.markdown(f"### Reply:\n{response_text}")
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Document-Grounded Chat", page_icon=":books:", layout="wide")
    st.title("Document-Grounded Chat Assistant")

    st.sidebar.header("Document Upload")
    st.sidebar.markdown(
        "Upload documents to ground the chatbot's knowledge."
    )

    with st.sidebar:
        pdf_docs = st.file_uploader(
            "Upload PDF documents:",
            accept_multiple_files=True,
            type=["pdf"]
        )
        if st.button("Process Documents"):
            if not pdf_docs:
                st.error("Please upload at least one PDF document.")
                return

            with st.spinner("Processing documents..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                if get_vector_store(text_chunks):
                    st.success("Documents processed successfully!")

    st.markdown(
        "### Ask Questions\n"
        "The assistant will only answer questions based on the uploaded documents."
    )

    user_question = st.text_input("Your question:", placeholder="Ask about the uploaded documents...")

    if user_question:
        if not os.path.exists("faiss_index"):
            st.error("Please upload and process documents first.")
            return

        with st.spinner("Finding answer..."):
            user_input(user_question)

    st.sidebar.info(
        """
        **Note:** This assistant is strictly grounded to the uploaded documents.
        It will not provide information from outside sources.
        """
    )

if __name__ == "__main__":
    main()