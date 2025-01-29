import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Get API keys from Streamlit secrets
groq_api_key = st.secrets["GROQ_API_KEY"]
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Validate API keys
if not groq_api_key or not openai_api_key:
    st.error("Please set up your API keys in Streamlit secrets.")
    st.stop()

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Creates and saves a FAISS vector store from text chunks."""
    try:
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def get_conversational_chain():
    """Sets up a conversational chain using Groq LLM."""
    try:
        prompt_template = """
        Answer the question as detailed as possible from the provided context. If the answer is not in
        the provided context, just say, "answer is not available in the context." Do not provide incorrect answers.

        Context:
        {context}?

        Question:
        {question}

        Answer:
        """

        model = ChatGroq(
            temperature=0.3,
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
        embeddings = OpenAIEmbeddings()
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()
        if chain is None:
            return

        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )

        # st.markdown(f"### Reply:\n{response['output_text']}")
        st.markdown(f"### Reply:\n{response}")
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Chat PDF", page_icon=":books:", layout="wide")
    st.title("Chat with PDF using Groq (Deepseek R1)")

    st.sidebar.header("Upload & Process PDF Files")
    st.sidebar.markdown(
        "Using Groq's Deepseek R1 Distill 70B model for advanced conversational capabilities.")

    with st.sidebar:
        pdf_docs = st.file_uploader(
            "Upload your PDF files:",
            accept_multiple_files=True,
            type=["pdf"]
        )
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
                return

            with st.spinner("Processing your files..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                if get_vector_store(text_chunks):
                    st.success("PDFs processed and indexed successfully!")

    st.markdown(
        "### Ask Questions from Your PDF Files :mag:\n"
        "Once you upload and process your PDFs, type your questions below."
    )

    user_question = st.text_input("Enter your question:", placeholder="What do you want to know?")

    if user_question:
        with st.spinner("Fetching your answer..."):
            user_input(user_question)

    st.sidebar.info(
        """
        **Note:** This app uses Groq's Deepseek R1 Distill 70B model for processing questions and generating answers.
        Make sure your PDFs are processed before asking questions.
        """
    )

if __name__ == "__main__":
    main()