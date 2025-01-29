import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import os
import re

# Initialize session state
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# Initialize configuration defaults
if 'config' not in st.session_state:
    st.session_state.config = {
        'k_value': 4,
        'chunk_size': 1000,
        'temperature': 0.1,
        'chunk_overlap': 200
    }

# Get API keys from Streamlit secrets
groq_api_key = st.secrets["GROQ_API_KEY"]

def get_pdf_text(pdf_docs):
    """Extracts text from uploaded PDF files."""
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def get_text_chunks(text):
    """Splits extracted text into manageable chunks."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=st.session_state.config['chunk_size'],
            chunk_overlap=st.session_state.config['chunk_overlap'],
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error splitting text: {str(e)}")
        return None

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

        Context:
        {context}

        Question:
        {question}

        Answer (based STRICTLY on the above context):
        """

        model = ChatGroq(
            temperature=st.session_state.config['temperature'],
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
        
        docs = new_db.similarity_search(user_question, k=st.session_state.config['k_value'])
        
        if not docs:
            st.markdown("### Reply:\nI cannot answer this question as it's not covered in the provided documents.")
            return

        # Show source chunks in expander
        with st.expander("View source chunks used", expanded=False):
            st.markdown(f"**Using {st.session_state.config['k_value']} most relevant chunks:**")
            for i, doc in enumerate(docs, 1):
                st.markdown(f"**Chunk {i}:**\n{doc.page_content}\n---")

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

def reset_app():
    """Resets the application state."""
    if os.path.exists("faiss_index"):
        try:
            import shutil
            shutil.rmtree("faiss_index")
        except Exception as e:
            st.error(f"Error clearing index: {str(e)}")
    st.session_state.processing_complete = False
    st.session_state.config = {
        'k_value': 4,
        'chunk_size': 1000,
        'temperature': 0.1,
        'chunk_overlap': 200
    }
    st.rerun()

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="Document-Grounded Chat",
        page_icon=":books:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Document-Grounded Chat Assistant")

    # Sidebar
    with st.sidebar:
        # Configuration Section
        st.header("Configuration")
        
        # Model Parameters
        st.subheader("Model Parameters")
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.config['temperature'],
            step=0.1,
            help="Controls randomness in the response. Lower values make responses more focused."
        )

        # Chunk Parameters
        st.subheader("Chunk Parameters")
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.number_input(
                "Chunk Size",
                min_value=100,
                max_value=2000,
                value=st.session_state.config['chunk_size'],
                step=100,
                help="Size of text chunks during processing"
            )
        
        with col2:
            chunk_overlap = st.number_input(
                "Chunk Overlap",
                min_value=0,
                max_value=500,
                value=st.session_state.config['chunk_overlap'],
                step=50,
                help="Overlap between chunks"
            )

        # Search Parameters
        st.subheader("Search Parameters")
        k_value = st.slider(
            "Number of chunks (k)",
            min_value=1,
            max_value=10,
            value=st.session_state.config['k_value'],
            help="Number of relevant chunks to use for answering"
        )

        # Update configuration
        st.session_state.config.update({
            'temperature': temperature,
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'k_value': k_value
        })

        # Reset configuration button
        if st.button("Reset Configuration"):
            reset_app()

        st.markdown("---")

        # Document Upload Section
        st.header("Document Upload")
        pdf_docs = st.file_uploader(
            "Upload PDF documents:",
            accept_multiple_files=True,
            type=["pdf"]
        )
        
        if not pdf_docs:
            st.info("Please upload PDF documents to begin.")
            st.button("Process Documents", disabled=True)
        else:
            st.success(f"{len(pdf_docs)} document(s) uploaded!")
            
            # Show document names
            st.markdown("### Uploaded Documents:")
            for doc in pdf_docs:
                st.markdown(f"- {doc.name}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Process Documents", type="primary", key="process"):
                    with st.spinner("Processing documents..."):
                        raw_text = get_pdf_text(pdf_docs)
                        if raw_text:
                            text_chunks = get_text_chunks(raw_text)
                            if text_chunks and get_vector_store(text_chunks):
                                st.session_state.processing_complete = True
                                st.success("Documents processed successfully!")
                                st.rerun()
            
            with col2:
                if st.button("Clear All", type="secondary", key="clear"):
                    reset_app()

        # System Status
        st.markdown("---")
        st.markdown("### System Status")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("Documents:")
            if pdf_docs:
                st.success("Uploaded")
            else:
                st.error("Not uploaded")
        
        with col2:
            st.markdown("Processing:")
            if st.session_state.processing_complete:
                st.success("Complete")
            else:
                st.warning("Pending")

    # Main chat interface
    st.markdown("### Ask Questions")
    st.markdown("The assistant will only answer questions based on the uploaded documents.")

    # Question input
    if not st.session_state.processing_complete:
        st.text_input(
            "Your question:",
            placeholder="Process documents first...",
            disabled=True
        )
        st.info("Please upload and process documents to start asking questions.")
    else:
        user_question = st.text_input(
            "Your question:",
            placeholder="Ask about the uploaded documents..."
        )
        if user_question:
            with st.spinner("Finding answer..."):
                user_input(user_question)

    # Information and guidelines
    with st.sidebar:
        st.markdown("---")
        st.info(
            """
            **Guidelines:**
            1. Upload PDF documents
            2. Adjust configuration if needed
            3. Click 'Process Documents'
            4. Ask questions about the documents
            5. Use 'Clear All' to reset
            
            **Note:** This assistant only answers questions based on the uploaded documents.
            """
        )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p style='color: #666; font-size: 0.8em;'>
                Document-Grounded Chat Assistant | Powered by Groq & BGE Embeddings
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()