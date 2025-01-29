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
import time
from datetime import datetime

# Disable file watcher to prevent inotify issues
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

# Model Configurations
AVAILABLE_MODELS = {
    "Mixtral-8x7b": {
        "name": "mixtral-8x7b-32768",
        "context_length": 32768,
        "description": "Mixtral 8x7B model with extended context",
        "temperature_range": (0.0, 1.0),
        "default_temperature": 0.3
    },
    "Llama-3.3-70b": {
        "name": "llama-3.3-70b-versatile",
        "context_length": 4096,
        "description": "Llama 3.3 70B model",
        "temperature_range": (0.0, 1.0),
        "default_temperature": 0.3
    },
    "DeepSeek-R1-70b": {
        "name": "deepseek-r1-distill-llama-70b",
        "context_length": 4096,
        "description": "DeepSeek 70B distilled model",
        "temperature_range": (0.0, 1.0),
        "default_temperature": 0.3
    }
}

# Initialize session states
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

if 'config' not in st.session_state:
    st.session_state.config = {
        'k_value': 4,
        'chunk_size': 1000,
        'temperature': 0.1,
        'chunk_overlap': 200,
        'selected_model': "DeepSeek-R1-70b"
    }

if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {
        model: {
            'uses': 0,
            'avg_response_time': 0,
            'last_used': None
        } for model in AVAILABLE_MODELS
    }

def validate_selected_model():
    """Ensure selected model exists in AVAILABLE_MODELS."""
    if st.session_state.config['selected_model'] not in AVAILABLE_MODELS:
        st.session_state.config['selected_model'] = list(AVAILABLE_MODELS.keys())[0]

def process_model_response(model_name, response_text):
    """Process model response based on model name."""
    if 'deepseek' in model_name.lower():
        return re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
    return response_text

def update_model_metrics(model_name, response_time):
    """Update usage metrics for the selected model."""
    metrics = st.session_state.model_metrics[model_name]
    metrics['uses'] += 1
    metrics['avg_response_time'] = (
        (metrics['avg_response_time'] * (metrics['uses'] - 1) + response_time) / metrics['uses']
    )
    metrics['last_used'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
    """Sets up a conversational chain using selected Groq model."""
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

        selected_model = st.session_state.config['selected_model']
        model_config = AVAILABLE_MODELS[selected_model]

        model = ChatGroq(
            temperature=st.session_state.config['temperature'],
            model_name=model_config['name'],
            groq_api_key=st.secrets["GROQ_API_KEY"]
        )
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error initializing Groq model: {str(e)}")
        return None

def compare_models(question, docs):
    """Compare responses from different models."""
    results = {}

    with st.expander("Model Comparison Results", expanded=True):
        st.markdown("### Model Comparison")
        st.markdown("Comparing responses from different models...")

        for model_name, model_config in AVAILABLE_MODELS.items():
            try:
                with st.spinner(f"Getting response from {model_name}..."):
                    start_time = time.time()

                    model = ChatGroq(
                        temperature=st.session_state.config['temperature'],
                        model_name=model_config['name'],
                        groq_api_key=st.secrets["GROQ_API_KEY"]
                    )

                    prompt_template = """
                    You are a specialized assistant that ONLY answers questions based on the provided context.
                    If the question cannot be answered using the information in the context, respond with:
                    "I cannot answer this question as it's not covered in the provided documents."

                    Context:
                    {context}

                    Question:
                    {question}

                    Answer (based STRICTLY on the above context):
                    """

                    chain = load_qa_chain(
                        model,
                        chain_type="stuff",
                        prompt=PromptTemplate(
                            template=prompt_template,
                            input_variables=["context", "question"]
                        )
                    )

                    response = chain(
                        {"input_documents": docs, "question": question},
                        return_only_outputs=True
                    )

                    end_time = time.time()
                    response_time = end_time - start_time

                    # Process response text based on model name
                    response_text = process_model_response(model_name, response['output_text'])

                    results[model_name] = {
                        'response': response_text,
                        'time': response_time
                    }

                    update_model_metrics(model_name, response_time)

                    st.markdown(f"**{model_name}**")
                    st.markdown(f"Response time: {response_time:.2f}s")
                    st.markdown(response_text)
                    st.markdown("---")
            except Exception as e:
                st.error(f"Error with {model_name}: {str(e)}")

    return results

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

        # Check if comparison is requested
        if getattr(st.session_state, 'show_comparison', False):
            compare_models(user_question, docs)
            st.session_state.show_comparison = False
        else:
            # Regular single model response
            start_time = time.time()
            chain = get_conversational_chain()
            if chain is None:
                return

            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )

            end_time = time.time()
            response_time = end_time - start_time

            # Process response text based on model name
            response_text = process_model_response(
                st.session_state.config['selected_model'],
                response['output_text']
            )

            update_model_metrics(st.session_state.config['selected_model'], response_time)

            st.markdown(f"### Reply:\n{response_text}")
            st.markdown(f"*Response time: {response_time:.2f}s*")

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
        'chunk_overlap': 200,
        'selected_model': "DeepSeek-R1-70b"
    }
    st.rerun()

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="Multi-Model Document Chat",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Multi-Model Document-Grounded Chat Assistant")

    # Sidebar
    with st.sidebar:
        # Validate selected model before creating the selectbox
        validate_selected_model()

        # Model Selection
        st.header("Model Selection")
        selected_model = st.selectbox(
            "Choose Model",
            options=list(AVAILABLE_MODELS.keys()),
            index=list(AVAILABLE_MODELS.keys()).index(st.session_state.config['selected_model']),
            help="Select the Groq model to use for responses"
        )

        # Model Information
        with st.expander("Model Information", expanded=False):
            model_info = AVAILABLE_MODELS[selected_model]
            st.markdown(f"""
            **Model:** {selected_model}
            - **Description:** {model_info['description']}
            - **Context Length:** {model_info['context_length']} tokens
            - **Temperature Range:** {model_info['temperature_range'][0]} to {model_info['temperature_range'][1]}
            """)

        # Model Parameters
        st.subheader("Model Parameters")
        temperature = st.slider(
            "Temperature",
            min_value=AVAILABLE_MODELS[selected_model]['temperature_range'][0],
            max_value=AVAILABLE_MODELS[selected_model]['temperature_range'][1],
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

        # Model Comparison
        st.subheader("Model Comparison")
        if st.button("Compare Models"):
            st.session_state.show_comparison = True

        # Update configuration
        st.session_state.config.update({
            'temperature': temperature,
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'k_value': k_value,
            'selected_model': selected_model
        })

        # Model Metrics
        if st.checkbox("Show Model Metrics"):
            st.markdown("### Model Usage Metrics")
            for model, metrics in st.session_state.model_metrics.items():
                st.markdown(f"""
                **{model}**
                - Uses: {metrics['uses']}
                - Avg Response Time: {metrics['avg_response_time']:.2f}s
                - Last Used: {metrics['last_used'] or 'Never'}
                """)

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

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p style='color: #666; font-size: 0.8em;'>
                Multi-Model Document-Grounded Chat Assistant | Powered by Groq
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()