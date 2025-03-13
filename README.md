# AI-99 RAG
A Retrieval Augmented Generation application built with Streamlit.

## Demo
The application is deployed and accessible at: https://ai-99-rag.streamlit.app

## Overview
AI-99 RAG implements Retrieval Augmented Generation technology, which enhances large language model responses by retrieving relevant information from a knowledge base before generating answers. This approach improves accuracy and contextual relevance in AI-generated responses.

## Features
- Knowledge retrieval from document sources
- Context-aware AI responses
- Interactive question answering
- User-friendly web interface
- Support for PDF and DOCX document formats

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/roydon/ai-99-rag
cd ai-99-rag
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory and add your API keys:
```
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
```

## Usage
1. Start the application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Upload your documents (PDF/DOCX) and start asking questions

## Deployment
The application is deployed on Streamlit Cloud. For your own deployment:

1. Fork this repository
2. Connect your GitHub account to Streamlit Cloud
3. Deploy the app through Streamlit Cloud dashboard
4. Configure the necessary environment variables in Streamlit Cloud settings

## Tech Stack
- Streamlit: Web interface
- LangChain: RAG implementation
- FAISS: Vector storage
- Sentence Transformers: Document embeddings
- OpenAI/Groq: Language models

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
