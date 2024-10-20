# Local GenAI Search

Local GenAI Search is an AI-powered document search and question-answering system that works with your local files. It uses advanced natural language processing techniques to understand and answer questions based on the content of your documents.

## Features

- Index and search through various document types (PDF, DOCX, PPTX, TXT)
- Extract and process images from documents
- Use AI to generate answers to questions based on document content
- User-friendly Streamlit interface
- Local processing for data privacy

## Requirements

- Python 3.7+
- Ollama (for running local AI models)

## Installation

1. Clone this repository:   ```
   git clone https://github.com/imanoop7/Generative-Search-Engine-For-Local-Files-v2
   cd Generative-Search-Engine-For-Local-Files-v2   ```

2. Create a virtual environment and activate it:   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`   ```

3. Install the required packages:   ```
   pip install -r requirements.txt   ```

4. Install Ollama by following the instructions at [https://ollama.ai/](https://ollama.ai/)

5. Pull the required models:   ```
   ollama pull tinyllama
   ollama pull llava   ```

## Usage

1. Start the Streamlit app:   ```
   streamlit run local_genai_search.py   ```

2. Open your web browser and go to the URL displayed in the terminal (usually `http://localhost:8501`).

3. Enter the path to your documents folder in the text input field.

4. Click the "Index Documents" button to process and index your documents. This may take some time depending on the number and size of your documents.

5. Once indexing is complete, you can start asking questions about your documents in the "Ask a Question" section.

6. Click "Search and Answer" to get AI-generated answers based on your document content.

## How it Works

1. Document Indexing: The system reads and processes various document types, extracting text and images. It then creates embeddings for the text content using a pre-trained sentence transformer model.

2. Semantic Search: When you ask a question, the system converts it into an embedding and searches for the most similar content in your indexed documents.

3. Answer Generation: The system uses the Ollama API to generate an answer based on the question and the most relevant document content found during the search.

4. Image Processing: For documents containing images, the system uses the LLaVA model to generate descriptions, which are then included in the search index.

## Troubleshooting

If you encounter any issues:

1. Make sure all required packages are installed correctly.
2. Ensure Ollama is running and the required models (tinyllama and llava) are downloaded.
3. Check the console output for any error messages.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Streamlit for the amazing web app framework
- Sentence Transformers for text embeddings
- Ollama for local AI model inference
- FAISS for efficient similarity search
