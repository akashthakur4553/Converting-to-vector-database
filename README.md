# Bosswallah RAG Chatbot

This repository contains a Retrieval-Augmented Generation (RAG) chatbot that can answer questions about Bosswallah courses using data stored in a vector database. The chatbot uses Google's Gemini AI for generating responses and supports multilingual queries.

## Features

- **Vector Database Integration**: Uses ChromaDB to store and retrieve document embeddings
- **Multilingual Support**: Automatically translates non-English queries using Gemini AI
- **Semantic Search**: Finds relevant documents based on meaning, not just keywords
- **Source Attribution**: Shows the source documents used to generate responses
- **Streamlit UI**: User-friendly interface for interacting with the chatbot



### RAG (Retrieval-Augmented Generation) Approach

The chatbot uses the following RAG workflow:

1. **Query Processing**: User queries are translated to English if needed
2. **Embedding Generation**: Queries are converted to vector embeddings
3. **Semantic Search**: The system finds similar documents in the vector database
4. **Context Preparation**: Retrieved documents are formatted as context
5. **Response Generation**: Gemini AI generates a response based on the context and query
6. **Language Matching**: Responses are provided in the same language as the query

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key

### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd Converting\ to\ vector\ database
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Gemini API key:
   - Get an API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create or edit the `.env` file in the project root directory:
     ```
     GEMINI_API_KEY=your_api_key_here
     ```



### Running the Chatbot

Start the Streamlit app:
```
streamlit run app.py
```

The chatbot will automatically:
1. Connect to the ChromaDB database
2. Load the first available collection
3. Present a chat interface for asking questions

## Usage

1. Type your question in the chat input field
2. The system will retrieve relevant documents and generate a response
3. Click on "📚 Source Documents" to see the sources used for the response
4. Use the sidebar to adjust the number of documents to retrieve
5. Clear chat history using the "🗑️ Clear Chat History" button

## Customization

- **Change the embedding model**: Modify the `SentenceTransformer` model in the code
- **Adjust search parameters**: Change the number of results retrieved in the sidebar
- **Modify the prompt**: Edit the prompt template in the `generate_response` method

## Troubleshooting

- If you see an API key error, make sure you've set your Gemini API key correctly
- If the database path doesn't exist, ensure you've run the data conversion script
- If translation isn't working, check your internet connection and API key validity

