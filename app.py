import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

GROUNDING_AVAILABLE = False
try:
    from google.genai import types as genai_types, Client as GenaiClient
    GROUNDING_AVAILABLE = True
except ImportError:
    genai_types = None
    GenaiClient = None

from typing import List, Dict, Optional
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class RAGChatbot:
    def __init__(self, db_path: str = "./chroma_db", gemini_api_key: Optional[str] = None):
        self.client = chromadb.PersistentClient(path=db_path)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        api_key = gemini_api_key or GEMINI_API_KEY
        if api_key and api_key != "YOUR_API_KEY_HERE":
            genai.configure(api_key=api_key)
            self.llm = genai.GenerativeModel('gemini-2.0-flash')
        else:
            self.llm = None
        
        self.grounding_client = None
        self.grounding_available = GROUNDING_AVAILABLE
        if GROUNDING_AVAILABLE and api_key:
            try:
                self.grounding_client = GenaiClient(api_key=api_key)
            except Exception as e:
                st.warning(f"Could not initialize grounding client: {e}")
                self.grounding_available = False
        
        self.collection = None
        self.available_collections = []
        self._load_collections()
    
    def _load_collections(self):
        try:
            self.available_collections = [col.name for col in self.client.list_collections()]
        except Exception as e:
            st.error(f"Error loading collections: {e}")
            self.available_collections = []
    
    def set_collection(self, collection_name: str):
        try:
            self.collection = self.client.get_collection(name=collection_name)
            return True
        except Exception as e:
            st.error(f"Error setting collection: {e}")
            return False
    
    def search_similar_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        if not self.collection:
            return []
        
        try:
            search_query = query
            try:
                if self.llm:
                    translation_prompt = f"""

                    Text to translate: {query}

                    """
                    translation_response = self.llm.generate_content(translation_prompt)
                    search_query = translation_response.text.strip() if translation_response.text else query
                    print(f"Original query: {query}")
                    print(f"Translated query for RAG search: {search_query}")
                else:
                    print(f"Gemini not available for translation, using original query: {query}")
            except Exception as e:
                print(f"Translation failed, using original query: {str(e)}")
                pass
            
            query_embedding = self.model.encode([search_query], convert_to_tensor=False)
            
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results['documents'] or not results['documents'][0]:
                return []
            
            similar_docs = []
            for i in range(len(results['documents'][0])):
                similar_docs.append({
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                    'distance': results['distances'][0][i] if results['distances'] and results['distances'][0] else 0.0
                })
            
            return similar_docs
        
        except Exception as e:
            st.error(f"Error searching documents: {e}")
            return []
    
    def generate_response_with_grounding(self, query: str) -> str:
        if not self.llm:
            return "Gemini API key not configured properly. Please set your API key in the code to query the dataset."
        
        if self.grounding_available and self.grounding_client and genai_types:
            try:
                grounding_tool = genai_types.Tool(
                    google_search=genai_types.GoogleSearch()
                )
                
                config = genai_types.GenerateContentConfig(
                    tools=[grounding_tool]
                )
                
                response = self.grounding_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=query,
                    config=config,
                )
                return response.text or ""
            except Exception as e:
                st.warning(f"Grounding failed, falling back to standard generation: {e}")
        
        try:
            prompt = f"""

            QUESTION (respond in the same language): {query}

            YOUR RESPONSE (in the same language as the question):
            """
            
            response = self.llm.generate_content(prompt)
            return response.text or ""
        except Exception as e:
            return f"Error generating grounded response: {e}"
    
    def is_external_query_needed(self, query: str, context_docs: List[Dict]) -> bool:
        if not context_docs:
            return True
            
        external_indicators = [
            "current", "today", "now", "latest", "stores near", 
            "weather", "news", "price", "availability", "location", "where can i buy",
            "open now", "working hours", "contact", "phone number", "address",
            "near me", "closest", "nearest"
        ]
        
        query_lower = query.lower()
        for indicator in external_indicators:
            if indicator in query_lower:
                return True
                
        if context_docs and len(context_docs) > 0:
            avg_distance = sum(doc['distance'] for doc in context_docs) / len(context_docs)
            if avg_distance > 0.7:
                return True
                
        return False
    
    def generate_response(self, query: str, context_docs: List[Dict]) -> str:
        if self.is_external_query_needed(query, context_docs):
            st.info("🔄 Query requires external information. Searching the web...")
            return self.generate_response_with_grounding(query)
            
        if not self.llm:
            return "Gemini API key not configured properly. Please set your API key in the code to query the dataset."
        
        context = ""
        for i, doc in enumerate(context_docs):
            context += f"Document {i+1}:\n{doc['document']}\n\n"
        
        prompt = f"""

        CONTEXT DOCUMENTS:
        {context}

        USER QUESTION (respond in the same language): {query}

        YOUR RESPONSE (in the same language as the question):
        """
        
        try:
            response = self.llm.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {e}"
    
    def chat(self, query: str, n_results: int = 5) -> tuple[str, List[Dict]]:
        similar_docs = self.search_similar_documents(query, n_results)
        
        response = self.generate_response(query, similar_docs)
        
        return response, similar_docs

def main():
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Boss Bot")
    st.markdown("Ask questions about Bosswallah courses")
    
    db_path = "./chroma_db"
    if 'chatbot' not in st.session_state:
        if GEMINI_API_KEY != "YOUR_API_KEY_HERE" and os.path.exists(db_path):
            st.session_state.chatbot = RAGChatbot(db_path, GEMINI_API_KEY)
            if st.session_state.chatbot.available_collections:
                first_collection = st.session_state.chatbot.available_collections[0]
                if st.session_state.chatbot.set_collection(first_collection):
                    st.session_state.collection_loaded = True
                    st.session_state.active_collection = first_collection
        else:
            if GEMINI_API_KEY == "YOUR_API_KEY_HERE":
                st.error("❌ Please set your API key in the code (GEMINI_API_KEY variable)")
                st.stop()
            if not os.path.exists(db_path):
                st.error(f"❌ Database path does not exist: {db_path}")
                st.stop()
    
    with st.sidebar:
        st.header("Settings")
        
        if hasattr(st.session_state, 'active_collection'):
            st.info(f"📁 Active Collection: {st.session_state.active_collection}")
        
        n_results = st.slider("Number of documents to retrieve", 1, 10, 5)
        
        st.subheader("Chat History")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask a question about your data..."):
        if 'chatbot' not in st.session_state:
            st.error("Please configure the API key and database path first")
            return
        
        if not hasattr(st.session_state, 'collection_loaded'):
            st.error("Please select and load a collection first")
            return
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, sources = st.session_state.chatbot.chat(prompt, n_results)
            
            st.markdown(response)
            
            if sources:
                with st.expander("📚 Source Documents"):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**Document {i+1}** (Distance: {doc['distance']:.3f})")
                        st.text(doc['document'][:300] + "..." if len(doc['document']) > 300 else doc['document'])
                        st.markdown("---")
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()