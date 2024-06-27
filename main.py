import os
from dotenv import load_dotenv
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.storage import StorageContext
from llama_index.core.indices.loading import load_index_from_storage
import google.generativeai as genai
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings

# Load environment variables
load_dotenv()


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
llm = genai.configure(api_key=GOOGLE_API_KEY)

# Set up HuggingFace Embedding
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Update global settings
Settings.llm = llm
Settings.embed_model = embed_model

# Define persistent storage directory
PERSIST_DIR = "./storage"

def load_or_create_index():
    if not os.path.exists(PERSIST_DIR):
        documents = SimpleDirectoryReader("data").load_data()
        st.write(f"Number of documents loaded: {len(documents)}")
        if len(documents) == 0:
            st.error("No documents found in the 'data' directory.")
            return None
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    return index

def main():
    st.title("RAG Project with Gemini API")

    st.write(f"API Key set: {'GOOGLE_API_KEY' in os.environ}")

    # Load or create index
    index = load_or_create_index()
    if index is None:
        st.error("Failed to create or load index. Please check your data directory.")
        return

    st.write(f"Index loaded. Number of nodes: {len(index.docstore.docs)}")

    # Set up query engine
    query_engine = index.as_query_engine()

    # Streamlit interface
    user_query = st.text_input("Enter your query:")
    if st.button("Submit"):
        if user_query:
            st.write(f"Query: {user_query}")
            response = query_engine.query(user_query)

            # Access response text
            response_text = response.response

            # Display answer
            st.write("Answer:", response_text)

            summary_prompt = f"This is a RAG system that retrieves documents and uses them to inform the generation of a response to your query '{user_query}'. Summarize this application's functionalities in 3-4 sentences."
            try:
                summary_response = genai.generate_text(model="models/gemini-1.5-flash-latest", prompt=summary_prompt,
                                                       temperature=0.2)
                st.write("Summary:")
                st.write(summary_response.result)
            except Exception as e:
                st.error(f"Failed to generate summary: {e}")
            else:
                st.warning("Please enter a query.")


if __name__ == "__main__":
    main()