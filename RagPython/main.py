import os                                 # Provides functions for interacting with the operating system (finding folders, files, etc.)
import streamlit as st                    # Streamlit is used to make the web UI/app interface
from dotenv import load_dotenv            # Loads environment variables from a .env file for API keys/settings

# Importing LlamaIndex modules for retrieval-augmented generation
from llama_index.llms.openai import OpenAI                     # OpenAI LLM for answering questions
from llama_index.embeddings.openai import OpenAIEmbedding      # OpenAI embedding model for vector encoding
from llama_index.readers.wikipedia import WikipediaReader      # Reads and fetches Wikipedia articles
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage  
# VectorStoreIndex: stores texts as embeddings
# StorageContext: manages saving/loading vector storage
# load_index_from_storage: loads a previously saved index

load_dotenv()                          # Loads environment variables from .env (e.g., OpenAI API key)

# Directory to store/load the vector index
INDEX_DIR = "wiki_rag"
# List of Wikipedia pages to use as knowledge base
PAGES = [
    "Goldfish",
    "Tuna",
    "Shark",
    "Clownfish",
    "Swordfish",
    "Eel",
    "Dolphin",
    "Whale",
    "Seal (animal)",
    "Sea otter",
    "Manatee",
    "Walrus"
]


@st.cache_resource                      # Streamlit decorator: cache the result (index object) for faster reloads
def get_index():
    # If the vector index directory exists, load the index from disk
    if os.path.isdir(INDEX_DIR):
        storage = StorageContext.from_defaults(persist_dir = INDEX_DIR)      # Get index storage context from folder
        return load_index_from_storage(storage)                              # Load and return precomputed index

    # If no saved index, download Wikipedia articles and build a new index
    docs = WikipediaReader().load_data(pages = PAGES, auto_suggest=False)    # Fetch articles from Wikipedia
    embedding_model = OpenAIEmbedding(model="text-embedding-3-small")         # Initialize embedding model
    index = VectorStoreIndex.from_documents(docs, embedding_model=embedding_model)  # Create embeddings and make index
    index.storage_context.persist(persist_dir=INDEX_DIR)                      # Save index to disk for future use
    return index                                                             # Return the new index


@st.cache_resource                      # Caches the query engine so it's not rebuilt every time
def get_query_engine():
    index = get_index()                 # Gets (or loads) the vector index

    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)   # Set up the OpenAI LLM for answering questions

    # Combine vector index and LLM: enables semantic search over docs, then LLM "answers" using them
    return index.as_query_engine(llm=llm, similarity_top_k=1) 


def main():
    st.title('Wikipedia RAG Application')              # Title for your Streamlit webpage

    question = st.text_input('Ask a question')         # Text box for user to enter a question

    if st.button('Submit') and question:               # If button pressed and question entered:
        with st.spinner('Thinking...'):                # Show "thinking..." spinner while working
            qa = get_query_engine()                    # Get the combined LLM+vector query engine
            response = qa.query(question)              # Get answer to question using RAG pipeline

        st.subheader('Answer')                         # Section for showing the answer
        st.write(response.response)                    # Display the answer

        st.subheader('Retrieved context')              # Section for showing the supporting context
        for src in response.source_nodes:              # For every returned source "Node" (text)
            st.markdown(src.node.get_content())        # Display the content as markdown

if __name__ == '__main__':
    main()                                             # If this file is run directly, launch the Streamlit app
