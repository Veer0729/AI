import os
import streamlit as st
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI # Going to answer our questions
from llama_index.embeddings.openai import OpenAIEmbedding # takes our queries and wikepedia articles and embed them into vector space
from llama_index.readers.wikipedia import WikipediaReader # for the text of the articles
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage # Out vector storage 

load_dotenv() # loads our env file from .env

INDEX_DIR = "wiki_rag"
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

@st.cache_resource # stores the result for fiture uses
def get_index(): # gets the info in need if already saved
    if os.path.isdir(INDEX_DIR):
        storage = StorageContext.from_defaults(persist_dir = INDEX_DIR)
        return load_index_from_storage(storage)
    
    # If not saved....gets the info from wikiepdia
    docs = WikipediaReader().load_data(pages = PAGES, auto_suggest= False)
    embedding_model = OpenAIEmbedding(model="text-embedding-3-small")
    index = VectorStoreIndex.from_documents(docs, embedding_model = embedding_model)
    index.storage_context.persist(persist_dir= INDEX_DIR)
    return index

@st.cache_resource
def get_query_engine():
    index = get_index()

    llm = OpenAI(model = "gpt-3.5-turbo", temperature=0)

    return index.as_query_engine(llm = llm, similarity_top_k = 1)

def main():
    st.title('Wikipedia RAG Application')

    question = st.text_input('Ask a question')

    if st.button('Submit') and question:
        with st.spinner('Thinking...'):
            qa = get_query_engine()
            response = qa.query(question)

        st.subheader('Answer')
        st.write(response.response)

        st.subheader('Retrieved context')
        for src in response.source_nodes:
            st.markdown(src.node.get_content())

if __name__ == '__main__':
    main()