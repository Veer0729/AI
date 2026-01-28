import os
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PDFReader

pdf_path = os.path.join("data", "India.pdf")
india_pdf = PDFReader().load_data(file = pdf_path)

def get_index(llm):
    embed_model = HuggingFaceEmbedding(model_name= "sentence-transformers/all-MiniLM-L6-v2")
    index_name = "india"
    if not os.path.exists(index_name):
        print("building index", index_name)
        index = VectorStoreIndex.from_documents(india_pdf, embed_model = embed_model, show_progress=True)
        index.storage_context.persist(persist_dir= index_name)

    else:
        print("loading index", index_name)
        storage_context = StorageContext.from_defaults(persist_dir=index_name)
        index = load_index_from_storage(storage_context, embed_model=embed_model)

    return index.as_query_engine(llm=llm)

india_engine = get_index()