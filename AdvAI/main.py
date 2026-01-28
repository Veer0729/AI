from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
from langchain_groq import ChatGroq
from llama_index.llms.langchain import LangChainLLM
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PDFReader
from prompts import instruction_str, new_prompt, context

# Load environment variables
load_dotenv()

# --- Define get_index BEFORE calling it ---
def get_index(llm):
    pdf_path = os.path.join("data", "India.pdf")
    india_pdf = PDFReader().load_data(file=pdf_path)
    
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_name = "india"

    if not os.path.exists(index_name):
        print("building index", index_name)
        index = VectorStoreIndex.from_documents(india_pdf, embed_model=embed_model, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        print("loading index", index_name)
        storage_context = StorageContext.from_defaults(persist_dir=index_name)
        index = load_index_from_storage(storage_context, embed_model=embed_model)

    return index.as_query_engine(llm=llm)

# --- LLM setup ---
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=1
)
llama_llm = LangChainLLM(llm=llm)

# --- PDF vector index engine ---
india_engine = get_index(llama_llm)

# --- CSV population engine ---
population_path = os.path.join("data", "population.csv")
population_df = pd.read_csv(population_path)

population_query_engine = PandasQueryEngine(
    df=population_df,
    llm=llama_llm,
    verbose=True,
    instruction_str=instruction_str
)
population_query_engine.update_prompts({"pandas_prompt": new_prompt})
population_query_engine.query("what is the population of India?")

# --- Tools ---
tools = [
    note_engine,
    QueryEngineTool(
        query_engine=population_query_engine,
        metadata=ToolMetadata(
            name="population_data",
            description="This gives information of the world population and demographics",
        )
    ),
    QueryEngineTool(
        query_engine=india_engine,
        metadata=ToolMetadata(
            name="india_data",
            description="This gives information about India the country",
        )
    ),
]

# --- Agent ---
agent = ReActAgent.from_tools(tools, llm=llama_llm, verbose=True, context=context)

# --- Prompt loop ---
while True:
    prompt = input("Enter a prompt: ")
    if prompt.lower() == "q":
        break
    result = agent.query(prompt)
    print(result)
