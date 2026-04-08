from crewai import LLM
from crewai.tools import tool
from crewai import Agent, Task, Crew
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
import os
from langchain_ollama import ChatOllama  # add this back
from crewai_tools import ScrapeWebsiteTool

scrape_tool = ScrapeWebsiteTool()

# CrewAI LLM - for agents only
llm = LLM(model="ollama/qwen2.5:3b", base_url="http://localhost:11434")

# LangChain LLM - for rag_chain only
langchain_llm = ChatOllama(model="qwen2.5:3b", base_url="http://localhost:11434")

# ---- BUILD RAG CHAIN ----
file_path = "./17.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below. Be concise.
Context: {context}
Question: {question}
""")

rag_chain = (
    {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
     "question": RunnablePassthrough()}
    | prompt
    | langchain_llm
    | StrOutputParser()
)

# ---- TOOL 1: PDF RAG TOOL ----
@tool("PDF Search Tool")
def rag_tool(question: str) -> str:
    """Search the PDF document for answers about ESOP. Input must be a question string."""
    return rag_chain.invoke(question)

# ---- TOOL 2: WEB SEARCH TOOL ----
duckduckgo = DuckDuckGoSearchRun()

@tool("Web Search Tool")
def web_search_tool(query: str) -> str:
    """Search the web for information not found in the PDF. Input must be a search query string."""
    return duckduckgo.run(query)

# ---- ROUTER TOOL ----
@tool("Router Tool")
def router_tool(question: str) -> str:
    """Routes the question to vectorstore or web search based on content."""
    if 'ESOP' in question.upper():
        return 'vectorstore'
    else:
        return 'websearch'

# ---- AGENTS ----
Router_Agent = Agent(
    role='Router',
    goal='Route user question to a vectorstore or web search',
    backstory=(
        "You are an expert at routing a user question to a vectorstore or web search."
        "Use the vectorstore for questions related to ESOP."
        "Otherwise, use web search."
    ),
    verbose=True,
    allow_delegation=False,
    tools=[router_tool],
    llm=llm,
    max_iter=2,
)

Retriever_Agent = Agent(
    role="Retriever",
    goal="Use the correct tool to retrieve information based on router output",
    backstory=(
        "You are an assistant for question-answering tasks."
        "Use rag_tool if the router said 'vectorstore'."
        "Use web_search_tool to find relevant URLs, then use scrape_tool to read the full page content."
        "Provide a clear concise answer."
    ),
    verbose=True,
    allow_delegation=False,
    tools=[rag_tool, web_search_tool, scrape_tool],  # ✅ both tools here
    llm=llm,
    max_iter=2,
)

Grader_agent = Agent(
    role='Answer Grader',
    goal='Filter out erroneous retrievals',
    backstory=(
        "You are a grader assessing relevance of a retrieved document to a user question."
        "If the document contains keywords related to the user question, grade it as relevant."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

hallucination_grader = Agent(
    role="Hallucination Grader",
    goal="Filter out hallucination",
    backstory=(
        "You are a hallucination grader assessing whether an answer is grounded in facts."
        "Meticulously review the answer and check if the response is in alignment with the question."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

answer_grader = Agent(
    role="Answer Grader",
    goal="Provide a final useful answer, falling back to web search if needed.",
    backstory=(
        "You are a grader assessing whether an answer is useful to resolve a question."
        "If the answer is relevant, return it clearly and concisely."
        "If not relevant, use web_search_tool to find URLs, then scrape_tool to read full content."
    ),
    verbose=True,
    allow_delegation=False,
    tools=[web_search_tool, scrape_tool],  # ✅ fallback web search
    llm=llm,
)

# ---- TASKS ----
router_task = Task(
    description=(
        "Analyse the keywords in the question {question}. "
        "Return 'vectorstore' if it is about ESOP. "
        "Return 'websearch' otherwise. "
        "Do not provide any preamble or explanation."
    ),
    expected_output="A single word: 'websearch' or 'vectorstore'.",
    agent=Router_Agent,
    tools=[router_tool],
)

retriever_task = Task(
    description=(
        "Based on the router task output, answer the question {question}. "
        "Use rag_tool if router said 'vectorstore'. "
        "Use web_search_tool if router said 'websearch'."
    ),
    expected_output="A clear and concise answer to the question.",
    agent=Retriever_Agent,
    context=[router_task],
    tools=[rag_tool, web_search_tool],
)

grader_task = Task(
    description=(
        "Evaluate whether the retrieved content is relevant to the question {question}."
    ),
    expected_output="'yes' or 'no' only. No preamble.",
    agent=Grader_agent,
    context=[retriever_task],
)

hallucination_task = Task(
    description=(
        "Evaluate whether the answer is grounded in facts for the question {question}."
    ),
    expected_output="'yes' or 'no' only. No preamble.",
    agent=hallucination_grader,
    context=[grader_task],
)

answer_task = Task(
    description=(
        "Based on the hallucination task result for {question}: "
        "If 'yes', return the answer clearly. "
        "If 'no', use web_search_tool to find a better answer."
    ),
    expected_output=(
        "A clear and concise final answer. "
        "If nothing found, respond: 'Sorry! unable to find a valid response'."
    ),
    context=[hallucination_task],
    agent=answer_grader,
    tools=[web_search_tool],
)

# ---- CREW ----
rag_crew = Crew(
    agents=[Router_Agent, Retriever_Agent, Grader_agent, hallucination_grader, answer_grader],
    tasks=[router_task, retriever_task, grader_task, hallucination_task, answer_task],
    verbose=True,
)

inputs = {"question": "what is machine learning?"}
result = rag_crew.kickoff(inputs=inputs)
print(result)