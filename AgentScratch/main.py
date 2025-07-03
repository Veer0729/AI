from dotenv import load_dotenv
import os # for seeing and getting variables from another file
from pydantic import BaseModel # for defining our data structure
from langchain_core.prompts import ChatPromptTemplate # telling our model it's role
from langchain_core.output_parsers import PydanticOutputParser # gives structured output
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_groq import ChatGroq  # Groq wrapper compatible with LangChain
from tools import search_tool, wiki_tool, save_tool
load_dotenv() # my env file

api_subscription_key = os.getenv("GROQ_API_KEY") # gets my key


# My whole setup, api key called, model number and temp controls the randomness or creativity of the model's output.
llm = ChatGroq(
    api_key = api_subscription_key,
    model="llama-3.3-70b-versatile",
    temperature=0.7,
)

# all of the fields I want as output
class ResarchResponse(BaseModel):
    topics: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# langchain will go through this parser, then the research class to see hoe to give it's output
parser = PydanticOutputParser(pydantic_object=ResarchResponse)

# telling my AI it's job
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool , save_tool]
agent = create_tool_calling_agent(
    llm = llm,
    prompt= prompt,
    tools= tools
)

agent_executor = AgentExecutor(agent=agent, tools= tools, verbose= True)
query = input("What can i help you with?")
result = agent_executor.invoke({"query": query}) # my chats
print(result)