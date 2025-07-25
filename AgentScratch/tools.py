from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, SerpAPIWrapper
from langchain.tools import Tool
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

# for making and saving a file
def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file.",
)

# to search the net 
serpapi_key = os.getenv("SERPAPI_API_KEY")
search = SerpAPIWrapper(serpapi_api_key=serpapi_key)
search_tool = Tool(
    name = "search",
    func= search.run,
    description= "Search the web for information",
)

# to search the wiki
api_wrapper  = WikipediaAPIWrapper(top_k_results= 1, doc_content_chars_max= 100)
wiki_tool = WikipediaQueryRun(api_wrapper = api_wrapper)