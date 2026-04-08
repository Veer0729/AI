import os
from crewai import LLM, Agent, Crew, Process, Task
from crewai_tools import FirecrawlCrawlWebsiteTool


llm = LLM(
    model="ollama/hermes3:3b",
    base_url="http://localhost:11434"
)

tools = [FirecrawlCrawlWebsiteTool(os.environ.get("FIRECRAWL_API_KEY"))]

blog_scrapper = Agent(
    name = "Website Scraper",
    role = "Web Content Researcher",
    goal = "Extract Complete and accurate information from given URL",
    backstory = "You are an expert web content researcher. Your task is to extract complete inforamtion of the given website",
    llm = llm,
    tools = tools,
    verbose = True,
    allow_delegation = False
)

blog_summarizer = Agent(
    name = "Website summarizer",
    role = "blog summarizer",
    goal = "Create a short and informative summary of the blog content",
    backstory = "You are an expert content analyst. Your task is to creat short, crisp and informative summary",
    llm = llm,
    verbose = True,
    allow_delegation = False
)

# Task
def scrape_blog_task(url):
    return Task(
        description=(
            f"Scrape content from the blog at {url} using FirecrawlScrapeWebsiteTool"
            "Extract the main content, including text, and any other relevant information, filtering out ads and images."
        ),
        expected_output="Full text content of the blog post in markdown format",
        agent=blog_scrapper,
    )


def summarize_blog_task(scrape_task):
    return Task(
        description=(
            "Summarize the content extracted by the Blog Scraper agent. "
            "Create a concise, informative summary that captures the main points and key insights of the blog"
        ),
        expected_output=(
            "Concise summary of the blog post in 100-200 words."
            "The summary will be used to generate a podcast script."
            "Summary should be audio-friendly, engaging, and suitable for a podcast audience."
        ),
        agent=blog_summarizer,
        context=[scrape_task]
    )

def create_blog_summary(url):
    scrape_task= scrape_blog_task(url)
    summarize= summarize_blog_task(scrape_task)

    crew= Crew(
        agents= [blog_scrapper, blog_summarizer],
        tasks= [scrape_task, summarize],
        verbose= True,
        process= Process.sequential,
    )

    return crew

def summarized_blog(url):
    crew= create_blog_summary(url)
    result= crew.kickoff()

    return result.raw

if __name__== "__main__":
    url = ""
    summary = summarized_blog(url)
    print("blog summary: ", summary)