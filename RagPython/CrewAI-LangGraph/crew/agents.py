# =============================================================================
# crew/agents.py — CREWAI AGENTS (The AI Specialists)
# =============================================================================
# This file defines the three AI agents that make up the CrewAI crew.
# Each agent is like a specialized employee with:
#   - role      → their job title
#   - goal      → what they are trying to achieve
#   - backstory → personality/expertise prompt that shapes their behavior
#   - tools     → Python functions the LLM can call to take real actions
#
# The three agents and their responsibilities:
#
#   Agent 1 — Email Filter Agent (email_filter_agent)
#       Reads email snippets and decides which ones are worth responding to.
#       Has NO tools — uses pure LLM reasoning to spot newsletters vs real emails.
#
#   Agent 2 — Email Action Agent (email_action_agent)
#       Pulls the full email thread from Gmail.
#       Understands context, urgency, and what action is needed.
#       Can also search the web for context using Tavily.
#
#   Agent 3 — Email Response Writer (email_response_writer)
#       Drafts actual email replies and saves them to Gmail Drafts.
#       Can re-read threads, do web research, and create drafts via tool.
#
# NOTE: allow_delegation=False means agents cannot hand off tasks to each other.
#       The task chaining is handled by CrewAI at the Crew level (crew.py).
# =============================================================================

from langchain_community.agent_toolkits import GmailToolkit
from langchain_community.tools.gmail.get_thread import GmailGetThread
from langchain_community.tools.tavily_search import TavilySearchResults

from textwrap import dedent
from crewai import Agent
from .tools import CreateDraftTool   # Our custom Gmail draft-creation tool


class EmailFilterAgents():
    def __init__(self):
        # Initialize Gmail toolkit for agents that need to read threads.
        # Shares the same authenticated Gmail session.
        self.gmail = GmailToolkit()

    def email_filter_agent(self):
        """
        AGENT 1: Senior Email Analyst — The Gatekeeper

        Job: Look at the list of email snippets and decide which ones
        are genuinely important (not spam, not newsletters, not notifications).

        Tools: NONE — this agent only needs its LLM brain to decide.
        Input: Email snippets formatted as a string in the task description.
        Output: A bullet-point list of thread IDs that passed the filter.
        """
        return Agent(
            role='Senior Email Analyst',
            goal='Filter out non-essential emails like newsletters and promotional content',
            backstory=dedent("""\
                As a Senior Email Analyst, you have extensive experience in email content analysis.
                You are adept at distinguishing important emails from spam, newsletters, and other
                irrelevant content. Your expertise lies in identifying key patterns and markers that
                signify the importance of an email."""),
            verbose=True,           # Print agent's thought process to console
            allow_delegation=False  # This agent works alone, no handoffs
        )

    def email_action_agent(self):
        """
        AGENT 2: Email Action Specialist — The Deep Reader

        Job: Take the thread IDs that passed Agent 1's filter.
        Pull the FULL email thread from Gmail (not just the snippet).
        Understand the full context and identify exactly what response is needed.

        Tools:
          - GmailGetThread: Reads the complete email conversation by thread ID
          - TavilySearchResults: Searches the web if context/research is needed

        Output: For each thread → thread_id, summary, key points, sender email,
                communication style, who to reply to.
        """
        return Agent(
            role='Email Action Specialist',
            goal='Identify action-required emails and compile a list of their IDs',
            backstory=dedent("""\
                With a keen eye for detail and a knack for understanding context, you specialize
                in identifying emails that require immediate action. Your skill set includes interpreting
                the urgency and importance of an email based on its content and context."""),
            tools=[
                GmailGetThread(api_resource=self.gmail.api_resource),  # Read full thread
                TavilySearchResults()                                   # Web search
            ],
            verbose=True,
            allow_delegation=False
        )

    def email_response_writer(self):
        """
        AGENT 3: Email Response Writer — The Drafter

        Job: Using the analysis from Agent 2, write actual email replies.
        Each reply should match the user's communication style.
        Save each draft to Gmail using the CreateDraftTool.

        Tools:
          - TavilySearchResults: Research facts/topics if needed before writing
          - GmailGetThread: Re-read the thread if more context is needed
          - CreateDraftTool.create_draft: Actually saves the draft to Gmail Drafts

        Output: Confirmation that all draft replies were created in Gmail.
        """
        return Agent(
            role='Email Response Writer',
            goal='Draft responses to action-required emails',
            backstory=dedent("""\
                You are a skilled writer, adept at crafting clear, concise, and effective email responses.
                Your strength lies in your ability to communicate effectively, ensuring that each response is
                tailored to address the specific needs and context of the email."""),
            tools=[
                TavilySearchResults(),                                  # Web research
                GmailGetThread(api_resource=self.gmail.api_resource),  # Re-read thread
                CreateDraftTool.create_draft                           # Save draft to Gmail
            ],
            verbose=True,
            allow_delegation=False
        )