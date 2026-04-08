# =============================================================================
# crew/tools.py — CUSTOM CREWAI TOOLS (Agent Superpowers)
# =============================================================================
# This file defines custom tools that CrewAI agents can call during their work.
# Tools are Python functions wrapped with the @tool decorator so that the LLM
# knows they exist, when to use them, and how to format its input for them.
#
# Currently there is ONE custom tool here:
#   CreateDraftTool.create_draft
#       Lets Agent 3 (Email Response Writer) actually save a draft email
#       to your Gmail Drafts folder.
#
# HOW THE @tool DECORATOR WORKS:
#   When you decorate a function with @tool("Tool Name"), LangChain exposes it
#   to the LLM as a callable action. The docstring inside the function is what
#   the LLM reads to understand WHEN and HOW to use the tool.
#   The LLM will format its tool call exactly as described in the docstring.
#
# INPUT FORMAT DESIGN CHOICE:
#   The tool accepts a single pipe-separated string like:
#       "user@gmail.com|Re: Meeting|Sure, I'm free at 3pm!"
#   This is simpler than a dict for LLMs to produce reliably.
#   The agent is instructed (in tasks.py) to pass exactly this format.
# =============================================================================

from langchain_community.agent_toolkits import GmailToolkit
from langchain_community.tools.gmail.create_draft import GmailCreateDraft
from langchain.tools import tool


class CreateDraftTool():

    @tool("Create Draft")
    def create_draft(data):
        """
        Useful to create an email draft in Gmail.

        The input to this tool should be a pipe (|) separated text
        of length 3 (three), representing who to send the email to,
        the subject of the email and the actual message.

        For example, `lorem@ipsum.com|Nice To Meet You|Hey it was great to meet you.`

        The agent MUST format the input exactly like that example:
            recipient_email | subject_line | email_body
        Do NOT include extra pipes or line breaks in the input.
        """
        # Split the pipe-separated input into its three parts
        email, subject, message = data.split('|')

        # Initialize Gmail toolkit and get the draft creation tool
        gmail = GmailToolkit()
        draft = GmailCreateDraft(api_resource=gmail.api_resource)

        # Call the Gmail API to create the draft
        result = draft({
            'to': [email],        # Recipient (must be a list)
            'subject': subject,   # Email subject line
            'message': message    # Email body text
        })

        # Return a confirmation string that the agent will see
        return f"\nDraft created: {result}\n"
