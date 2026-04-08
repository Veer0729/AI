# =============================================================================
# crew/tasks.py — CREWAI TASKS (Job Descriptions for Each Agent)
# =============================================================================
# This file defines the three Tasks — one per agent.
# A Task is essentially a prompt/instruction card that tells an agent:
#   - WHAT to do (the description)
#   - WHAT to produce (the expected output format, written in the description)
#   - WHO does it (the agent parameter)
#
# CrewAI runs tasks sequentially by default.
# Task 2 automatically receives Task 1's output as context.
# Task 3 automatically receives Task 2's output as context.
# This is CrewAI's built-in "task chaining" — no extra code needed.
#
# The three tasks mirror the three agents:
#   Task 1 → filter_emails_task     → given to email_filter_agent
#   Task 2 → action_required_emails_task → given to email_action_agent
#   Task 3 → draft_responses_task   → given to email_response_writer
# =============================================================================

from crewai import Task
from textwrap import dedent


class EmailFilterTasks:

    def filter_emails_task(self, agent, emails):
        """
        TASK 1: Filter emails for Agent 1 (Senior Email Analyst)

        Instructions:
        - Look at the provided email snippets
        - Remove newsletters, promotions, notifications, and anything not
          directly addressed to the user
        - Pay attention to the sender email address

        Input:
            agent  → the email_filter_agent instance
            emails → pre-formatted string of all emails (from _format_emails in crew.py)

        Expected output:
            Bullet-point list of thread_ids and senders that passed the filter.
        """
        return Task(
            description=dedent(f"""\
                Analyze a batch of emails and filter out
                non-essential ones such as newsletters, promotional content and notifications.

                Use your expertise in email content analysis to distinguish
                important emails from the rest, pay attention to the sender and avoid invalid emails.

                Make sure to filter for the messages actually directed at the user and avoid notifications.

                EMAILS
                -------
                {emails}

                Your final answer MUST be a the relevant thread_ids and the sender, use bullet points.
                """),
            agent=agent   # This task is assigned to email_filter_agent
        )

    def action_required_emails_task(self, agent):
        """
        TASK 2: Deep-read threads for Agent 2 (Email Action Specialist)

        Instructions:
        - Take thread IDs from Task 1's output (passed automatically by CrewAI)
        - Pull each full email thread using GmailGetThread tool
        - Analyse context, tone, urgency
        - Identify what reply is needed

        Input:
            agent → the email_action_agent instance
            (emails are passed implicitly from Task 1's output by CrewAI)

        Expected output for each thread:
            - thread_id
            - Summary of the conversation
            - Key points / main concerns
            - Who the user is replying to
            - Communication style (formal/casual/etc.)
            - Sender's actual email address
        """
        return Task(
            description=dedent("""\
                For each email thread, pull and analyze the complete threads using only the actual Thread ID.
                Understand the context, key points, and the overall sentiment
                of the conversation.

                Identify the main query or concerns that needs to be
                addressed in the response for each.

                Your final answer MUST be a list for all emails with:
                - the thread_id
                - a summary of the email thread
                - a highlighting with the main points
                - identify the user and who he will be answering to
                - communication style in the thread
                - the sender's email address
                """),
            agent=agent   # This task is assigned to email_action_agent
        )

    def draft_responses_task(self, agent):
        """
        TASK 3: Write and save draft replies for Agent 3 (Email Response Writer)

        Instructions:
        - Use the analysis from Task 2 (passed automatically by CrewAI)
        - Write a reply for each email that needs action
        - Match the user's communication style from the thread
        - Research if needed (using Tavily) BEFORE writing
        - Use CreateDraftTool to save each draft to Gmail Drafts
        - Must confirm all drafts were saved before finishing

        Input:
            agent → the email_response_writer instance
            (context is passed implicitly from Task 2's output by CrewAI)

        Expected output:
            Text confirmation that all email drafts have been created in Gmail.
        """
        return Task(
            description=dedent(f"""\
                Based on the action-required emails identified, draft responses for each.
                Ensure that each response is tailored to address the specific needs
                and context outlined in the email.

                - Assume the persona of the user and mimic the communication style in the thread.
                - Feel free to do research on the topic to provide a more detailed response, IF NECESSARY.
                - IF a research is necessary do it BEFORE drafting the response.
                - If you need to pull the thread again do it using only the actual Thread ID.

                Use the tool provided to draft each of the responses.
                When using the tool pass the following input:
                - to (sender to be responded)
                - subject
                - message

                You MUST create all drafts before sending your final answer.
                Your final answer MUST be a confirmation that all responses have been drafted.
                """),
            agent=agent   # This task is assigned to email_response_writer
        )