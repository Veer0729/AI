# =============================================================================
# nodes.py — LANGGRAPH NODE FUNCTIONS (Pure Python, No AI)
# =============================================================================
# This file contains all the NON-AI steps in the LangGraph workflow.
# These are plain Python functions — no LLM, no CrewAI — just logic.
#
# Each function here represents one "node" (step) in the graph.
# Every node function must:
#   - Accept `state` (the shared EmailsState dict) as input
#   - Return an updated version of `state` as output
#
# The three functions defined here are:
#   1. check_email()    → Fetch new emails from Gmail
#   2. wait_next_run()  → Sleep for 3 minutes before next cycle
#   3. new_emails()     → Decision function: are there new emails? yes/no
#                         (This is NOT a node — it's a conditional router)
# =============================================================================

import os
import time

from langchain_community.agent_toolkits import GmailToolkit
from langchain_community.tools.gmail.search import GmailSearch


class Nodes():
    def __init__(self):
        # Initialize Gmail toolkit — handles OAuth2 authentication with Gmail API.
        # On first run, this will open a browser window to ask you to log in.
        # After that, it caches the credentials locally.
        self.gmail = GmailToolkit()

    def check_email(self, state):
        """
        NODE 1: Check for new emails in Gmail.

        What it does:
        - Searches Gmail for emails received in the last 24 hours
        - Filters out emails we've already seen (checked_emails_ids)
        - Filters out duplicate threads (same conversation counted once)
        - Filters out emails sent BY YOU (no point replying to yourself)
        - Returns the fresh list of new emails + updated seen-IDs list

        Input state fields used:  checked_emails_ids
        Output state fields set:  emails, checked_emails_ids
        """
        print("# Checking for new emails")

        # Create a Gmail search tool using the authenticated API resource
        search = GmailSearch(api_resource=self.gmail.api_resource)

        # Search for emails from the last 1 day
        # 'after:newer_than:1d' is Gmail's search syntax
        emails = search('after:newer_than:1d')

        # Get the list of email IDs we've already processed (or empty list if first run)
        checked_emails = state['checked_emails_ids'] if state['checked_emails_ids'] else []

        thread = []       # Track thread IDs to avoid processing the same thread twice
        new_emails = []   # The final list of genuinely new emails to process

        for email in emails:
            # Only include the email if ALL three conditions are met:
            # 1. We haven't seen this email ID before
            # 2. We haven't seen this thread before (avoid duplicate threads)
            # 3. The email was NOT sent by us (filter out sent emails)
            if (email['id'] not in checked_emails) \
                    and (email['threadId'] not in thread) \
                    and (os.environ['MY_EMAIL'] not in email['sender']):

                # Mark this thread as seen so we don't pick it up twice
                thread.append(email['threadId'])

                # Add just the fields we need to the new_emails list
                new_emails.append(
                    {
                        "id": email['id'],
                        "threadId": email['threadId'],
                        "snippet": email['snippet'],   # Short preview of the email
                        "sender": email["sender"]
                    }
                )

        # Mark ALL emails from this search as "seen" (even ones we skipped)
        # so they don't show up as candidates in the next cycle
        checked_emails.extend([email['id'] for email in emails])

        # Return updated state — spread existing state (**state) and override these two fields
        return {
            **state,
            "emails": new_emails,
            "checked_emails_ids": checked_emails
        }

    def wait_next_run(self, state):
        """
        NODE 3: Wait for 3 minutes before starting the next check cycle.

        This node is hit in two scenarios:
        1. No new emails were found → wait, then check again
        2. Emails were processed by CrewAI → wait, then check again

        It simply sleeps for 180 seconds and returns the state unchanged.

        Input state fields used:  (none — just passes state through)
        Output state fields set:  (none — state returned as-is)
        """
        print("## Waiting for 180 seconds")
        time.sleep(180)   # Pause for 3 minutes
        return state      # Return state unchanged — nothing new to add here

    def new_emails(self, state):
        """
        ROUTER FUNCTION: Decide which path the graph should take next.

        This is NOT a node — it's a conditional edge function.
        LangGraph calls this after 'check_new_emails' to decide where to go next.

        Returns:
            "end"       → No new emails → go to wait_next_run (skip CrewAI)
            "continue"  → New emails found → go to draft_responses (run CrewAI)

        These string values map to actual node names in graph.py's
        add_conditional_edges() call.
        """
        if len(state['emails']) == 0:
            print("## No new emails")
            return "end"       # Route to: wait_next_run
        else:
            print("## New emails")
            return "continue"  # Route to: draft_responses (CrewAI kickoff)
