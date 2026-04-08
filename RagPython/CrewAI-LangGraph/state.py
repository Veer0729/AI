# =============================================================================
# state.py — SHARED STATE (The Common Notepad)
# =============================================================================
# This file defines the "state" — a shared dictionary that every node in
# the LangGraph graph reads from and writes to.
#
# Think of it like a Google Doc that every worker (node) can open and update.
# When one node finishes, it writes its results here.
# The next node picks up from exactly where the previous one left off.
#
# LangGraph requires you to define the shape of this state using TypedDict,
# so it knows exactly what fields to expect and pass around.
# =============================================================================

import datetime
from typing import TypedDict


class EmailsState(TypedDict):
    # List of email IDs we have already seen/processed in previous cycles.
    # This prevents us from re-processing the same email on the next loop.
    checked_emails_ids: list[str]

    # The fresh batch of NEW emails found in the current check cycle.
    # Each email is a dict with keys: id, threadId, snippet, sender.
    # This gets reset every loop with only the new emails.
    emails: list[dict]

    # The final output from the CrewAI crew after drafting responses.
    # Contains the result/confirmation from the Email Response Writer agent.
    action_required_emails: dict