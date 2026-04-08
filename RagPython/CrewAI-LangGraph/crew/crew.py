# =============================================================================
# crew/crew.py — THE CREWAI CREW (Team Assembly + LangGraph Bridge)
# =============================================================================
# This file is the most important piece of the CrewAI side.
# It does two things:
#
#   1. ASSEMBLES THE TEAM
#      Combines the three agents (from agents.py) and three tasks (from tasks.py)
#      into a single Crew object that CrewAI can execute.
#
#   2. ACTS AS THE BRIDGE BETWEEN LANGGRAPH AND CREWAI
#      The kickoff() method is what LangGraph calls as a node.
#      It takes the LangGraph state as input, extracts the emails,
#      runs the entire crew, and returns the updated state.
#
# The kickoff() function signature is the key design decision here:
#   def kickoff(self, state: EmailsState) → EmailsState
#
# LangGraph doesn't know or care it's calling CrewAI — it just sees
# a regular function that takes state and returns state.
# =============================================================================

from crewai import Crew

from .agents import EmailFilterAgents
from .tasks import EmailFilterTasks


class EmailFilterCrew():
    def __init__(self):
        # Create all three agents upfront when the crew is initialized.
        # This happens once when graph.py sets up the workflow.
        agents = EmailFilterAgents()
        self.filter_agent = agents.email_filter_agent()    # Agent 1: Gatekeeper
        self.action_agent = agents.email_action_agent()   # Agent 2: Deep Reader
        self.writer_agent = agents.email_response_writer() # Agent 3: Drafter

    def kickoff(self, state):
        """
        THE BRIDGE FUNCTION — called by LangGraph as the 'draft_responses' node.

        This is where LangGraph hands control over to CrewAI.

        What happens here:
        1. Format the emails from state into a readable string for Agent 1
        2. Create fresh Task objects for this cycle (tasks need the current emails)
        3. Assemble the Crew with agents + tasks
        4. Run the crew (agents execute tasks sequentially: filter → analyse → draft)
        5. Return the updated state with the crew's result added

        Args:
            state: The LangGraph EmailsState dict — must have 'emails' populated
                   by the check_new_emails node before this is called.

        Returns:
            Updated state dict with 'action_required_emails' set to crew result.
        """
        print("### Filtering emails")

        # Create task instances fresh for each run
        # (tasks are created here because filter_emails_task needs the current emails)
        tasks = EmailFilterTasks()

        crew = Crew(
            agents=[
                self.filter_agent,   # 1st to run: filters junk
                self.action_agent,   # 2nd: deep-reads filtered threads
                self.writer_agent    # 3rd: writes & saves draft replies
            ],
            tasks=[
                # Task 1: Pass in formatted email string so Agent 1 can read them
                tasks.filter_emails_task(
                    self.filter_agent,
                    self._format_emails(state['emails'])  # Converts list → readable string
                ),
                # Task 2: No extra input — CrewAI passes Task 1's output automatically
                tasks.action_required_emails_task(self.action_agent),
                # Task 3: No extra input — CrewAI passes Task 2's output automatically
                tasks.draft_responses_task(self.writer_agent)
            ],
            verbose=True  # Print each agent's reasoning steps to console
        )

        # Run all agents and tasks in order
        result = crew.kickoff()

        # Return updated state — spread existing state and add the crew's result
        # LangGraph will pick this up and continue the graph with updated state
        return {**state, "action_required_emails": result}

    def _format_emails(self, emails):
        """
        HELPER: Convert the list of email dicts into a single readable string.

        The LLM in Agent 1 gets this as part of its prompt, so it needs to be
        human-readable text, not a raw Python list.

        Example output:
            ID: 18f2abc...
            - Thread ID: 18f2abc...
            - Snippet: Hey, just wanted to follow up on...
            - From: john@example.com
            --------
            ID: 19a3def...
            ...
        """
        emails_string = []
        for email in emails:
            print(email)  # Log each email to console for debugging
            arr = [
                f"ID: {email['id']}",
                f"- Thread ID: {email['threadId']}",
                f"- Snippet: {email['snippet']}",
                f"- From: {email['sender']}",
                f"--------"
            ]
            emails_string.append("\n".join(arr))
        return "\n".join(emails_string)