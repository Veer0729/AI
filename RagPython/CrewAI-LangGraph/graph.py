# =============================================================================
# graph.py — THE LANGGRAPH WORKFLOW (The Master Blueprint)
# =============================================================================
# This is the most important file in the project.
# It defines the full flow of the system — what happens first, what happens
# next, and when to loop back.
#
# Think of it as drawing a flowchart:
#   - Boxes = Nodes (steps)
#   - Arrows = Edges (connections between steps)
#   - Diamond = Conditional Edge (a decision point)
#
# The flow looks like this:
#
#   START
#     ↓
#   [check_new_emails]
#     ↓ (decision: any new emails?)
#     ├── NO  → [wait_next_run] ──────────────────┐
#     └── YES → [draft_responses] (CrewAI!) → [wait_next_run]
#                                                  ↓
#                                     loops back to [check_new_emails]
#
# This is an INFINITE LOOP — the system runs forever until you kill it.
# =============================================================================

from dotenv import load_dotenv
load_dotenv()  # Load API keys and config from .env file at startup

from langgraph.graph import StateGraph

from .state import EmailsState         # The shared state schema (TypedDict)
from .nodes import Nodes               # The non-AI node functions
from .crew.crew import EmailFilterCrew # The CrewAI crew (handles AI work)


class WorkFlow():
    def __init__(self):
        # Instantiate node functions (check_email, wait_next_run, new_emails)
        nodes = Nodes()

        # Create a new StateGraph, telling it what shape our shared state has.
        # Every node will receive and return an EmailsState dict.
        workflow = StateGraph(EmailsState)

        # ---------------------------------------------------------------
        # REGISTER NODES
        # Each node needs a unique string name and a callable function.
        # ---------------------------------------------------------------

        # Node 1: Fetch new emails from Gmail (pure Python)
        workflow.add_node("check_new_emails", nodes.check_email)

        # Node 2: Sleep for 3 minutes (pure Python)
        workflow.add_node("wait_next_run", nodes.wait_next_run)

        # Node 3: Run the full CrewAI crew to filter + draft email responses.
        # THIS IS THE BRIDGE — EmailFilterCrew().kickoff is a plain function
        # that takes state, runs CrewAI internally, and returns updated state.
        # LangGraph just treats it like any other node function.
        workflow.add_node("draft_responses", EmailFilterCrew().kickoff)

        # ---------------------------------------------------------------
        # SET ENTRY POINT
        # Tell LangGraph which node to start at when app.invoke() is called.
        # ---------------------------------------------------------------
        workflow.set_entry_point("check_new_emails")

        # ---------------------------------------------------------------
        # ADD CONDITIONAL EDGE (the decision diamond)
        # After "check_new_emails" runs, LangGraph calls nodes.new_emails(state).
        # That function returns either "continue" or "end".
        # The dict below maps those return values to actual node names.
        # ---------------------------------------------------------------
        workflow.add_conditional_edges(
            "check_new_emails",    # FROM this node
            nodes.new_emails,      # CALL this function to decide the next step
            {
                "continue": 'draft_responses',  # "continue" → run CrewAI
                "end":      'wait_next_run'      # "end"      → skip CrewAI, just wait
            }
        )

        # ---------------------------------------------------------------
        # ADD REGULAR EDGES (straight arrows, no decision)
        # ---------------------------------------------------------------

        # After CrewAI drafts responses → wait before next cycle
        workflow.add_edge('draft_responses', 'wait_next_run')

        # After waiting → go back to checking emails (this creates the infinite loop)
        workflow.add_edge('wait_next_run', 'check_new_emails')

        # ---------------------------------------------------------------
        # COMPILE
        # Locks in the graph definition and returns a runnable application object.
        # After compile(), you can call self.app.invoke() or self.app.stream().
        # ---------------------------------------------------------------
        self.app = workflow.compile()