# =============================================================================
# main.py — ENTRY POINT
# =============================================================================
# This is the file you run to start the entire system.
# It does three simple things:
#   1. Imports the WorkFlow class from graph.py
#   2. Creates the compiled LangGraph application
#   3. Starts the infinite loop with an empty initial state
#
# Think of this as the "ON button" for the whole project.
# Once you run this, the system will:
#   - Check Gmail → Filter emails → Draft replies → Wait → Repeat forever
# =============================================================================

from src.graph import WorkFlow

# Build the LangGraph graph and compile it into a runnable app
app = WorkFlow().app

# Start the graph with an empty state {}
# LangGraph will fill in the state as it moves through nodes
app.invoke({})