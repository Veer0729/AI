# 📚 CrewAI + LangGraph — Complete Notes (Beginner Friendly)

> Written for someone who wants to understand this project from ZERO, and then build something similar on their own.

---

## 🧠 Part 1: The Big Picture — What Problem Does This Project Solve?

Imagine you get 50 emails a day.
- Some are newsletters → **ignore**
- Some are promotional spam → **ignore**
- Some need a reply → **open, read, think, draft a reply**

You want an **AI system** that:
1. Keeps checking Gmail every 3 minutes
2. Filters out junk emails automatically
3. Reads the important ones
4. Researches context if needed
5. **Drafts a reply in your Gmail drafts** — automatically

This is exactly what this project does. It is an **autonomous AI email assistant**.

---

## 🤖 Part 2: The Two Frameworks — What Are They?

Before understanding how they connect, let's understand what each one does **alone**.

---

### 🔷 LangGraph — The "Brain" / Orchestrator

**What is it?**
LangGraph is a library for building **stateful, looping workflows** using a concept called a **graph** (like a flowchart).

**Think of it like this:**
> LangGraph is the **manager** of the whole operation. It decides WHAT happens next.

**Key concepts:**
| Concept | What it means |
|---|---|
| **Node** | A single step / action (e.g., "check email", "wait", "process emails") |
| **Edge** | A connection from one node to the next |
| **Conditional Edge** | A decision point — "if X goto A, else goto B" |
| **State** | A shared memory that is passed between every step |
| **Graph** | The full flowchart combining all nodes and edges |

**The graph in this project looks like this:**

```
START
  │
  ▼
[check_new_emails]  ←───────────────────--──┐
  │                                         │
  ├── No new emails  →  [wait_next_run] ────┘
  │
  └── New emails found  →  [draft_responses]  →  [wait_next_run] ──┘
```

It's an **infinite loop**. The system never stops — it keeps polling Gmail forever.

---

### 🟠 CrewAI — The "Workers" / AI Agents

**What is it?**
CrewAI is a framework for creating **multiple AI agents** that each have a **role**, a **goal**, and **tools**, and they work **together on tasks** like a real team.

**Think of it like this:**
> CrewAI is the **team of specialists** that LangGraph calls when there's real work to do.

**Key concepts:**
| Concept | What it means |
|---|---|
| **Agent** | An AI "person" with a role, goal, and backstory |
| **Task** | A specific job assigned to an agent |
| **Tool** | A real Python function the agent can call (e.g., search web, read email thread) |
| **Crew** | The assembled team of agents + their tasks |
| **kickoff()** | The method that starts the crew and runs all tasks in order |

---

## 🔗 Part 3: WHY Do We Need BOTH Together?

This is the most important question. Here's the honest answer:

| LangGraph Alone | CrewAI Alone |
|---|---|
| Great at controlling FLOW (loops, conditions, routing) | Great at multi-agent collaboration on complex tasks |
| Does NOT have built-in multi-agent teamwork | Does NOT have built-in looping / flow control |
| Can't easily say "agent A passes result to agent B then C" | Can't easily say "keep doing this forever, with a condition" |

**When you combine them:**
- LangGraph handles the **outer loop** — when to check, when to wait, when to trigger AI work
- CrewAI handles the **inner work** — actually reading, analyzing, and writing emails with a team of agents

👉 **LangGraph is the skeleton. CrewAI is the muscle.**

---

## 📁 Part 4: File-by-File Breakdown

### 📄 `state.py` — The Shared Memory

```python
class EmailsState(TypedDict):
    checked_emails_ids: list[str]  # IDs of emails we've already seen
    emails: list[dict]             # New emails we found this cycle
    action_required_emails: dict   # Final output from CrewAI
```

**What this is:**
This is the **shared memory/notepad** that every node in LangGraph reads from and writes to.

Think of it like a Google Doc that all workers share. Every step can read what the previous step wrote.

- `checked_emails_ids` — keeps track of emails already processed so we don't re-process them on the next loop
- `emails` — the fresh batch of emails found in the current check cycle
- `action_required_emails` — the final CrewAI output (drafted responses)

---

### 📄 `nodes.py` — The LangGraph "Steps" (Non-AI work)

This file contains the **pure Python logic** that LangGraph runs as nodes. No AI here — just code.

#### `check_email(state)` — Node 1

```python
def check_email(self, state):
    search = GmailSearch(api_resource=self.gmail.api_resource)
    emails = search('after:newer_than:1d')  # Get emails from last 1 day
    ...
    # Filter out: already seen emails, same thread duplicates, emails YOU sent
    return {**state, "emails": new_emails, "checked_emails_ids": checked_emails}
```

**What it does:**
1. Searches Gmail for emails from the last 24 hours
2. Filters out emails you already processed (`checked_emails_ids`)
3. Filters out duplicate threads (don't process same conversation twice)
4. Filters out emails YOU sent (no point replying to yourself)
5. Returns updated state with fresh `emails` list

**Note:** `{**state, "emails": new_emails}` means "take everything in the current state, and overwrite just these two fields"

---

#### `new_emails(state)` — The Decision Function (Router)

```python
def new_emails(self, state):
    if len(state['emails']) == 0:
        return "end"     # → go to wait_next_run
    else:
        return "continue"  # → go to draft_responses (CrewAI)
```

**What it does:**
This is **not a node**. It's a **routing function** — it tells LangGraph which path to take.
- If no new emails → go wait and check again later
- If new emails exist → call CrewAI crew to process them

---

#### `wait_next_run(state)` — Node 3

```python
def wait_next_run(self, state):
    time.sleep(180)  # Wait 3 minutes
    return state     # State unchanged
```

**What it does:**
Just pauses for 3 minutes, then returns the same state. After this, the graph loops back to `check_new_emails`.

---

### 📄 `graph.py` — The LangGraph Wiring

This is the **most important file** for understanding how everything is connected.

```python
class WorkFlow():
    def __init__(self):
        nodes = Nodes()
        workflow = StateGraph(EmailsState)  # Create graph with shared state type

        # Add the three nodes
        workflow.add_node("check_new_emails", nodes.check_email)
        workflow.add_node("wait_next_run", nodes.wait_next_run)
        workflow.add_node("draft_responses", EmailFilterCrew().kickoff)  # ← CrewAI here!

        # Set where the graph starts
        workflow.set_entry_point("check_new_emails")

        # Add decision: after checking emails, where to go?
        workflow.add_conditional_edges(
            "check_new_emails",       # FROM this node
            nodes.new_emails,         # Run this function to decide
            {
                "continue": 'draft_responses',  # if "continue" → run CrewAI
                "end": 'wait_next_run'           # if "end" → skip CrewAI, just wait
            }
        )

        # After CrewAI is done → wait
        workflow.add_edge('draft_responses', 'wait_next_run')

        # After waiting → check again (the loop!)
        workflow.add_edge('wait_next_run', 'check_new_emails')

        self.app = workflow.compile()  # Lock in the graph and make it runnable
```

**The crucial line:**
```python
workflow.add_node("draft_responses", EmailFilterCrew().kickoff)
```
This is **where LangGraph hands over to CrewAI**. The `kickoff` method of `EmailFilterCrew` is registered as a LangGraph node. When the graph reaches `draft_responses`, it calls CrewAI's team.

**LangGraph expects:** every node function to receive `state` and return updated `state`.
**CrewAI's `kickoff(state)`** does exactly that — takes state with emails, runs agents, returns state with `action_required_emails` added.

---

### 📄 `crew/agents.py` — The Three AI Workers

There are **3 agents**, each with a different personality and role:

#### 🕵️ Agent 1: `email_filter_agent` — The Gatekeeper

```python
Agent(
    role='Senior Email Analyst',
    goal='Filter out non-essential emails like newsletters and promotional content',
    backstory='...extensive experience in email content analysis...',
    allow_delegation=False
)
```

- **Job:** Look at the batch of emails and decide which ones are worth responding to
- **Tools:** None (uses its LLM intelligence alone)
- **Output:** A list of thread IDs that actually need attention

---

#### 🔍 Agent 2: `email_action_agent` — The Analyst

```python
Agent(
    role='Email Action Specialist',
    goal='Identify action-required emails and compile a list of their IDs',
    backstory='...specializes in identifying emails that require immediate action...',
    tools=[
        GmailGetThread(...)   # Can READ full email thread
        TavilySearchResults() # Can SEARCH the web for context
    ]
)
```

- **Job:** For each thread that passed the filter, read the full email thread and understand what needs to be done
- **Tools:** Can pull the full Gmail thread, can search the internet
- **Output:** For each email: thread ID, summary, key points, sender's email, communication style

---

#### ✍️ Agent 3: `email_response_writer` — The Writer

```python
Agent(
    role='Email Response Writer',
    goal='Draft responses to action-required emails',
    backstory='...skilled writer, adept at crafting clear, concise, and effective email responses...',
    tools=[
        TavilySearchResults(),          # Can SEARCH web for facts if needed
        GmailGetThread(...),             # Can RE-READ thread for more context
        CreateDraftTool.create_draft    # Can CREATE draft in Gmail!
    ]
)
```

- **Job:** Write actual email replies and save them to Gmail Drafts
- **Tools:** Web search, re-read thread, create Gmail draft
- **Output:** Confirmation that drafts were created

---

### 📄 `crew/tasks.py` — The Instructions for Each Agent

Each `Task` is like a **job description card** handed to an agent.

#### Task 1: `filter_emails_task` → given to Agent 1

```
"Here are all the emails. Filter out newsletters, promotions, notifications.
Return the thread IDs and senders of emails worth responding to."
```

Input: the raw emails string (passed in via `state['emails']`)

---

#### Task 2: `action_required_emails_task` → given to Agent 2

```
"Take the thread IDs from the previous step.
Pull the full thread from Gmail. Understand the context.
Return: thread_id, summary, key points, who to reply to, their communication style, sender email."
```

Note: This task **doesn't get explicit input** — it automatically uses the output of Task 1 (CrewAI chains tasks by default).

---

#### Task 3: `draft_responses_task` → given to Agent 3

```
"Based on the analysis above, draft replies for each email.
Match the user's communication style.
Do web research if needed.
Use the CreateDraftTool to save each draft to Gmail.
Confirm all drafts were created."
```

---

### 📄 `crew/crew.py` — Assembling the Team

```python
class EmailFilterCrew():
    def __init__(self):
        agents = EmailFilterAgents()
        self.filter_agent = agents.email_filter_agent()
        self.action_agent = agents.email_action_agent()
        self.writer_agent = agents.email_response_writer()

    def kickoff(self, state):  # ← This is called by LangGraph!
        tasks = EmailFilterTasks()
        crew = Crew(
            agents=[self.filter_agent, self.action_agent, self.writer_agent],
            tasks=[
                tasks.filter_emails_task(self.filter_agent, self._format_emails(state['emails'])),
                tasks.action_required_emails_task(self.action_agent),
                tasks.draft_responses_task(self.writer_agent)
            ],
            verbose=True
        )
        result = crew.kickoff()
        return {**state, "action_required_emails": result}  # ← Back to LangGraph!

    def _format_emails(self, emails):
        # Helper: converts list of email dicts into a readable string for the LLM
        ...
```

**Key design pattern:**
- `kickoff(self, state)` receives the LangGraph state
- It extracts `state['emails']`
- Runs the crew
- Returns updated state → LangGraph continues

This is the **bridge function** that connects both worlds.

---

### 📄 `crew/tools.py` — The Gmail Draft Creator

```python
class CreateDraftTool():
    @tool("Create Draft")
    def create_draft(data):
        """
        Input: "email@example.com|Subject Here|Body of the email"
        """
        email, subject, message = data.split('|')
        gmail = GmailToolkit()
        draft = GmailCreateDraft(api_resource=gmail.api_resource)
        result = draft({'to': [email], 'subject': subject, 'message': message})
        return f"Draft created: {result}"
```

**What this is:**
A custom LangChain `@tool` that the writer agent can call.
The agent passes a pipe-separated string like `"user@gmail.com|Re: Meeting|Sure, I'm free at 3pm!"` and the tool creates an actual Gmail draft.

The pipe `|` separator is a design choice — the LLM is instructed (in the task description) to pass data in this exact format.

---

### 📄 `main.py` — The Entry Point

```python
from src.graph import WorkFlow

app = WorkFlow().app
app.invoke({})
```

**What this does:**
1. Creates the compiled LangGraph graph
2. Starts it running with an empty initial state `{}`
3. The graph runs indefinitely (infinite loop built into graph edges)

`app.invoke({})` = "start the graph and give it an empty starting state"

---

## 🔄 Part 5: Full Execution Flow — Step by Step

Here is exactly what happens when you run `python main.py`:

```
1. WorkFlow() is created
   └── LangGraph graph is built and compiled

2. app.invoke({}) starts the graph

3. Node: check_new_emails
   ├── Connects to Gmail API
   ├── Fetches emails from last 24 hours
   ├── Filters out already-seen, duplicate threads, your own emails
   └── Updates state: {emails: [...], checked_emails_ids: [...]}

4. Decision: new_emails(state)
   ├── IF state['emails'] is empty → go to wait_next_run
   └── IF state['emails'] has items → go to draft_responses

5a. IF no new emails:
    Node: wait_next_run
    └── Sleeps 180 seconds → loops back to step 3

5b. IF new emails found:
    Node: draft_responses (← this is EmailFilterCrew().kickoff)
    │
    ├── CrewAI starts
    │
    ├── Agent 1 (Email Analyst) runs filter_emails_task
    │   ├── Reads email snippets
    │   └── Returns: thread IDs worth responding to
    │
    ├── Agent 2 (Action Specialist) runs action_required_emails_task
    │   ├── Pulls full threads from Gmail
    │   ├── May search web for context
    │   └── Returns: structured summary per thread
    │
    ├── Agent 3 (Response Writer) runs draft_responses_task
    │   ├── Writes draft replies
    │   ├── May search web for facts
    │   ├── Calls CreateDraftTool → saves to Gmail Drafts
    │   └── Returns: confirmation
    │
    └── kickoff() returns {**state, "action_required_emails": result}

6. Node: wait_next_run
   └── Sleeps 180 seconds

7. → Back to step 3 (infinite loop)
```

---

## 🛠️ Part 6: Tools & APIs Used

| Tool / Library | What it does in this project |
|---|---|
| `langgraph` | Controls the flow graph — loop, conditions, state |
| `crewai` | Multi-agent system — agents, tasks, crew |
| `langchain_community` | Provides Gmail tools (read, search, draft) |
| `GmailToolkit` | Connects to Gmail via OAuth |
| `GmailSearch` | Searches inbox by query |
| `GmailGetThread` | Reads full email thread by thread ID |
| `GmailCreateDraft` | Creates a draft email in Gmail |
| `TavilySearchResults` | Web search (like Google, but API-based) |
| `python-dotenv` | Loads API keys from `.env` file |

---

## 🔑 Part 7: Environment Variables (`.env` file)

The project needs these secrets:

```
OPENAI_API_KEY=...       # LLM for all AI agents
TAVILY_API_KEY=...       # Web search for agents
MY_EMAIL=...             # Your Gmail address (to filter out your own emails)
```

Gmail authentication is handled via OAuth (Google's standard login flow). You'll need to set up a Google Cloud project and download `credentials.json`.

---

## 🧩 Part 8: Key Design Patterns to Learn From

### Pattern 1: LangGraph as the Outer Loop
LangGraph is used ONLY for flow control:
- infinite loop
- conditional routing
- waiting between cycles

It does NOT do any AI work itself here.

### Pattern 2: CrewAI as a Node
The entire CrewAI system is plugged in as a **single LangGraph node**.
`EmailFilterCrew().kickoff` is passed directly to `workflow.add_node()`.
This makes the integration seamless — LangGraph just treats CrewAI like any other function.

### Pattern 3: State as the Contract
`EmailsState` is the shared contract between LangGraph and CrewAI:
- LangGraph writes `emails` into state
- CrewAI reads `emails` from state, processes them
- CrewAI writes `action_required_emails` back into state
- LangGraph continues with updated state

### Pattern 4: Sequential CrewAI Tasks with Implicit Chaining
CrewAI's tasks are ordered. Task 2 doesn't explicitly receive Task 1's output — CrewAI automatically passes the output of each task as context to the next one. This is **CrewAI's built-in task chaining**.

### Pattern 5: Custom Tools with @tool Decorator
`CreateDraftTool` shows how to wrap any Python function as an LLM-callable tool:
```python
@tool("Tool Name")
def my_function(input: str):
    """Description that the LLM reads to understand when/how to use this tool"""
    ...
```

---

## 📊 Part 9: Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                     LangGraph                        │
│  ┌──────────────────────────────────────────────┐   │
│  │  GRAPH (infinite loop)                        │   │
│  │                                               │   │
│  │  [check_new_emails] ──── no emails ──→        │   │
│  │        │                          [wait]      │   │
│  │    new emails                       │         │   │
│  │        │                            │         │   │
│  │        ▼                            │         │   │
│  │  [draft_responses] ──────────→ [wait] ────┐   │   │
│  │        │                                  │   │   │
│  │        │ (calls CrewAI)                   └───┘   │
│  │        ▼                                          │
│  │  ┌──────────────────────────────────────┐         │
│  │  │            CrewAI Crew               │         │
│  │  │                                      │         │
│  │  │  Agent 1: Filter Emails              │         │
│  │  │      ↓                               │         │
│  │  │  Agent 2: Analyze + Research         │         │
│  │  │      ↓                               │         │
│  │  │  Agent 3: Write + Create Gmail Draft │         │
│  │  └──────────────────────────────────────┘         │
│  └──────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────┘
```

---

## 🚀 Part 10: What to Know Before Building Your Own Project

When you build your own project using this pattern, here's the mental model:

### Step 1: Design your LangGraph flow first
- What are the steps? (nodes)
- What decisions need to be made? (conditional edges)
- What loops back on itself?
- What is the shared state?

### Step 2: Identify which nodes are "heavy AI work"
- Those become CrewAI nodes
- Everything else (checking, waiting, routing) stays as plain Python

### Step 3: Design your CrewAI crew
- What agents do you need? (what roles?)
- What tools does each agent need?
- What tasks do they do in sequence?
- How do tasks chain together?

### Step 4: Build the bridge
- Your crew's `kickoff(state)` method should:
  - Accept LangGraph state as input
  - Extract what it needs from state
  - Run the crew
  - Return updated state

### Step 5: Wire it all together in `graph.py`
```python
workflow.add_node("your_ai_work", YourCrew().kickoff)
```

---

## 📦 Part 11: Dependencies Explained

```
crewai==0.130.0          → The multi-agent AI framework
langgraph==1.0.10rc1     → The graph-based flow control framework
langchain-community      → Gmail tools, Tavily search, etc.
python-dotenv            → Load .env file (API keys)
google-api-python-client → Google's own Python SDK for Gmail
google-auth-oauthlib     → Gmail OAuth2 login flow
google-auth-httplib2     → HTTP transport for Google auth
tavily-python            → Tavily web search (used by agents)
beautifulsoup4           → HTML parsing (for web content)
```

---

## 💡 Summary: The One-Liner Explanation

> **LangGraph decides WHEN to do things and in WHAT ORDER. CrewAI decides HOW to do the actual AI work with a team of specialized agents. They are connected through a shared `state` dictionary, where LangGraph writes the inputs and CrewAI writes the outputs.**

---

*These notes were prepared based on the full source code of the `CrewAI-LangGraph` example from the official CrewAI examples repository.*
