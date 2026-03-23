from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI(title="Dental Appointment AI Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dental instructions once at startup
with open("dental.txt", "rb") as f:
    dental_txt = f.read()

# Initialize LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    api_key="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"
)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a Dental Appointment AI Agent.

Your job:
- Guide the user step by step to book a dental appointment.
- Follow the appointment workflow instructions.
- Do NOT skip steps.
- Use the conversation history to determine the next step.
- Only use the information provided in instructions, do not add anything else.

Instructions:
{dental_txt}

Conversation History:
{user_and_ai}
"""
    ),
    ("human", "{input}")
])

# In-memory session store: { session_id: [ (user, ai), ... ] }
sessions: dict[str, list[tuple[str, str]]] = {}


# ---------- Request / Response Models ----------

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    ended: bool

class SessionResponse(BaseModel):
    session_id: str
    history: list[dict]


# ---------- Endpoints ----------

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Send a message and get an AI reply."""
    session_id = req.session_id
    user_message = req.message.strip()

    # Create session if new
    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]

    # Build and invoke prompt
    formatted = prompt.format_prompt(
        dental_txt=dental_txt,
        user_and_ai=history,
        input=user_message
    )
    ai_response = llm.invoke(formatted.to_messages())
    reply = ai_response.content

    # Save to history
    history.append((user_message, reply))

    ended = user_message.lower() == "bye"
    if ended:
        del sessions[session_id]  # Clean up session on exit

    return ChatResponse(session_id=session_id, reply=reply, ended=ended)


@app.get("/session/{session_id}", response_model=SessionResponse)
def get_session(session_id: str):
    """Get full conversation history for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    history = [
        {"user": u, "ai": a} for u, a in sessions[session_id]
    ]
    return SessionResponse(session_id=session_id, history=history)


@app.delete("/session/{session_id}")
def end_session(session_id: str):
    """Manually end and clear a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del sessions[session_id]
    return {"message": f"Session '{session_id}' ended."}


@app.get("/health")
def health():
    return {"status": "ok"}