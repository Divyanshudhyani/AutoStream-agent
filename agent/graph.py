from typing import TypedDict, Annotated, Optional, List
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os
from concurrent.futures import ThreadPoolExecutor

from agent.rag_pipeline import AutoStreamRAG
from agent.intent_detector import IntentDetector
from tools.lead_capture import mock_lead_capture

# State Schema

class AgentState(TypedDict):
    # Full conversation history (appended, not overwritten)
    messages: Annotated[list, add_messages]
    
    # Detected intent for the latest turn
    current_intent: Optional[str]
    
    # Lead collection progress
    collecting_lead: bool
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    
    # Whether lead has been captured already
    lead_captured: bool
    
    # RAG context retrieved for the current turn
    rag_context: Optional[str]

# System Prompt Builder

AGENT_SYSTEM_TEMPLATE = """You are Aria, a friendly and knowledgeable AI assistant for AutoStream — an AI-powered video editing SaaS for content creators.

Your personality: warm, professional, enthusiastic about helping creators grow.

## Style Guidelines:
- Use simple, direct language. 
- Avoid corporate jargon like "streamline," "optimize," or "synergize."
- Instead of "streamline," say "make faster," "simplify," or "clean up."
- Keep it conversational and helpful.

## Knowledge Base Context (use ONLY this for product facts):
{rag_context}

## Your Responsibilities:
1. Answer product/pricing questions accurately using ONLY the knowledge base above.
2. When a user shows interest in signing up, smoothly guide them through lead capture.
3. Collect: Name → Email → Creator Platform (in order, one at a time).
4. Never make up features or prices not in the knowledge base.
5. Keep responses concise (2-4 sentences max) unless explaining pricing in detail.

## Lead Collection State:
- Collecting lead: {collecting_lead}
- Name collected: {lead_name}
- Email collected: {lead_email}  
- Platform collected: {lead_platform}

## Instructions based on state:
{lead_instructions}

Always be helpful, never pushy. If the user seems uncertain, reassure them about the free trial.
"""


def build_lead_instructions(state: AgentState) -> str:
    if not state.get("collecting_lead"):
        return "Respond naturally. If user shows high intent, warmly transition to asking for their name."
    
    if not state.get("lead_name"):
        return "Ask for the user's full name to get started."
    elif not state.get("lead_email"):
        return f"You have their name ({state['lead_name']}). Now ask for their email address."
    elif not state.get("lead_platform"):
        return f"You have name and email. Ask which creator platform they primarily use (YouTube, Instagram, TikTok, etc.)."
    else:
        return "All lead info collected. Confirm and wrap up warmly."

# AutoStream Agent
class AutoStreamAgent:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-flash-lite-latest",
            google_api_key=api_key,
            temperature=0.7,
            streaming=True
        )
        
        # RAG pipeline
        self.rag = AutoStreamRAG()
        
        # Intent detector (separate Gemini call)
        self.intent_detector = IntentDetector(api_key=api_key)
        
        # Build the LangGraph
        self.graph = self._build_graph()

    # Node: Detect Intent 
    def detect_intent_node(self, state: AgentState) -> dict:
        """Use Gemini to classify the latest user message intent."""
        messages = state.get("messages", [])
        if not messages:
            return {"current_intent": "greeting"}
        
        # Get the last human message
        last_human = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_human = msg.content
                break
        
        if not last_human:
            return {"current_intent": "inquiry"}
        
        # Build history for context
        history = []
        for msg in messages[:-1]:  # Exclude latest
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        
        result = self.intent_detector.detect(last_human, history)
        return {"current_intent": result.get("intent", "inquiry")}

    # Node: RAG Retrieval 
    def retrieve_context_node(self, state: AgentState) -> dict:
        """Retrieve relevant knowledge base context for the current query."""
        messages = state.get("messages", [])
        last_human = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_human = msg.content
                break
        
        context = self.rag.retrieve(last_human) if last_human else ""
        return {"rag_context": context}

    # Node: Extract Lead Info 
    def extract_lead_info_node(self, state: AgentState) -> dict:
        """
        When collecting a lead, attempt to extract name/email/platform
        from the latest user message using Gemini.
        """
        if not state.get("collecting_lead"):
            return {}
        
        messages = state.get("messages", [])
        last_human = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_human = msg.content
                break
        
        if not last_human:
            return {}

        updates = {}
        
        # What are we currently waiting for?
        if not state.get("lead_name"):
            # Extract name from message
            name = self._extract_field(last_human, "name", state)
            if name:
                updates["lead_name"] = name
        
        elif not state.get("lead_email"):
            # Extract email
            import re
            email_match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", last_human)
            if email_match:
                updates["lead_email"] = email_match.group(0)
        
        elif not state.get("lead_platform"):
            # Extract platform
            platforms = ["youtube", "instagram", "tiktok", "twitter", "facebook", "linkedin", "twitch"]
            msg_lower = last_human.lower()
            for platform in platforms:
                if platform in msg_lower:
                    updates["lead_platform"] = platform.capitalize()
                    break
            if not updates.get("lead_platform") and len(last_human.strip()) < 50:
                # Treat the whole message as the platform name
                updates["lead_platform"] = last_human.strip()
        
        return updates

    def _extract_field(self, text: str, field: str, state: AgentState) -> Optional[str]:
        """Use heuristics to extract a specific field from user text."""
        text = text.strip()
        
        if field == "name":
            # Likely just said their name
            # Remove common prefixes
            import re
            cleaned = re.sub(r"^(my name is|i am|i'm|call me|it's|its)\s*", "", text, flags=re.IGNORECASE).strip()
            # If it's 1-4 words and looks like a name, use it
            words = cleaned.split()
            if 1 <= len(words) <= 4 and all(w[0].isalpha() for w in words if w):
                return cleaned.title()
        
        return None

    # Node: Check Lead Completion & Fire Tool
    def check_lead_complete_node(self, state: AgentState) -> dict:
        """If all lead fields collected, fire the mock_lead_capture tool."""
        if (state.get("collecting_lead") and
                state.get("lead_name") and
                state.get("lead_email") and
                state.get("lead_platform") and
                not state.get("lead_captured")):
            
            # Fire the tool
            mock_lead_capture(
                name=state["lead_name"],
                email=state["lead_email"],
                platform=state["lead_platform"]
            )
            return {"lead_captured": True}
        
        return {}

    #  Node: Generate Response 
    def _prepare_llm_messages(self, state: AgentState) -> tuple[list, bool]:
        """Prepare system prompt and message history for the LLM."""
        intent = state.get("current_intent", "inquiry")
        
        # Trigger lead collection on high intent
        collecting = state.get("collecting_lead", False)
        if intent == "high_intent" and not collecting and not state.get("lead_captured"):
            collecting = True
        
        rag_context = state.get("rag_context", self.rag.get_full_context())
        lead_instructions = build_lead_instructions({**state, "collecting_lead": collecting})
        
        system_prompt = AGENT_SYSTEM_TEMPLATE.format(
            rag_context=rag_context,
            collecting_lead=collecting,
            lead_name=state.get("lead_name") or "Not collected",
            lead_email=state.get("lead_email") or "Not collected",
            lead_platform=state.get("lead_platform") or "Not collected",
            lead_instructions=lead_instructions
        )
        
        # Special override: if lead just captured
        if state.get("lead_captured"):
            system_prompt += (
                "\n\nThe lead has been SUCCESSFULLY captured. "
                "Thank the user warmly, confirm their details, and let them know the team will reach out soon. "
                "Mention the 7-day free trial if they haven't started yet."
            )
        
        # Build messages for LLM
        llm_messages = [SystemMessage(content=system_prompt)]
        
        # Add conversation history (last 10 turns)
        history = state.get("messages", [])
        llm_messages.extend(history[-10:])
        
        return llm_messages, collecting

    #  Node: Generate Response 
    def generate_response_node(self, state: AgentState) -> dict:
        """Generate the AI response using Gemini with full context."""
        llm_messages, collecting = self._prepare_llm_messages(state)
        
        response = self.llm.invoke(llm_messages)
        ai_message = AIMessage(content=response.content)
        
        updates = {
            "messages": [ai_message],
            "collecting_lead": collecting
        }
        
        return updates

    # Graph Builder 
    def _build_graph(self):
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("detect_intent", self.detect_intent_node)
        graph.add_node("retrieve_context", self.retrieve_context_node)
        graph.add_node("extract_lead_info", self.extract_lead_info_node)
        graph.add_node("check_lead_complete", self.check_lead_complete_node)
        graph.add_node("generate_response", self.generate_response_node)
        
        # Define edges (linear pipeline)
        graph.set_entry_point("detect_intent")
        graph.add_edge("detect_intent", "retrieve_context")
        graph.add_edge("retrieve_context", "extract_lead_info")
        graph.add_edge("extract_lead_info", "check_lead_complete")
        graph.add_edge("check_lead_complete", "generate_response")
        graph.add_edge("generate_response", END)
        
        return graph.compile()

    def chat(self, user_message: str, state: AgentState):
        """
        Stream a single user turn.
        
        Yields:
            tokens (str) or the final state (dict)
        """
        # Append user message to state
        state["messages"] = state.get("messages", []) + [HumanMessage(content=user_message)]
        
        # 1. Run all nodes EXCEPT the final generation
        # We'll use a sub-graph or just run nodes manually for more control over streaming
        # But for simplicity, we can run the graph up to check_lead_complete
        
        # Create a partial graph execution
        # We can use the graph's internal nodes if needed, but let's just use the compiled graph
        # and intercept the results.
        
        # Actually, simpler: Run the graph normally to get all pre-processing done.
        # Then manually call the streaming LLM.
        
        # We need to run up to 'check_lead_complete'
        # Let's find a way to stop before 'generate_response'
        
        current_state = state
        
        # Parallelize the independent nodes to reduce latency
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 1. Detect Intent
            future_intent = executor.submit(self.detect_intent_node, current_state)
            # 2. Retrieve Context (keyword-based, very fast but runs while intent is being detected)
            future_context = executor.submit(self.retrieve_context_node, current_state)
            
            # Wait for both
            intent_updates = future_intent.result()
            context_updates = future_context.result()
            
            # Apply updates
            for updates in [intent_updates, context_updates]:
                for k, v in updates.items():
                    if k == "messages":
                        current_state["messages"] = current_state.get("messages", []) + v
                    else:
                        current_state[k] = v

        # 3. These nodes are fast/sequential
        nodes_to_run = ["extract_lead_info", "check_lead_complete"]
        
        for node_name in nodes_to_run:
            node_fn = getattr(self, f"{node_name}_node")
            updates = node_fn(current_state)
            for k, v in updates.items():
                if k == "messages":
                    current_state["messages"] = current_state.get("messages", []) + v
                else:
                    current_state[k] = v
        
        # Now we have the state ready for generation
        llm_messages, collecting = self._prepare_llm_messages(current_state)
        current_state["collecting_lead"] = collecting
        
        # 2. Stream from LLM
        full_response = ""
        for chunk in self.llm.stream(llm_messages):
            content = chunk.content
            # Handle structured content if it's a list
            if isinstance(content, list):
                text = "".join(c.get("text", "") for c in content if isinstance(c, dict) and "text" in c)
            else:
                text = str(content)
            
            full_response += text
            yield text
        
        # 3. Finalize state
        ai_message = AIMessage(content=full_response)
        current_state["messages"] = current_state.get("messages", []) + [ai_message]
        
        # Yield the final state at the end (marked by a special type or just the last yield)
        yield current_state


def create_initial_state() -> AgentState:
    """Create a fresh initial agent state."""
    return AgentState(
        messages=[],
        current_intent=None,
        collecting_lead=False,
        lead_name=None,
        lead_email=None,
        lead_platform=None,
        lead_captured=False,
        rag_context=None
    )


