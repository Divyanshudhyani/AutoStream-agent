import os
import sys
from dotenv import load_dotenv

load_dotenv()

def check_env():
    if not os.environ.get("GEMINI_API_KEY"):
        print("\nERROR: GEMINI_API_KEY is not set.")
        sys.exit(1)

def print_banner():
    print("  AutoStream — AI Sales Agent")
    print("  Type your message and press Enter.")
    print("  Type 'quit' or 'exit' to end the session.")
    print("  Type 'reset' to start a fresh conversation.")

def print_turn_info(intent: str, collecting: bool, lead_name: str, lead_email: str, lead_platform: str):
    if collecting:
        fields = []
        if lead_name: fields.append(f"name='{lead_name}'")
        if lead_email: fields.append(f"email='{lead_email}'")
        if lead_platform: fields.append(f"platform='{lead_platform}'")
        if fields:
            print(f" | {', '.join(fields)}", end="")
    print()


def main():
    check_env()
    
    # Import after env check
    from agent.graph import AutoStreamAgent, create_initial_state
    
    print_banner()
    print("  Initializing AutoStream Agent...")
    
    try:
        agent = AutoStreamAgent()
        print("  Agent ready!\n")
    except Exception as e:
        print(f"\n Failed to initialize agent: {e}\n")
        sys.exit(1)

    state = create_initial_state()
    debug_mode = "--debug" in sys.argv

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! \n")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("\nGoodbye! \n")
            break

        if user_input.lower() == "reset":
            state = create_initial_state()
            print("\n  Conversation reset.\n")
            continue

        try:
            print("\n  Thinking...", end="\r", flush=True)
            
            # agent.chat is now a generator
            full_reply = ""
            first_chunk = True
            for chunk in agent.chat(user_input, state):
                if isinstance(chunk, str):
                    if first_chunk:
                        print("Aria: ", end="", flush=True)
                        first_chunk = False
                    print(chunk, end="", flush=True)
                    full_reply += chunk
                else:
                    # Final chunk is the updated state
                    state = chunk
            
            if debug_mode:
                print_turn_info(
                    intent=state.get("current_intent", "—"),
                    collecting=state.get("collecting_lead", False),
                    lead_name=state.get("lead_name"),
                    lead_email=state.get("lead_email"),
                    lead_platform=state.get("lead_platform")
                )
            
            print("\n")

        except Exception as e:
            print(f"\n  Error processing message: {e}\n")
            if "--verbose" in sys.argv:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
