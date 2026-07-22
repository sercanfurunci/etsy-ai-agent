# ponytail: temporary test script — remove once Claude is wired into the real pipeline
from agent.claude_client import ask

if __name__ == "__main__":
    prompt = input("Test prompt: ").strip()
    if not prompt:
        print("No prompt entered.")
    else:
        try:
            print("\nClaude says:\n")
            print(ask(prompt))
        except ValueError as e:
            print(f"[Config error] {e}")
        except Exception as e:
            print(f"[API error] {e}")
