from agent.core import run

if __name__ == "__main__":
    topic = input("Enter a product topic to research: ").strip()
    if not topic:
        print("No input provided.")
    else:
        run(topic)
