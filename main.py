from app.graph import create_graph
from langchain_core.messages import HumanMessage

def main():
    print("Initializing Conversational Stylist...")
    app = create_graph()
    
    initial_state = {
        "messages": [HumanMessage(content="I need a dress for a beach wedding under $200.")],
        "user_profile": None,
        "search_results": [],
        "selected_item": None,
        "next_step": None
    }
    
    print("\n--- Starting Conversation ---\n")
    for output in app.stream(initial_state):
        for key, value in output.items():
            print(f"Finished: {key}")
            if "messages" in value:
                print(f"Message: {value['messages'][0].content}")
    print("\n--- End of Conversation ---")

if __name__ == "__main__":
    main()
