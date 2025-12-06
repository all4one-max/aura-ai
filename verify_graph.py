from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.graph import create_graph


def run_test(test_name, initial_message, user_message1, user_message2, expected_flow):
    print(f"--- Running Test: {test_name} ---")

    # Create graph with REAL agents
    graph = create_graph()

    # Initial state
    state = {
        "messages": [HumanMessage(content=initial_message)],
        "user_profile": {},
        "thread_id": "thread_id-1",
    }
    config = {"configurable": {"thread_id": "thread_id-1", "user_id": "sjha"}}

    try:
        events = graph.stream(state, config=config)

        step_count = 0
        executed_flow = []

        interrupt = False
        for event in events:
            for key, value in event.items():
                print(f"Node: {key}")
                if key == "__interrupt__":
                    interrupt = True
                executed_flow.append(key)
                if "messages" in value:
                    print(f"  Message: {value['messages'][0].content}")
                if "next_step" in value:
                    print(f"  Next Step: {value['next_step']}")
            step_count += 1
            if step_count > 10:
                print("Force stopping loop")
                break

        if interrupt:
            result = graph.invoke(
                Command(resume=HumanMessage(content=user_message1)),
                config=config,
            )
            if result.get("__interrupt__"):
                executed_flow.append("__interrupt__")
                result = graph.invoke(
                    Command(resume=HumanMessage(content=user_message2)),
                    config=config,
                )
            print(result)
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback

        traceback.print_exc()

    print(f"Executed Flow: {executed_flow}")

    # Check if expected flow is a SUBSEQUENCE of executed flow
    matches = True
    for node in expected_flow:
        if node not in executed_flow:
            matches = False
            break

    if matches:
        print("✅ Test Passed")
    else:
        print(f"❌ Test Failed. Expected {expected_flow}")
    print("\n")


if __name__ == "__main__":
    # Test 1: Recommendation -> Research -> Styling
    # run_test(
    #     "Recommendation and Fitment",
    #     "Recommend me some shirts",
    #     "the occasion is beach wedding",
    #     ["context_agent", "__interrupt__"],
    # )

    run_test(
        "Recommendation and Fitment",
        "Recommend me some shirts",
        "I am going to thailand",
        "I am going for a beach party",
        ["context_agent", "__interrupt__", "__interrupt__"],
    )

    run_test(
        "Test using ChatQuery state so that we don't ask input every time",
        "show me something in red",
        "I am going to thailand",
        "I am going for a beach party",
        ["context_agent", "research_agent", "styling_agent"],
    )

    # # Test 2: Refinement (Matching Pant) -> Research -> Styling
    # # Note: For real LLM we might need to be more explicit or carry over state,
    # # but let's see if it infers solely from "Get me a matching pant" that it should research first.
    # run_test(
    #     "Refinement (Matching Pant)",
    #     "Get me a matching pant with it",
    #     ["context_agent", "research_agent", "styling_agent"]
    # )

    # # Test 3: Price Check (Cheaper) -> Research -> End (No styling)
    # run_test(
    #     "Price Check (Cheaper)",
    #     "Can you get me a cheaper one?",
    #     ["context_agent", "research_agent"]
    # )

    # Test 4: Purchase
    # run_test(
    #     "Purchase",
    #     "Order these 2 pairs",
    #     ["context_agent", "fulfillment_agent"]
    # )

    # Test 7: In the last run research agent was callend but in the next run we call style agent
    # oh i like the red one, can you show how would i look
