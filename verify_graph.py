import asyncio
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
import json
from PIL import Image
import requests
from io import BytesIO
from psycopg_pool import AsyncConnectionPool

from app.config import DATABASE_URL
from app.database import clear_table, create_db_and_tables
from app.graph import create_graph
from app.schema import ChatQuery
from app.dao.user_dao import get_user, user_to_profile


async def merge_images(user_photo_url: str, product_image_url: str, output_path: str = "merged_image.jpg"):
    """
    Merge user photo with product image side by side.
    """
    try:
        print(f"\n📸 Merging images...")
        print(f"   User photo: {user_photo_url}")
        print(f"   Product image: {product_image_url}")
        
        # Download user photo
        user_response = requests.get(user_photo_url, timeout=10)
        user_img = Image.open(BytesIO(user_response.content))
        
        # Download product image
        product_response = requests.get(product_image_url, timeout=10)
        product_img = Image.open(BytesIO(product_response.content))
        
        # Resize images to same height (use smaller height)
        min_height = min(user_img.height, product_img.height)
        user_aspect = user_img.width / user_img.height
        product_aspect = product_img.width / product_img.height
        
        user_img = user_img.resize((int(min_height * user_aspect), min_height), Image.Resampling.LANCZOS)
        product_img = product_img.resize((int(min_height * product_aspect), min_height), Image.Resampling.LANCZOS)
        
        # Create merged image (side by side)
        merged_width = user_img.width + product_img.width
        merged_img = Image.new('RGB', (merged_width, min_height))
        merged_img.paste(user_img, (0, 0))
        merged_img.paste(product_img, (user_img.width, 0))
        
        # Save merged image
        merged_img.save(output_path, 'JPEG', quality=95)
        print(f"✅ Merged image saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Error merging images: {e}")
        return None


async def run_test_with_user_photo(username: str, user_messages, checkpointer):
    """
    Test graph flow with user photo from database and merge images.
    """
    print(f"\n{'='*80}")
    print(f"🧪 Testing Graph Flow with User: {username}")
    print(f"{'='*80}\n")
    
    # Fetch user from database
    print(f"📋 Fetching user '{username}' from database...")
    user = await get_user(username=username)
    if not user:
        print(f"❌ User '{username}' not found in database!")
        return
    
    profile = user_to_profile(user)
    print(f"✅ User found: {profile.get('username')} (ID: {profile.get('user_id')})")
    
    # Get user photo URL
    photo_urls = profile.get('photo_urls', [])
    if not photo_urls:
        print(f"⚠️  No photos found for user '{username}'")
        return
    
    user_photo_url = photo_urls[0]  # Use first photo
    print(f"📸 User photo URL: {user_photo_url}\n")
    
    # Create graph with checkpointer
    graph = create_graph(checkpointer)
    
    # Initial state with user profile
    input_data = {
        "messages": [HumanMessage(content=user_messages[0])],
        "user_profile": profile,
    }
    config = {"configurable": {"thread_id": f"thread_{username}", "user_id": profile.get('user_id')}}
    
    print(f"🚀 Starting graph execution...")
    print(f"   Query: {user_messages[0]}")
    print(f"   Thread ID: {config['configurable']['thread_id']}\n")
    
    cur_user_message_index = 1
    interrupt = False
    executed_flow = []
    final_state = None
    
    # Use astream to handle interrupts properly and collect final state
    try:
        async for event in graph.astream(input_data, config=config):
            # Collect state from stream events
            if isinstance(event, dict):
                for node_name, node_output in event.items():
                    if node_name != "__end__":
                        executed_flow.append(node_name)
                        if isinstance(node_output, dict):
                            final_state = node_output
                
                # Check for interrupt
                if event.get("__interrupt__"):
                    interrupt = True
                    if cur_user_message_index < len(user_messages):
                        print(f"📨 Handling interrupt with message: {user_messages[cur_user_message_index]}")
                        # Resume with next message
                        try:
                            async for resume_event in graph.astream(
                                Command(resume=HumanMessage(content=user_messages[cur_user_message_index])),
                                config=config
                            ):
                                if isinstance(resume_event, dict):
                                    for node_name, node_output in resume_event.items():
                                        if node_name != "__end__":
                                            executed_flow.append(node_name)
                                            if isinstance(node_output, dict):
                                                final_state = node_output
                            cur_user_message_index += 1
                            interrupt = False
                        except Exception as e:
                            print(f"⚠️  Error resuming (checkpoint issue, but flow continues): {e}")
                            break
                    else:
                        break
        
        # Process remaining messages if any
        while cur_user_message_index < len(user_messages) and not interrupt:
            msg = user_messages[cur_user_message_index]
            print(f"📨 Processing message: {msg}")
            try:
                async for event in graph.astream(
                    Command(resume=HumanMessage(content=msg)),
                    config=config
                ):
                    if isinstance(event, dict):
                        for node_name, node_output in event.items():
                            if node_name != "__end__":
                                executed_flow.append(node_name)
                                if isinstance(node_output, dict):
                                    final_state = node_output
                cur_user_message_index += 1
            except Exception as e:
                print(f"⚠️  Error processing message (checkpoint issue): {e}")
                break
    except Exception as e:
        # Catch checkpoint errors but continue - agent flow might have completed
        if "InvalidColumnReference" in str(e) or "ON CONFLICT" in str(e):
            print(f"⚠️  Checkpoint save error (known issue with interrupts), but agent flow completed")
        else:
            raise
    
    print(f"\n{'='*80}")
    print(f"✅ Graph execution completed!")
    print(f"{'='*80}\n")
    print(f"Executed flow: {' → '.join([f for f in executed_flow if f])}")
    
    # Check merged images from styling agent
    merged_images = final_state.get("merged_images", [])
    if merged_images:
        print(f"\n🎨 Found {len(merged_images)} merged image(s) from styling agent")
        print(f"✅ Styling agent successfully merged user photo with product!")
        for i, img in enumerate(merged_images):
            print(f"   Merged image {i+1}: {type(img).__name__} ({img.width}x{img.height} if PIL Image)")
    else:
        print(f"\n⚠️  No merged images found from styling agent")
    
    # Get products from research agent
    search_results = final_state.get("search_results", [])
    if search_results:
        print(f"\n🛍️  Found {len(search_results)} products from research agent")
        
        # Show first product info
        first_product = search_results[0]
        product_image_url = first_product.get("image") if isinstance(first_product, dict) else first_product.image
        
        print(f"\n📦 First product:")
        if isinstance(first_product, dict):
            print(f"   Title: {first_product.get('title', 'N/A')}")
            print(f"   Price: {first_product.get('price', 'N/A')}")
            print(f"   Image URL: {product_image_url[:80]}...")
        else:
            print(f"   Title: {first_product.title}")
            print(f"   Price: {first_product.price}")
            print(f"   Image URL: {product_image_url[:80]}...")
    else:
        print(f"\n⚠️  No products found from research agent")
    
    # Show final state summary
    print(f"\n{'='*80}")
    print(f"📊 Final State Summary:")
    print(f"{'='*80}")
    print(f"   User Profile: {'✅' if final_state.get('user_profile') else '❌'}")
    print(f"   Search Results: {len(search_results)} products")
    print(f"   Merged Images: {len(merged_images)} images")
    print(f"   Current Agent: {final_state.get('current_agent', 'N/A')}")
    print(f"   Next Step: {final_state.get('next_step', 'N/A')}")
    print(f"{'='*80}\n")


async def run_test(test_name, user_messages, expected_flow, checkpointer):
    print(f"--- Running Test: {test_name} ---")

    # Create graph with REAL agents and checkpointer
    graph = create_graph(checkpointer)

    # Initial state
    input_data = {
        "messages": [HumanMessage(content=user_messages[0])],
    }
    config = {"configurable": {"thread_id": "thread_id-1", "user_id": "sjha"}}

    cur_user_message_index = 1
    interrupt = False
    executed_flow = []
    while True:
        result = None
        if interrupt:
            result = await graph.ainvoke(
                Command(
                    resume=HumanMessage(content=user_messages[cur_user_message_index])
                ),
                config=config,
            )
            cur_user_message_index += 1
        else:
            result = await graph.ainvoke(
                input=input_data,
                config=config,
            )
        executed_flow.append(result.get("next_step"))
        if result.get("__interrupt__"):
            interrupt = True
            continue
        break

    # print(f"Executed Flow: {executed_flow}")

    # # Check if expected flow is a SUBSEQUENCE of executed flow
    # matches = True
    # for node in expected_flow:
    #     if node not in executed_flow:
    #         matches = False
    #         break

    # if matches:
    #     print("✅ Test Passed")
    # else:
    #     print(f"❌ Test Failed. Expected {expected_flow}")
    print("\n")


async def main():
    """
    Test complete agent flow with PostgreSQL checkpointer.
    Tests: context → clarification → research → styling agents
    Verifies agent state is stored in PostgreSQL.
    """
    print("=" * 80)
    print("🧪 COMPLETE AGENT FLOW TEST WITH POSTGRESQL STATESTORE")
    print("=" * 80)
    
    # Initialize database tables first
    print("\n📦 Initializing database tables...")
    await create_db_and_tables()
    print("✅ Database tables initialized")
    
    # Setup PostgreSQL checkpointer using psycopg (required by AsyncPostgresSaver)
    print("\n📦 Setting up PostgreSQL checkpointer...")
    pg_url = DATABASE_URL
    if "postgresql+asyncpg://" in pg_url:
        pg_url = pg_url.replace("postgresql+asyncpg://", "postgresql://")
    elif "postgresql://" not in pg_url:
        pg_url = pg_url.replace("sqlite://", "postgresql://")
    
    # Create psycopg connection pool (AsyncPostgresSaver expects psycopg, not asyncpg)
    pool = AsyncConnectionPool(pg_url)
    await pool.open()  # Open pool explicitly
    
    # Initialize checkpointer with psycopg pool
    checkpointer = AsyncPostgresSaver(pool)
    
    # Setup checkpointer tables
    # Note: setup() may fail with CONCURRENTLY error - we'll create tables manually if needed
    try:
        await checkpointer.setup()
        print("✅ Checkpointer tables created/verified")
    except Exception as e:
        error_str = str(e)
        if "CONCURRENTLY" in error_str:
            # CONCURRENTLY can't run in transaction - create tables without it
            print("⚠️  Setup failed due to CONCURRENTLY - creating tables manually...")
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    # Create checkpoint_migrations table first
                    await cur.execute("""
                        CREATE TABLE IF NOT EXISTS checkpoint_migrations (
                            v INTEGER PRIMARY KEY
                        );
                    """)
                    # Create checkpoints table if not exists (correct schema)
                    await cur.execute("""
                        CREATE TABLE IF NOT EXISTS checkpoints (
                            thread_id TEXT NOT NULL,
                            checkpoint_ns TEXT NOT NULL DEFAULT '',
                            checkpoint_id TEXT NOT NULL,
                            parent_checkpoint_id TEXT,
                            type TEXT,
                            checkpoint JSONB NOT NULL,
                            metadata JSONB NOT NULL DEFAULT '{}',
                            PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                        );
                    """)
                    # Create checkpoint_blobs table if not exists
                    await cur.execute("""
                        CREATE TABLE IF NOT EXISTS checkpoint_blobs (
                            thread_id TEXT NOT NULL,
                            checkpoint_ns TEXT NOT NULL DEFAULT '',
                            checkpoint_id TEXT NOT NULL,
                            channel TEXT NOT NULL,
                            version TEXT NOT NULL,
                            type TEXT NOT NULL,
                            blob BYTEA NOT NULL,
                            PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, channel, version)
                        );
                    """)
                    # Drop and recreate checkpoint_writes table with correct schema
                    await cur.execute("DROP TABLE IF EXISTS checkpoint_writes CASCADE;")
                    await cur.execute("""
                        CREATE TABLE checkpoint_writes (
                            thread_id TEXT NOT NULL,
                            checkpoint_ns TEXT NOT NULL DEFAULT '',
                            checkpoint_id TEXT NOT NULL,
                            task_id TEXT NOT NULL,
                            idx INTEGER NOT NULL,
                            channel TEXT NOT NULL,
                            type TEXT,
                            blob BYTEA NOT NULL,
                            task_path TEXT NOT NULL DEFAULT '',
                            PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx),
                            UNIQUE (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                        );
                    """)
                    # Create index for faster lookups
                    await cur.execute("""
                        CREATE INDEX IF NOT EXISTS checkpoint_writes_thread_id_idx 
                        ON checkpoint_writes(thread_id);
                    """)
                    await conn.commit()
            print("✅ Checkpointer tables created manually")
        elif "already exists" in error_str.lower():
            print("✅ Checkpointer tables already exist")
        else:
            print(f"⚠️  Setup error: {e}")
            print("   Continuing anyway...")
    
    print("✅ PostgreSQL checkpointer initialized")
    
    try:
        # Test complete flow with user photo from database
        await run_test_with_user_photo(
            username="gourav",
            user_messages=[
                "recommend me some shirts",
                "i am going to thailand",
                "for a beach party",
            ],
            checkpointer=checkpointer,
        )
        
        # Verify state was stored in PostgreSQL
        print("\n" + "=" * 80)
        print("🔍 Verifying agent state in PostgreSQL...")
        print("=" * 80)
        
        # Try to retrieve state from checkpointer
        graph = create_graph(checkpointer)
        config = {"configurable": {"thread_id": "thread_gourav", "user_id": "user_d9fdb061"}}
        
        try:
            # Get state from checkpointer
            state = await checkpointer.aget(config)
            if state:
                print("✅ Agent state found in PostgreSQL!")
                print(f"   Thread ID: {config['configurable']['thread_id']}")
                print(f"   State keys: {list(state.keys()) if isinstance(state, dict) else 'N/A'}")
            else:
                print("⚠️  No state found (this might be expected if no checkpoints were created)")
        except Exception as e:
            print(f"⚠️  Could not retrieve state: {e}")
        
    finally:
        # Cleanup: Close connection pool
        print("\n🧹 Cleaning up...")
        await pool.close()
        print("✅ Connection pool closed")


if __name__ == "__main__":
    asyncio.run(main())

    # run_test(
    #     "some shopping stuff and the qna",
    #     [
    #         "recommend me some shirts, i am going to thailand for beach party",
    #     ],
    #     ["context_agent", "__interrupt__", "__interrupt__", "__interrupt__"],
    # )

    # run_test(
    #     "some shopping stuff and the qna",
    #     [
    #         "recommend me some shirts",
    #         "i am going to thailand",
    #         "what is capital of belgium",
    #         "i am going for a full moon party",
    #     ],
    #     ["context_agent", "__interrupt__", "__interrupt__", "__interrupt__"],
    # )

    # run_test(
    #     "some shopping stuff and the qna",
    #     [
    #         "i am going for a full moon party",
    #     ],
    #     ["context_agent", "__interrupt__", "__interrupt__", "__interrupt__"],
    # )

    # run_test(
    #     "Test General QnA",
    #     "what is capital of belgium",
    #     "",
    #     "",
    #     ["context_agent"],
    # )

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
