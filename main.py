"""
FastAPI server for Aura AI chat application.
Provides REST API endpoints for LangGraph-based shopping assistant.
"""

from contextlib import asynccontextmanager

# Removed: AsyncPostgresSaver - we now use only AgentStateTable
# Removed: AsyncConnectionPool - not needed since we use MemorySaver

from fastapi import FastAPI, HTTPException, Query, Form, UploadFile, File, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Optional
import httpx
import json as JSON
import asyncio

from api_models import ChatRequest, ChatResponse
from api_models.user import (
    LoginRequest,
    LoginResponse,
    UploadUrlResponse,
    ImageUrlResponse,
    UpdateRequest,
    UpdateResponse,
    LikeResponse,
    CreateChatRequest,
    CreateChatResponse,
    ChatInfo,
)
from app.graph import create_graph
from app.database import create_db_and_tables
from app.config import DATABASE_URL, AWS_S3_BUCKET
from app.services.s3_service import s3_service
from app.services.user_service import UserService
from app.state import UserProfile
from app.dao.user_dao import create_user, get_user, update_user_profile, user_to_profile
from app.dao.product_embedding_dao import get_product_embedding_by_id
from app.schema import User, UserEmbedding, ProductEmbedding  # Import ProductEmbedding so SQLModel discovers it
from langchain_core.messages import HumanMessage
import uuid
import json
import numpy as np

# Global variables
compiled_graph = None
checkpointer_instance = None
user_service_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    global compiled_graph, checkpointer_instance, user_service_instance
    
    # Startup
    print("🚀 Starting Aura AI server...")
    
    # Create database tables
    await create_db_and_tables()
    print("✅ Database tables created/verified")
    
    # Initialize checkpointer (always use MemorySaver - no PostgreSQL checkpointing)
    from langgraph.checkpoint.memory import MemorySaver
    checkpointer_instance = MemorySaver()
    print("✅ Checkpointer initialized (MemorySaver)")
    
    # Create graph with checkpointer
    compiled_graph = create_graph(checkpointer=checkpointer_instance)
    print("✅ LangGraph compiled and ready")
    
    # Initialize user service
    user_service_instance = UserService()
    print("✅ User service initialized")
    
    print("🎉 Server startup complete!")
    
    yield

    # Shutdown
    print("🛑 Shutting down server...")


app = FastAPI(
    title="Aura AI Chat API",
    description="REST API for Aura AI shopping assistant powered by LangGraph",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Health Check ====================

@app.get("/")
async def root():
    """Root endpoint - health check."""
    return {
        "status": "ok",
        "message": "Aura AI Chat API is running",
        "graph_initialized": compiled_graph is not None,
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "graph_initialized": compiled_graph is not None,
    }


# ==================== User Endpoints ====================

@app.post("/api/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Login endpoint - creates user if doesn't exist, returns user profile."""
    try:
        user = await get_user(username=request.username)
        if not user:
            # Create new user with auto-generated user_id
            user = await create_user(username=request.username)
        
        # Convert to profile
        profile = user_to_profile(user)
        return LoginResponse(
            user_id=user.user_id,
            username=user.username,
            profile=profile,
        )
    except Exception as e:
        print(f"Error in login: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/user/{username}", response_model=LoginResponse)
async def get_user_profile(username: str):
    """Get user profile by username."""
    try:
        user = await get_user(username=username)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        profile = user_to_profile(user)
        return LoginResponse(
            user_id=user.user_id,
            username=user.username,
            profile=profile,
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting user profile: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting user profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/update/{username}", response_model=UpdateResponse)
async def update_user(
    username: str,
    request: Request,
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Update user profile. Accepts both JSON and FormData."""
    try:
        user = await get_user(username=username)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Check content type and parse accordingly
        content_type = request.headers.get("content-type", "")
        
        if "application/json" in content_type:
            # Parse JSON body
            body = await request.json()
            upper_body_size = body.get("upper_body_size")
            lower_body_size = body.get("lower_body_size")
            region = body.get("region")
            gender = body.get("gender")
            age_group = body.get("age_group")
            photo_urls = body.get("photo_urls")
            query_filters = body.get("query_filters")
            liked_items = body.get("liked_items")
        else:
            # Parse FormData
            form_data = await request.form()
            upper_body_size = form_data.get("upper_body_size")
            lower_body_size = form_data.get("lower_body_size")
            region = form_data.get("region")
            gender = form_data.get("gender")
            age_group = form_data.get("age_group")
            photo_urls = form_data.get("photo_urls")
            query_filters = form_data.get("query_filters")
            liked_items = form_data.get("liked_items")
            
            # Convert to strings if they're UploadFile objects (shouldn't happen, but safe)
            if upper_body_size and hasattr(upper_body_size, 'read'):
                upper_body_size = None
            if lower_body_size and hasattr(lower_body_size, 'read'):
                lower_body_size = None
            if region and hasattr(region, 'read'):
                region = None
            if gender and hasattr(gender, 'read'):
                gender = None
            if age_group and hasattr(age_group, 'read'):
                age_group = None
        
        # Parse JSON strings from FormData (handle empty strings)
        parsed_photo_urls = None
        if photo_urls:
            if isinstance(photo_urls, str) and photo_urls.strip():
                try:
                    parsed_photo_urls = JSON.loads(photo_urls)
                except Exception as e:
                    print(f"Warning: Failed to parse photo_urls JSON: {e}")
            elif isinstance(photo_urls, list):
                parsed_photo_urls = photo_urls
        
        parsed_query_filters = None
        if query_filters:
            if isinstance(query_filters, str) and query_filters.strip():
                try:
                    parsed_query_filters = JSON.loads(query_filters)
                except Exception as e:
                    print(f"Warning: Failed to parse query_filters JSON: {e}")
            elif isinstance(query_filters, dict):
                parsed_query_filters = query_filters
        
        parsed_liked_items = None
        if liked_items:
            if isinstance(liked_items, str) and liked_items.strip():
                try:
                    parsed_liked_items = JSON.loads(liked_items)
                except Exception as e:
                    print(f"Warning: Failed to parse liked_items JSON: {e}")
            elif isinstance(liked_items, list):
                parsed_liked_items = liked_items
        
        # Update user profile (only include fields that are provided and not empty)
        profile_updates = {}
        if parsed_photo_urls is not None:
            profile_updates["photo_urls"] = parsed_photo_urls
        if upper_body_size and str(upper_body_size).strip():
            profile_updates["upper_body_size"] = str(upper_body_size).strip()
        if lower_body_size and str(lower_body_size).strip():
            profile_updates["lower_body_size"] = str(lower_body_size).strip()
        if region and str(region).strip():
            profile_updates["region"] = str(region).strip()
        if gender and str(gender).strip():
            profile_updates["gender"] = str(gender).strip()
        if age_group and str(age_group).strip():
            profile_updates["age_group"] = str(age_group).strip()
        if parsed_query_filters is not None:
            profile_updates["query_filters"] = parsed_query_filters
        if parsed_liked_items is not None:
            profile_updates["liked_items"] = parsed_liked_items
        
        # If no updates provided, return error
        if not profile_updates:
            raise HTTPException(status_code=400, detail="No valid fields provided for update")
        
        updated_user = await update_user_profile(
            user_id=user.user_id,
            profile_updates=profile_updates,
        )
        
        # TODO: Update embeddings in background if photo_urls changed (when method is implemented)
        # if request.photo_urls and user.photo_urls != request.photo_urls:
        #     background_tasks.add_task(
        #         user_service_instance.update_user_embeddings,
        #         updated_user.user_id,
        #     )
        
        profile = user_to_profile(updated_user)
        return UpdateResponse(
            user_id=updated_user.user_id,
            username=updated_user.username,
            profile=profile,
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error updating user: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ==================== S3 Upload Endpoints ====================

@app.post("/api/upload-url", response_model=UploadUrlResponse)
async def get_upload_url(
    username: str = Form(...),
    file_name: str = Form(...),
    file_type: str = Form(...),
):
    """Generate presigned URL for S3 upload."""
    try:
        result = s3_service.generate_upload_url(username, file_name, file_type)
        return UploadUrlResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/image-url", response_model=ImageUrlResponse)
async def get_image_url(username: str = Form(...), s3_key: str = Form(...)):
    """Generate presigned URL for S3 image access."""
    try:
        result = s3_service.generate_image_url(username, s3_key)
        return ImageUrlResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/image/{s3_key:path}")
async def get_image(s3_key: str):
    """Get image from S3 by s3_key."""
    try:
        # Use boto3 to get object directly (more reliable than presigned URLs)
        # This uses IAM credentials directly and doesn't require presigned URLs
        image_content = s3_service.get_object(s3_key)
        content_type = s3_service.get_object_content_type(s3_key)
        
        from fastapi.responses import Response
        return Response(content=image_content, media_type=content_type)
    except Exception as e:
        print(f"Error getting image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/image-proxy/{username}")
async def get_image_proxy(
    username: str,
    s3_key: str = Query(..., description="S3 key of the image"),
    image_url: Optional[str] = Query(None, description="Original presigned URL (optional, for logging)")
):
    """
    Proxy endpoint for S3 images to avoid CORS and expiration issues.
    Frontend calls this with s3_key extracted from presigned URL.
    Uses IAM credentials directly (no expiration).
    """
    try:
        if not s3_key:
            raise HTTPException(status_code=400, detail="s3_key is required")
        
        print(f"📋 Image proxy request for {username}: s3_key={s3_key}")
        
        # Use S3 service to get image directly (no expiration issues)
        image_content = s3_service.get_object(s3_key)
        content_type = s3_service.get_object_content_type(s3_key)
        
        from fastapi.responses import Response
        return Response(content=image_content, media_type=content_type)
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error proxying image for {username} (s3_key: {s3_key}): {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error proxying image: {str(e)}")


async def _generate_and_store_user_embeddings(
    user_id: str, photo_urls: list, user_embedding_service
):
    """
    Background task to generate and store user embeddings from photos.
    """
    try:
        print(f"🔄 Generating embeddings for user {user_id}...")
        user_embeddings = await user_embedding_service.update_user_embeddings_from_photos(
            user_id=user_id,
            photo_urls=photo_urls,
        )
        
        # Store embeddings in user profile
        await update_user_profile(
            user_id=user_id,
            profile_updates={"user_embeddings": user_embeddings.model_dump()},
        )
        print(f"✅ Successfully generated and stored embeddings for user {user_id}")
    except Exception as e:
        print(f"❌ Error generating embeddings for user {user_id}: {e}")
        import traceback
        traceback.print_exc()


@app.post("/api/upload")
async def upload_file(
    username: str = Form(...),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Upload file to S3 and update user profile."""
    try:
        # Step 1: Get presigned URL
        result = s3_service.generate_upload_url(
            username=username,
            file_name=file.filename or "image.jpg",
            file_type=file.content_type or "image/jpeg"
        )
        
        upload_url = result["upload_url"]
        s3_key = result["s3_key"]
        
        # Step 2: Upload file to S3 using presigned URL
        file_content = await file.read()
        async with httpx.AsyncClient() as client:
            response = await client.put(
                upload_url,
                content=file_content,
                headers={"Content-Type": file.content_type or "image/jpeg"},
            )
            response.raise_for_status()
        
        # Step 3: Update user profile with new photo URL
        user = await get_user(username)
        if user:
            # Ensure photo_urls is a list (might be stored as JSON string)
            current_photos = user.photo_urls or []
            if isinstance(current_photos, str):
                import json
                try:
                    current_photos = JSON.loads(current_photos)
                except:
                    current_photos = []
            if not isinstance(current_photos, list):
                current_photos = []
            
            # Generate image URL using generate_image_url method
            image_url_result = s3_service.generate_image_url(username, s3_key)
            new_photo_url = image_url_result["image_url"]
            updated_photos = list(set(current_photos + [new_photo_url]))
            
            updated_user_record = await update_user_profile(
                user_id=user.user_id,
                profile_updates={"photo_urls": updated_photos},
            )
            
            # Generate user embeddings from photos in background
            from app.services.user_embedding_service import UserEmbeddingService
            user_embedding_service = UserEmbeddingService()
            
            background_tasks.add_task(
                _generate_and_store_user_embeddings,
                user.user_id,
                updated_photos,
                user_embedding_service,
            )
            
            # Return updated user object (frontend expects this format)
            from app.dao.user_dao import user_to_profile
            profile = user_to_profile(updated_user_record)
            return {
                "user_id": updated_user_record.user_id,
                "username": updated_user_record.username,
                "profile": profile,
            }
        else:
            # Generate image URL even if user doesn't exist yet
            image_url_result = s3_service.generate_image_url(username, s3_key)
            new_photo_url = image_url_result["image_url"]
            
            # Return upload info if user doesn't exist
            return {
                "s3_key": s3_key,
                "image_url": new_photo_url,
                "message": "File uploaded successfully. Please create a user profile first.",
            }
    except Exception as e:
        print(f"Error uploading file: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/image/{username}/{s3_key:path}")
async def delete_image(username: str, s3_key: str):
    """Delete image from S3 and user profile."""
    try:
        # Delete from S3
        s3_service.delete_object(s3_key)
        
        # Remove from user profile photo_urls
        existing_photos = []
        user = await get_user(username)
        if user and user.photo_urls:
            # Ensure photo_urls is a list (might be stored as JSON string)
            photo_urls = user.photo_urls
            if isinstance(photo_urls, str):
                import json
                try:
                    photo_urls = JSON.loads(photo_urls)
                except:
                    photo_urls = []
            if not isinstance(photo_urls, list):
                photo_urls = []
            
            # Extract s3_key from each photo URL and compare with the s3_key to delete
            # Presigned URLs have different query parameters, so we compare s3_keys, not full URLs
            import re
            def extract_s3_key_from_url(url):
                """Extract s3_key from a presigned URL."""
                # Pattern: https://s3.../users/{username}/profile/{filename}
                match = re.search(r'/users/([^/]+)/profile/([^?]+)', url)
                if match:
                    username_part, filename = match.groups()
                    return f"users/{username_part}/profile/{filename}"
                return None
            
            # Filter out photos that match the s3_key to delete
            existing_photos = []
            for photo_url in photo_urls:
                photo_s3_key = extract_s3_key_from_url(photo_url)
                if photo_s3_key != s3_key:
                    existing_photos.append(photo_url)
        
        if user:
            updated_user_record = await update_user_profile(
                user_id=user.user_id,
                profile_updates={"photo_urls": existing_photos},
            )
            
            # Return updated photo_urls (frontend expects this)
            from app.dao.user_dao import user_to_profile
            profile = user_to_profile(updated_user_record)
            return {
                "message": "Image deleted successfully",
                "photo_urls": profile.get("photo_urls", []),
            }
        
        return {
            "message": "Image deleted successfully",
            "photo_urls": [],
        }
    except Exception as e:
        print(f"Error deleting image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Like Endpoint ====================

@app.post("/api/like/{username}/{image_id}", response_model=LikeResponse)
async def like_product(username: str, image_id: str):
    """Like a product - updates user embeddings based on liked product."""
    try:
        # Get user
        user = await get_user(username)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get product embedding
        product_embedding = await get_product_embedding_by_id(image_id)
        if not product_embedding:
            raise HTTPException(status_code=404, detail="Product not found")
        
        # TODO: Update user embeddings (70% existing, 30% liked product) when method is implemented
        # user_service_instance.update_user_embeddings_from_like(
        #     user_id=user.user_id,
        #     product_embedding=product_embedding.embedding,
        # )
        
        # Update liked_items
        liked_items = user.liked_items or []
        if image_id not in liked_items:
            liked_items.append(image_id)
            await update_user_profile(
                user_id=user.user_id,
                profile_updates={"liked_items": liked_items},
            )
        
        return LikeResponse(
            message="Product liked successfully",
            user_id=user.user_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error liking product: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Chat Endpoints ====================

@app.post("/api/createChat/", response_model=CreateChatResponse)
async def create_chat(request: CreateChatRequest):
    """Create a new chat room and initialize agent_state."""
    try:
        from app.dao.user_chat_dao import create_user_chat
        from app.dao.agent_state_dao import sync_agent_state_from_checkpoint
        
        chat_room_id = f"chat_{uuid.uuid4().hex[:8]}"
        # Use user_id from request (sent by frontend)
        user_id = request.user_id
        
        # Extract username from user_id if needed (for backward compatibility)
        # user_id format: "user_username" or just the user_id
        username = user_id.replace("user_", "") if user_id.startswith("user_") else user_id
        
        # Create UserChat entry
        user_chat = await create_user_chat(
            username=username,
            chat_room_id=chat_room_id,
            user_id=user_id,
        )
        
        # Create initial agent_state for this chat room
        # This ensures agent_state exists even before any messages are sent
        try:
            await sync_agent_state_from_checkpoint(
                thread_id=chat_room_id,
                user_id=user_id,
                state={
                    "thread_id": chat_room_id,
                    "user_id": user_id,
                    "messages": [],  # Empty initially, will be populated when user sends messages
                    "user_profile": None,
                    "search_results": [],
                    "chat_query_json": None,
                    "styled_products": None,
                    "ranked_products": None,
                    "merged_images": None,
                    "current_agent": None,
                    "user_intent": None,
                    "next_step": None,
                },
                request_id=None,
            )
            print(f"✅ Created agent_state for chat_room_id: {chat_room_id}")
        except Exception as agent_state_error:
            print(f"⚠️  Warning: Failed to create agent_state: {agent_state_error}")
            # Continue even if agent_state creation fails - it will be created on first message
        
        return CreateChatResponse(
            id=user_chat.chat_room_id,
            chat_id=user_chat.chat_room_id,
            username=user_chat.username,
            session_name=None,
            messages=[],
        )
    except Exception as e:
        print(f"Error creating chat: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chats/{user_id}", response_model=list[ChatInfo])
async def get_chats(user_id: str):
    """Get all chats for a user by user_id."""
    try:
        from app.dao.user_chat_dao import get_user_chats_by_user_id
        from app.dao.agent_state_dao import get_agent_state
        
        user_chats = await get_user_chats_by_user_id(user_id)
        
        chats = []
        for user_chat in user_chats:
            # Get agent state for this chat (with timeout handling)
            agent_state = None
            messages = []
            try:
                agent_state = await get_agent_state(user_chat.chat_room_id)
            except Exception as e:
                print(f"⚠️  Error loading agent state for chat {user_chat.chat_room_id}: {e}")
                # Continue with empty messages - don't fail the entire request
                agent_state = None
            
            # Get messages from individual column (not from state JSON)
            messages = []
            if agent_state and agent_state.messages:
                # Debug: Check what's in agent_state.messages
                print(f"🔍 Debug: agent_state.messages type: {type(agent_state.messages)}")
                print(f"🔍 Debug: agent_state.messages value: {agent_state.messages}")
                
                state_messages = agent_state.messages if isinstance(agent_state.messages, list) else []
                
                # If messages is a dict (JSONB), try to parse it
                if isinstance(agent_state.messages, dict):
                    # It might be stored as a dict with keys, try to extract list
                    if "messages" in agent_state.messages:
                        state_messages = agent_state.messages["messages"]
                    elif isinstance(agent_state.messages, list):
                        state_messages = agent_state.messages
                    else:
                        # Try to convert dict values to list
                        state_messages = list(agent_state.messages.values()) if agent_state.messages else []
                
                print(f"🔍 Debug: state_messages count: {len(state_messages)}")
                
                for msg in state_messages:
                    if isinstance(msg, dict):
                        # Handle serialized LangChain messages
                        msg_type = msg.get("type", "").lower()
                        content = msg.get("content", "")
                        
                        # Determine role
                        role = "assistant"
                        if "human" in msg_type or "user" in msg_type:
                            role = "user"
                        elif "ai" in msg_type or "assistant" in msg_type:
                            role = "assistant"
                        
                        # Extract ranked_products from additional_kwargs if present
                        msg_dict = {
                            "role": role,
                            "content": content,
                        }
                        
                        # Include ranked_products from additional_kwargs
                        additional_kwargs = msg.get("additional_kwargs", {})
                        if isinstance(additional_kwargs, dict):
                            ranked_products = additional_kwargs.get("ranked_products")
                            if ranked_products:
                                msg_dict["ranked_products"] = ranked_products
                        
                        messages.append(msg_dict)
                    else:
                        # Handle LangChain message objects (if not serialized)
                        class_name = str(type(msg)).lower()
                        content = str(msg) if hasattr(msg, "__str__") else repr(msg)
                        
                        role = "assistant"
                        if "humanmessage" in class_name:
                            role = "user"
                        
                        messages.append({
                            "role": role,
                            "content": content,
                        })
            
            chats.append(ChatInfo(
                id=user_chat.chat_room_id,
                chat_id=user_chat.chat_room_id,
                username=user_chat.username,
                session_name=None,  # Can be added to UserChat table later if needed
                messages=messages,
            ))
        
        return chats
    except Exception as e:
        print(f"Error fetching chats: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching chats: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint - returns immediate response for every user message.
    
    Flow:
    1. User sends message (e.g., "i want a tshirt")
    2. Context agent checks for required fields
    3. If clarification needed → returns immediately with question
    4. If all fields present → continues processing and returns final result
    """
    if compiled_graph is None:
        raise HTTPException(
            status_code=503,
            detail="Graph not initialized. Server may still be starting up.",
        )

    try:
        # Validate user has photos before allowing chat
        user_id = request.user_id
        
        # Try to get user by user_id first, then fallback to username
        user = await get_user(user_id=user_id)
        
        # If not found by user_id, try extracting username (for backward compatibility)
        if not user:
            username = user_id.replace("user_", "") if user_id.startswith("user_") else user_id
            user = await get_user(username=username)
        
        if not user:
            raise HTTPException(
                status_code=404,
                detail="User not found. Please login again."
            )
        
        # Update user_id to match the actual user_id from database
        user_id = user.user_id
        username = user.username
        
        # Check if user has photos (frontend already validates, but backend check for security)
        photo_urls = []
        if user.photo_urls:
            if isinstance(user.photo_urls, str):
                try:
                    photo_urls = JSON.loads(user.photo_urls)
                except Exception as e:
                    print(f"⚠️  Warning: Failed to parse photo_urls JSON for user {user_id}: {e}")
                    photo_urls = []
            elif isinstance(user.photo_urls, list):
                photo_urls = user.photo_urls
        
        # Log for debugging
        print(f"🔍 User {user_id} (username: {username}) photo check: photo_urls type={type(user.photo_urls)}, count={len(photo_urls) if photo_urls else 0}")
        
        if not photo_urls or len(photo_urls) == 0:
            # Since frontend already validates, this shouldn't happen, but log it
            print(f"⚠️  Warning: User {user_id} reached chat endpoint without photos. Frontend validation may have failed.")
            raise HTTPException(
                status_code=403,
                detail="Please upload at least one photo before starting a chat. Go to your dashboard to upload photos."
            )
        
        # Check if user embeddings exist (they should be generated after photo upload)
        # If not, generate them synchronously (or return error asking to wait)
        if not user.user_embeddings:
            # Embeddings might still be generating in background
            # For now, we'll allow chat but ranking won't work optimally
            print(f"⚠️  User {user_id} has photos but no embeddings yet. They may still be generating.")
        
        # Generate thread_id if not provided
        thread_id = request.thread_id or f"thread_{uuid.uuid4().hex[:8]}"
        
        # Create/update UserChat entry when chat is accessed
        from app.dao.user_chat_dao import create_user_chat
        await create_user_chat(
            username=username,
            chat_room_id=thread_id,
            user_id=user_id,
        )
        
        # Fetch or create agent_state for this thread_id and user_id
        # This ensures agent_state exists for storing/retrieving messages
        from app.dao.agent_state_dao import get_agent_state, sync_agent_state_from_checkpoint
        existing_agent_state = await get_agent_state(thread_id)
        if not existing_agent_state:
            # Create initial agent_state if it doesn't exist
            print(f"📝 Creating agent_state for thread_id: {thread_id}, user_id: {user_id}")
            await sync_agent_state_from_checkpoint(
                thread_id=thread_id,
                user_id=user_id,
                state={
                    "thread_id": thread_id,
                    "user_id": user_id,
                    "messages": [],  # Empty initially, will be populated after response
                    "user_profile": None,
                    "search_results": [],
                    "chat_query_json": None,
                    "styled_products": None,
                    "ranked_products": None,
                    "merged_images": None,
                    "current_agent": None,
                    "user_intent": None,
                    "next_step": None,
                },
                request_id=None,
            )
        else:
            print(f"✅ Found existing agent_state for thread_id: {thread_id}, user_id: {user_id}")
        
        # Generate unique request_id for this request
        request_id = f"req_{uuid.uuid4().hex[:8]}"

        # Load user profile from User table using user_id
        user_record = await get_user(user_id=user_id)
        
        # Convert User to UserProfile
        user_profile = None
        if user_record:
            user_profile = user_to_profile(user_record)
            print(f"✅ Loaded user profile for user_id {user_id} with {len(user_profile.get('photo_urls', []))} photos")
        else:
            print(f"⚠️  Warning: User with user_id {user_id} not found in database")
            # Create minimal user_profile even if user doesn't exist
            user_profile = {
                "user_id": user_id,
                "photo_urls": [],
            }
        
        # Load existing agent state to get conversation history
        from app.dao.agent_state_dao import get_agent_state
        from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
        existing_state_record = await get_agent_state(thread_id)
        
        # Get existing messages from previous conversations
        existing_messages = []
        if existing_state_record and existing_state_record.messages:
            # Deserialize messages from JSONB and convert back to LangChain message objects
            messages_data = existing_state_record.messages
            if isinstance(messages_data, list):
                for msg_data in messages_data:
                    if isinstance(msg_data, dict):
                        # Convert dict back to LangChain message object
                        msg_type = msg_data.get("type", "").lower()
                        content = msg_data.get("content", "")
                        
                        if "human" in msg_type or "user" in msg_type:
                            existing_messages.append(HumanMessage(content=content))
                        elif "ai" in msg_type or "assistant" in msg_type:
                            existing_messages.append(AIMessage(content=content))
                        else:
                            # Default to AIMessage if type is unclear
                            existing_messages.append(AIMessage(content=content))
                    elif isinstance(msg_data, BaseMessage):
                        # Already a LangChain message object
                        existing_messages.append(msg_data)
            elif isinstance(messages_data, str):
                import json
                try:
                    messages_list = JSON.loads(messages_data)
                    for msg_data in messages_list:
                        if isinstance(msg_data, dict):
                            msg_type = msg_data.get("type", "").lower()
                            content = msg_data.get("content", "")
                            if "human" in msg_type or "user" in msg_type:
                                existing_messages.append(HumanMessage(content=content))
                            else:
                                existing_messages.append(AIMessage(content=content))
                except:
                    existing_messages = []
        
        # User profile already loaded above (lines 796-810), skip duplicate loading
        # user_profile is already set from the first load above
        
        # Add new user message to the conversation history
        # This will be stored in agent state messages via LangGraph's add_messages reducer
        new_user_message = HumanMessage(content=request.message)
        all_messages = existing_messages + [new_user_message]
        print(f"💬 User message added: {request.message}")
        print(f"📝 Total messages before graph: {len(all_messages)}")
        
        # Prepare input for graph with full conversation history and user profile
        # The messages will be automatically added to state via add_messages reducer
        input_data = {
            "messages": all_messages,  # User message stored here
            "request_id": request_id,
        }
        
        # Add user_profile to input if available
        if user_profile:
            input_data["user_profile"] = user_profile

        # Configuration with user and thread IDs
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": request.user_id,
            }
        }

        # Stream graph execution to get immediate responses
        # LangGraph returns partial state updates keyed by agent name
        # e.g., {'context_agent': {'current_agent': 'context_agent', 'next_step': 'clarification_agent', ...}}
        from app.utils.agent_state_sync import sync_state_after_agent
        
        # Initialize merged_state with user_profile and other initial data to ensure they persist
        # Ensure username is always stored in user_profile for later retrieval
        if user_profile and not user_profile.get("username"):
            user_profile["username"] = username
        
        merged_state = {
            "messages": all_messages,  # Initialize with user message - LangGraph will add AI responses via add_messages reducer
            "user_profile": user_profile,  # Ensure user_profile persists through all agent updates
            "thread_id": thread_id,
            "request_id": request_id,
        }
        final_result = None
        
        async for state_update in compiled_graph.astream(input_data, config):
            # Extract the actual state from the agent's key
            # LangGraph returns: {'agent_name': {actual_state_dict}}
            agent_state = None
            agent_name = None
            for key, value in state_update.items():
                if isinstance(value, dict):
                    agent_state = value
                    agent_name = key
                    break
            
            if not agent_state:
                # Fallback: use state_update directly if structure is different
                agent_state = state_update if isinstance(state_update, dict) else {}
                agent_name = list(state_update.keys())[0] if state_update else None
            
            # Merge with previous state to maintain full context
            # Preserve user_profile and messages if they exist in merged_state (don't overwrite with None/empty)
            existing_user_profile = merged_state.get("user_profile")
            existing_messages = merged_state.get("messages", [])
            
            merged_state.update(agent_state)
            
            # Restore user_profile if agent update removed it
            if existing_user_profile and not merged_state.get("user_profile"):
                merged_state["user_profile"] = existing_user_profile
            
            # Preserve messages - LangGraph's add_messages reducer handles merging
            # The state_update from LangGraph should contain the full accumulated messages
            # But if messages are missing or empty, preserve existing ones
            current_messages = merged_state.get("messages")
            if (current_messages is None or (isinstance(current_messages, list) and len(current_messages) == 0)) and existing_messages:
                merged_state["messages"] = existing_messages
                print(f"⚠️  Restored {len(existing_messages)} messages that were lost during merge")
            
            # Sync state after each agent step
            # NOTE: Messages are NOT stored here - they are only stored in main function after getting response
            # This sync only stores other agent fields (search_results, styled_products, chat_query_json, etc.)
            try:
                # Create state without messages for agent sync (messages stored separately at end)
                state_without_messages = {k: v for k, v in merged_state.items() if k != "messages"}
                
                await sync_state_after_agent(
                    state=state_without_messages,  # Don't store messages during agent execution
                    thread_id=thread_id,
                    user_id=request.user_id,
                    request_id=request_id,
                )
                
                # Log what was stored
                if agent_name == "context_agent":
                    print(f"✅ Context agent state synced (chat_query_json stored)")
                elif agent_name == "research_agent":
                    print(f"✅ Research agent state synced (search_results stored)")
                elif agent_name == "styling_agent":
                    print(f"✅ Styling agent state synced (styled_products + merged_images stored)")
                elif agent_name == "ranking_agent":
                    print(f"✅ Ranking agent state synced (ranked_products stored)")
            except Exception as sync_error:
                print(f"⚠️  Warning: Failed to sync agent state: {sync_error}")
                # Continue even if sync fails
            
            # Check if clarification agent ran or END state
            current_agent = merged_state.get("current_agent") or agent_name
            next_step = merged_state.get("next_step")
            messages = merged_state.get("messages", [])
            
            # If clarification agent completed or END, return immediately
            if (next_step == "clarification_agent" or next_step == "END" or 
                current_agent == "clarification_agent" or agent_name == "clarification_agent"):
                if messages:
                    last_message = messages[-1]
                    response_text = (
                        last_message.content
                        if hasattr(last_message, "content")
                        else str(last_message)
                    )

                    # Store messages in agent_state AFTER getting response and BEFORE returning
                    # This is the ONLY place where messages are stored for clarification responses
                    try:
                        from langchain_core.messages import HumanMessage, AIMessage
                        
                        # Ensure we have both user message and AI response
                        # Check if user message is already in messages
                        user_message_found = False
                        for msg in messages:
                            if hasattr(msg, "content") and request.message in str(msg.content):
                                user_message_found = True
                                break
                        
                        # If user message is missing, prepend it
                        if not user_message_found:
                            print(f"⚠️  User message not in merged_state.messages, adding it")
                            messages = [HumanMessage(content=request.message)] + messages
                        
                        # Ensure AI response is in messages (should be the last one)
                        ai_response_found = False
                        if messages:
                            last_msg = messages[-1]
                            if hasattr(last_msg, "content") and response_text in str(last_msg.content):
                                ai_response_found = True
                        
                        # If AI response is missing, append it
                        if not ai_response_found:
                            print(f"⚠️  AI response not in merged_state.messages, adding it")
                            messages.append(AIMessage(content=response_text))
                        
                        print(f"🔍 Storing {len(messages)} messages:")
                        for i, msg in enumerate(messages):
                            content = msg.content if hasattr(msg, "content") else str(msg)
                            msg_type = type(msg).__name__
                            print(f"   {i+1}. [{msg_type}] {content[:80]}...")
                        
                        from app.dao.agent_state_dao import sync_agent_state_from_checkpoint
                        await sync_agent_state_from_checkpoint(
                            thread_id=thread_id,
                            user_id=request.user_id,
                            state={**merged_state, "messages": messages},  # Store messages here
                            request_id=request_id,
                        )
                        print(f"✅ Messages stored in agent_state: {len(messages)} messages")
                        print(f"   ✅ User query: {request.message}")
                        print(f"   ✅ AI response: {response_text[:80]}...")
                    except Exception as sync_error:
                        print(f"⚠️  Warning: Failed to store messages: {sync_error}")
                        import traceback
                        traceback.print_exc()

                    return ChatResponse(
                        response=response_text,
                        thread_id=thread_id,
                        user_id=request.user_id,
                        request_id=request_id,
                        merged_images=None,
                        styled_products=None,
                    )
            
            # Keep track of final state
            final_result = merged_state
        
        # If we get here, processing completed (research → styling → ranking)
        # Extract final response
        if final_result:
            messages = final_result.get("messages", [])
            if messages:
                last_message = messages[-1]
                response_text = (
                    last_message.content
                    if hasattr(last_message, "content")
                    else str(last_message)
                )
                
                # Store messages in agent_state AFTER getting response and BEFORE returning
                # This is the ONLY place where messages are stored for final responses
                # Other fields (search_results, styled_products, etc.) are stored during agent execution
                try:
                    from langchain_core.messages import HumanMessage, AIMessage
                    
                    # Get all messages from final_result (should include user message + all AI responses)
                    final_messages = final_result.get("messages", [])
                    
                    # Verify products are in state
                    ranked_products = final_result.get("ranked_products", [])
                    styled_products = final_result.get("styled_products", [])
                    products_count = len(ranked_products) if ranked_products else len(styled_products)
                    
                    # Ensure we have both user message and final AI response
                    # Check if user message is in the list
                    user_message_found = False
                    for msg in final_messages:
                        if isinstance(msg, HumanMessage) or (hasattr(msg, "content") and request.message in str(msg.content)):
                            user_message_found = True
                            break
                    
                    # If user message is missing, prepend it
                    if not user_message_found:
                        print(f"⚠️  User message not in final_messages, adding it")
                        final_messages = [HumanMessage(content=request.message)] + final_messages
                    
                    # Ensure final AI response (from ranking agent) is included
                    # The last message should be from ranking agent (or styling if ranking didn't run)
                    ai_response_found = False
                    last_msg_with_products = None
                    if final_messages:
                        last_msg = final_messages[-1]
                        if isinstance(last_msg, AIMessage) or (hasattr(last_msg, "content") and response_text in str(last_msg.content)):
                            ai_response_found = True
                            last_msg_with_products = last_msg
                    
                    # Get ranked products (prioritize ranked_products over styled_products)
                    ranked_products_for_msg = final_result.get("ranked_products", [])
                    styled_products_for_msg = final_result.get("styled_products", [])
                    products_for_msg = ranked_products_for_msg if ranked_products_for_msg else styled_products_for_msg
                    
                    # If final AI response is missing, append it with products
                    # If it exists, update it to include products in additional_kwargs
                    if not ai_response_found:
                        print(f"⚠️  Final AI response not in final_messages, adding it with products")
                        # Create message with ranked_products data
                        from langchain_core.messages import AIMessage
                        import numpy as np
                        
                        # Create message with products data
                        message_content = response_text
                        if products_for_msg:
                            # Add products to message as additional_kwargs
                            products_data = []
                            for product in products_for_msg:
                                embedding = product.embedding
                                if isinstance(embedding, np.ndarray):
                                    embedding = embedding.tolist()
                                elif hasattr(embedding, 'tolist'):
                                    embedding = embedding.tolist()
                                
                                products_data.append({
                                    "id": product.id,
                                    "image": product.image,
                                    "price": product.price,
                                    "link": product.link,
                                    "rating": product.rating,
                                    "title": product.title,
                                    "source": product.source,
                                    "reviews": product.reviews,
                                    "merged_image_url": product.merged_image_url,
                                    "user_photo_url": product.user_photo_url,
                                    "embedding": embedding,
                                })
                            
                            # Create AIMessage with ranked_products in additional_kwargs
                            final_messages.append(AIMessage(
                                content=message_content,
                                additional_kwargs={"ranked_products": products_data}
                            ))
                            print(f"   ✅ Added message with {len(products_data)} ranked products")
                        else:
                            final_messages.append(AIMessage(content=message_content))
                    elif last_msg_with_products and products_for_msg:
                        # Update existing message to include products if not already present
                        if not hasattr(last_msg_with_products, "additional_kwargs") or "ranked_products" not in getattr(last_msg_with_products, "additional_kwargs", {}):
                            print(f"⚠️  Updating existing message to include ranked_products")
                            import numpy as np
                            
                            products_data = []
                            for product in products_for_msg:
                                embedding = product.embedding
                                if isinstance(embedding, np.ndarray):
                                    embedding = embedding.tolist()
                                elif hasattr(embedding, 'tolist'):
                                    embedding = embedding.tolist()
                                
                                products_data.append({
                                    "id": product.id,
                                    "image": product.image,
                                    "price": product.price,
                                    "link": product.link,
                                    "rating": product.rating,
                                    "title": product.title,
                                    "source": product.source,
                                    "reviews": product.reviews,
                                    "merged_image_url": product.merged_image_url,
                                    "user_photo_url": product.user_photo_url,
                                    "embedding": embedding,
                                })
                            
                            # Update the last message's additional_kwargs
                            if not hasattr(last_msg_with_products, "additional_kwargs"):
                                last_msg_with_products.additional_kwargs = {}
                            last_msg_with_products.additional_kwargs["ranked_products"] = products_data
                            print(f"   ✅ Updated message with {len(products_data)} ranked products")
                    
                    # Log what we're storing
                    print(f"🔍 Storing {len(final_messages)} messages:")
                    for i, msg in enumerate(final_messages, 1):
                        content = msg.content if hasattr(msg, "content") else str(msg)
                        msg_type = type(msg).__name__
                        print(f"   {i}. [{msg_type}] {content[:80]}...")
                    
                    # Store messages - this is the ONLY place messages are stored for final responses
                    from app.dao.agent_state_dao import sync_agent_state_from_checkpoint
                    await sync_agent_state_from_checkpoint(
                        thread_id=thread_id,
                        user_id=request.user_id,
                        state={**final_result, "messages": final_messages},  # Ensure messages are included
                        request_id=request_id,
                    )
                    print(f"✅ Messages stored in agent_state: {len(final_messages)} messages")
                    print(f"   ✅ User query stored: {request.message}")
                    print(f"   ✅ Final AI response stored: {response_text[:100]}...")
                    print(f"   ✅ Products stored: {products_count} products")
                except Exception as sync_error:
                    print(f"⚠️  Warning: Failed to store messages in agent_state: {sync_error}")
                    import traceback
                    traceback.print_exc()
                
                # Get ranked products (sorted) - prioritize ranked_products over styled_products
                ranked_products = final_result.get("ranked_products", [])
                styled_products = final_result.get("styled_products", [])
                
                # Always use ranked products if available, otherwise fallback to styled products
                products_to_return = ranked_products if ranked_products else styled_products
                
                print(f"📦 Products to return: {len(products_to_return)} ({'ranked' if ranked_products else 'styled'})")
                
                products_dict = []
                if products_to_return:
                    import numpy as np
                    for product in products_to_return:
                        # Convert embedding to list if it's a numpy array
                        embedding = product.embedding
                        if isinstance(embedding, np.ndarray):
                            embedding = embedding.tolist()
                        elif hasattr(embedding, 'tolist'):
                            embedding = embedding.tolist()
                        
                        product_dict = {
                            "id": product.id,
                            "image": product.image,
                            "price": product.price,
                            "link": product.link,
                            "rating": product.rating,
                            "title": product.title,
                            "source": product.source,
                            "reviews": product.reviews,
                            "merged_image_url": product.merged_image_url,
                            "user_photo_url": product.user_photo_url,
                            "embedding": embedding,  # Include embedding in response
                        }
                        products_dict.append(product_dict)
                
                merged_images = final_result.get("merged_images", [])
                # Ensure merged_images is a list, not a dict
                if isinstance(merged_images, dict):
                    merged_images = list(merged_images.values()) if merged_images else []
                elif not isinstance(merged_images, list):
                    merged_images = []
                
                # Return ranked_products (prioritized) and styled_products (for backward compatibility)
                ranked_products_dict = products_dict if ranked_products else None
                styled_products_dict = products_dict if not ranked_products and styled_products else None
                
                return ChatResponse(
                    response=response_text,
                    thread_id=thread_id,
                    user_id=request.user_id,
                    request_id=request_id,
                    merged_images=merged_images if merged_images else None,
                    ranked_products=ranked_products_dict,  # Prioritized: ranked products with embeddings
                    styled_products=styled_products_dict,  # Fallback: styled products (for backward compatibility)
                )
        
        # Fallback: return empty response if no messages
        return ChatResponse(
            response="I'm processing your request. Please wait...",
            thread_id=thread_id,
            user_id=request.user_id,
            request_id=request_id,
            merged_images=None,
            styled_products=None,
        )

    except Exception as e:
        print(f"Error processing chat request: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    # Run server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
    )
