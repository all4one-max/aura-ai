"""
Data Access Object for ProductEmbedding operations.
"""

import json
import uuid
from datetime import datetime
from typing import Optional, List
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
import numpy as np

from app.database import engine
from app.schema import ProductEmbedding, ProductWithEmbedding


async def create_product_embedding(
    product: ProductWithEmbedding,
    user_id: str,
    merged_image_s3_key: Optional[str] = None,
    merged_image_url: Optional[str] = None,
) -> ProductEmbedding:
    """
    Create a new ProductEmbedding record in the database.
    
    Args:
        product: ProductWithEmbedding object containing product data and embedding
        user_id: User identifier who this product was shown to
        merged_image_s3_key: S3 key of the merged image
        merged_image_url: S3 URL of the merged image
        
    Returns:
        Created ProductEmbedding object
    """
    async with AsyncSession(engine) as session:
        # Check if product_embedding already exists (by product_id) within this session
        statement = select(ProductEmbedding).where(ProductEmbedding.product_id == product.id)
        result = await session.exec(statement)
        existing = result.first()
        
        # Convert embedding to JSON string
        embedding_json = json.dumps(product.embedding) if isinstance(product.embedding, (list, np.ndarray)) else product.embedding
        
        if existing:
            # Update existing record
            existing.user_id = user_id
            existing.image = product.image
            existing.price = product.price
            existing.link = product.link
            existing.rating = product.rating
            existing.title = product.title
            existing.source = product.source
            existing.reviews = product.reviews
            existing.user_photo_url = product.user_photo_url
            existing.merged_image_url = merged_image_url
            existing.merged_image_s3_key = merged_image_s3_key
            existing.embedding = embedding_json
            existing.updated_at = datetime.utcnow().isoformat()
            session.add(existing)
            await session.commit()
            await session.refresh(existing)
            return existing
        
        # Create new record
        product_embedding = ProductEmbedding(
            product_id=product.id,
            user_id=user_id,
            image=product.image,
            price=product.price,
            link=product.link,
            rating=product.rating,
            title=product.title,
            source=product.source,
            reviews=product.reviews,
            embedding=embedding_json,
            user_photo_url=product.user_photo_url,
            merged_image_url=merged_image_url,
            merged_image_s3_key=merged_image_s3_key,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
        )
        
        session.add(product_embedding)
        await session.commit()
        await session.refresh(product_embedding)
        return product_embedding


async def get_product_embedding_by_id(product_id: str, user_id: Optional[str] = None) -> Optional[ProductEmbedding]:
    """
    Get ProductEmbedding by product_id.
    
    Args:
        product_id: Product identifier
        user_id: Optional user_id to filter by
        
    Returns:
        ProductEmbedding if found, None otherwise
    """
    async with AsyncSession(engine) as session:
        if user_id:
            statement = select(ProductEmbedding).where(
                ProductEmbedding.product_id == product_id,
                ProductEmbedding.user_id == user_id
            )
        else:
            statement = select(ProductEmbedding).where(
                ProductEmbedding.product_id == product_id
            )
        
        result = await session.exec(statement)
        return result.first()


async def get_product_embeddings_by_user(user_id: str, limit: Optional[int] = None) -> List[ProductEmbedding]:
    """
    Get all ProductEmbeddings for a user.
    
    Args:
        user_id: User identifier
        limit: Optional limit on number of results
        
    Returns:
        List of ProductEmbedding objects
    """
    async with AsyncSession(engine) as session:
        # Order by id descending (newest first) since created_at is a string
        statement = select(ProductEmbedding).where(
            ProductEmbedding.user_id == user_id
        ).order_by(ProductEmbedding.id.desc())
        
        if limit:
            statement = statement.limit(limit)
        
        result = await session.exec(statement)
        return list(result.all())


async def get_product_embedding_by_db_id(db_id: int) -> Optional[ProductEmbedding]:
    """
    Get ProductEmbedding by database ID.
    
    Args:
        db_id: Database primary key ID
        
    Returns:
        ProductEmbedding if found, None otherwise
    """
    async with AsyncSession(engine) as session:
        statement = select(ProductEmbedding).where(ProductEmbedding.id == db_id)
        result = await session.exec(statement)
        return result.first()


def product_embedding_to_product_with_embedding(product_embedding: ProductEmbedding) -> ProductWithEmbedding:
    """
    Convert ProductEmbedding SQLModel to ProductWithEmbedding BaseModel.
    
    Args:
        product_embedding: ProductEmbedding object from database
        
    Returns:
        ProductWithEmbedding object
    """
    # Parse embedding from JSON string
    embedding = json.loads(product_embedding.embedding) if isinstance(product_embedding.embedding, str) else product_embedding.embedding
    
    return ProductWithEmbedding(
        id=product_embedding.product_id,
        image=product_embedding.image,
        price=product_embedding.price,
        link=product_embedding.link,
        rating=product_embedding.rating,
        title=product_embedding.title,
        source=product_embedding.source,
        reviews=product_embedding.reviews,
        embedding=embedding,
        user_photo_url=product_embedding.user_photo_url,
        merged_image_url=product_embedding.merged_image_url,
    )

