"""
AWS S3 service for handling file uploads and presigned URLs.
"""

import uuid
from datetime import timedelta
from typing import Optional
import boto3
from botocore.exceptions import ClientError

from app.config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION,
    AWS_S3_BUCKET,
    S3_PRESIGNED_URL_EXPIRY,
)


class S3Service:
    """Service for AWS S3 operations."""

    def __init__(self):
        """Initialize S3 client."""
        # Use regional endpoint URL to avoid redirects that invalidate signatures
        endpoint_url = f"https://s3.{AWS_REGION}.amazonaws.com" if AWS_REGION else None
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
            endpoint_url=endpoint_url,
        )
        self.bucket = AWS_S3_BUCKET
        self.expiry = S3_PRESIGNED_URL_EXPIRY

    def generate_upload_url(
        self, username: str, file_name: str, file_type: str
    ) -> dict:
        """
        Generate presigned URL for uploading a file to S3.

        Args:
            username: Username
            file_name: Original file name
            file_type: MIME type (e.g., image/jpeg)

        Returns:
            dict with upload_url, image_url, s3_key, expires_in
        """
        # Generate unique file name
        file_extension = file_name.split(".")[-1] if "." in file_name else ""
        unique_filename = f"{uuid.uuid4().hex}.{file_extension}" if file_extension else f"{uuid.uuid4().hex}"
        
        # S3 key path
        s3_key = f"users/{username}/profile/{unique_filename}"

        try:
            # Generate presigned URL for PUT operation
            upload_url = self.s3_client.generate_presigned_url(
                "put_object",
                Params={
                    "Bucket": self.bucket,
                    "Key": s3_key,
                    "ContentType": file_type,
                },
                ExpiresIn=self.expiry,
            )

            # Generate presigned URL for GET operation (to view the image)
            image_url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": s3_key},
                ExpiresIn=self.expiry,
            )

            return {
                "upload_url": upload_url,
                "image_url": image_url,
                "s3_key": s3_key,
                "expires_in": self.expiry,
            }
        except ClientError as e:
            raise Exception(f"Error generating presigned URL: {str(e)}")

    def generate_image_url(
        self, username: str, s3_key: Optional[str] = None
    ) -> dict:
        """
        Generate presigned URL for viewing an image from S3.

        Args:
            username: Username
            s3_key: Optional S3 key. If not provided, gets latest profile image.

        Returns:
            dict with image_url, s3_key, expires_in
        """
        if not s3_key:
            # If no s3_key provided, you might want to get the latest profile image
            # For now, we'll require s3_key
            raise ValueError("s3_key is required")

        try:
            # Generate presigned URL for GET operation
            image_url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": s3_key},
                ExpiresIn=self.expiry,
            )

            return {
                "image_url": image_url,
                "s3_key": s3_key,
                "expires_in": self.expiry,
            }
        except ClientError as e:
            raise Exception(f"Error generating image URL: {str(e)}")
    
    def get_object(self, s3_key: str) -> bytes:
        """
        Get object directly from S3 using boto3 (for server-side proxy).
        This bypasses presigned URLs and uses IAM credentials directly.
        
        Args:
            s3_key: S3 key of the object
            
        Returns:
            Object content as bytes
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
            return response['Body'].read()
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'AccessDenied':
                raise Exception(f"Access denied. IAM user needs s3:GetObject permission. Check IAM policies and bucket policies.")
            raise Exception(f"Error getting object from S3: {str(e)}")
    
    def get_object_content_type(self, s3_key: str) -> str:
        """
        Get content type of an object from S3.
        
        Args:
            s3_key: S3 key of the object
            
        Returns:
            Content type string
        """
        try:
            response = self.s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            return response.get('ContentType', 'image/jpeg')
        except ClientError as e:
            return 'image/jpeg'  # Default fallback
    
    def delete_object(self, s3_key: str) -> bool:
        """
        Delete an object from S3.
        
        Args:
            s3_key: S3 key of the object to delete
            
        Returns:
            True if deletion was successful
            
        Raises:
            Exception: If deletion fails
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'AccessDenied':
                raise Exception(f"Access denied. IAM user needs s3:DeleteObject permission. Check IAM policies and bucket policies.")
            raise Exception(f"Error deleting object from S3: {str(e)}")
    
    def upload_image(self, image_data: bytes, s3_key: str, content_type: str = "image/jpeg") -> str:
        """
        Upload an image directly to S3.
        
        Args:
            image_data: Image data as bytes (from PIL Image.save() or similar)
            s3_key: S3 key where the image will be stored
            content_type: MIME type of the image (default: image/jpeg)
            
        Returns:
            S3 key of the uploaded image
            
        Raises:
            Exception: If upload fails
        """
        try:
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=image_data,
                ContentType=content_type,
            )
            return s3_key
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'AccessDenied':
                raise Exception(f"Access denied. IAM user needs s3:PutObject permission. Check IAM policies and bucket policies.")
            raise Exception(f"Error uploading image to S3: {str(e)}")
    
    def get_merged_image_url(self, s3_key: str) -> str:
        """
        Generate presigned URL for a merged image stored in S3.
        
        Args:
            s3_key: S3 key of the merged image
            
        Returns:
            Presigned URL for viewing the image
        """
        try:
            image_url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": s3_key},
                ExpiresIn=self.expiry,
            )
            return image_url
        except ClientError as e:
            raise Exception(f"Error generating merged image URL: {str(e)}")


# Singleton instance
s3_service = S3Service()

