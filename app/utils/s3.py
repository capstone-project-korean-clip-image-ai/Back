import boto3
from app.config import BUCKET_NAME, CLOUDFRONT_DOMAIN
from io import BytesIO
import uuid

def upload_image_to_s3(image, folder: str):
    s3 = boto3.client("s3")
    bucket_name = BUCKET_NAME

    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    # 경로 및 파일명
    filename = f"{uuid.uuid4()}.jpg"
    s3_key = f"{folder}/{filename}"

    s3.put_object(
        Bucket=bucket_name,
        Key=s3_key,
        Body=buffer,
        ContentType="image/jpeg"
    )

    # presigned URL 생성
    cf_url = f"{CLOUDFRONT_DOMAIN}/{s3_key}"

    return s3_key, cf_url
