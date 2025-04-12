import boto3
from io import BytesIO
import uuid

def upload_image_to_s3(image, bucket_name: str, folder: str):
    s3 = boto3.client("s3")

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
    url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket_name, "Key": s3_key},
        ExpiresIn=300  # 5분
    )

    return url
