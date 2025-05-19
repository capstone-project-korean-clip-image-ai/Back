from app.utils.s3 import upload_image_to_s3
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from simple_lama_inpainting import SimpleLama
from PIL import Image
from io import BytesIO

router = APIRouter()
simple_lama = SimpleLama()

async def erase_object(image: UploadFile = File(...), object: UploadFile = File(...)):
    original = Image.open(BytesIO(await image.read())).convert("RGB")
    mask_image = Image.open(BytesIO(await object.read())).convert("L")  # 마스크는 흑백
    result = simple_lama(original, mask_image)

    # S3에 업로드
    items = []
    s3_key, url = upload_image_to_s3(result, folder="inpaint")
    items.append({"key": s3_key, "url": url})

    return JSONResponse(content={"results": items})
