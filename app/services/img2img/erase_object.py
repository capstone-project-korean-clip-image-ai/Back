from app.config import MODEL_PATHS, DEVICE, BUCKET_NAME
from app.utils.s3 import upload_image_to_s3
from fastapi import APIRouter, UploadFile, File
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from io import BytesIO
import torch

router = APIRouter()

async def erase_object(image: UploadFile = File(...), object: UploadFile = File(...)):
    model_path = MODEL_PATHS.get("img2img", {}).get("base")

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(DEVICE)

    # 이미지 열기
    original = Image.open(BytesIO(await image.read())).convert("RGB")
    mask_image = Image.open(BytesIO(await object.read())).convert("L")  # 마스크는 흑백

    # 모델에 입력, 이미지의 가로 세로는 64의 배수 권장
    assert original.height % 64 == 0 and original.width % 64 == 0, "이미지는 64의 배수여야 합니다."

    result = pipe(
        prompt="remove the object completely and restore the area with plain background in same style",
        negative_prompt="object, person, logo, jewelry, text, texture, pattern, artifact, unnatural shapes",
        image=original,
        mask_image=mask_image,
        height=original.height,
        width=original.width,
        guidance_scale=8.5,  # ✅ 강화된 지시
        num_inference_steps=30,  # ✅ 좀 더 안정적 생성
    ).images[0]

    # S3에 업로드
    s3_url = upload_image_to_s3(result, bucket_name=BUCKET_NAME, folder="inpaint")

    return { "image_url": s3_url }
