from fastapi import APIRouter
from app.models.request_models import GenerateRequest
from app.services.image_generator import generate_image

router = APIRouter()

@router.post("")
async def txt2img(request: GenerateRequest):
    image_path = generate_image(request)
    return image_path

@router.get("")
async def ignore_get_generate():
    return {"message": "GET 요청은 지원되지 않습니다. POST 요청을 사용하세요."}