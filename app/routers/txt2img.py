from fastapi import APIRouter
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
import json

from app.db import get_session
from app.models.db_models import GenerationRequest, GeneratedImage
from app.utils.s3 import upload_image_to_s3
from app.models.request_models import GenerateRequest
from app.services.image_generator import generate_image
from app.auth import get_current_user

router = APIRouter()

@router.post("")
async def txt2img(
    request: GenerateRequest,
    session: AsyncSession = Depends(get_session),
    user = Depends(get_current_user),
):
    # 이미지 생성 (기존 로직)
    response = generate_image(request)  # JSONResponse({"results": [{key, url}, …]})
    if response.status_code != 200:
        return response
    data = json.loads(response.body.decode())
    items = data["results"]

    # DB table GenerationRequest
    gen = GenerationRequest(
        user_id=user.id,
        generation_type="txt2img",
        sub_type=None,
        prompt=request.prompt,
        model_name=request.model,
        lora=request.lora,
        negative_prompt=request.negative_prompt,
        inference_steps=request.inference_steps,
        guidance_scale=request.guidance_scale,
        clip_skip=request.clip_skip,
        width=request.width,
        height=request.height,
        extra_params={},  # 필요 시 advancedOptions 나머지 필드 병합
    )
    session.add(gen)
    await session.commit()
    await session.refresh(gen)

    # DB table GeneratedImage
    for idx, item in enumerate(items):
        img = GeneratedImage(
            request_id=gen.id,
            index=idx,
            s3_key=item["key"],
            s3_url=item["url"],
            width=request.width,
            height=request.height,
        )
        session.add(img)
    await session.commit()

    return response

@router.get("")
async def ignore_get_generate():
    return {"message": "GET 요청은 지원되지 않습니다. POST 요청을 사용하세요."}
