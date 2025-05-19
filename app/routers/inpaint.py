from app.models.request_models import GenerateRequest
from fastapi import APIRouter, UploadFile, File, Depends, Form
from sqlalchemy.ext.asyncio import AsyncSession
import json

from app.db import get_session
from app.auth import get_current_user
from app.models.db_models import GenerationRequest, GeneratedImage
from app.services.inpaint.object_detect import object_detect_process
from app.services.inpaint.erase_object import erase_object
from app.services.inpaint.redraw_object import redraw_image

router = APIRouter()

@router.post("/object_detect")
async def image_object_detect(
    file: UploadFile, 
    x: int = Form(...), 
    y: int = Form(...)
):
    # x: int = Form(...) <- FromData 안의 필드로 받음
    print(f"이미지 파일: {file.filename}")
    print(f"좌표: ({x}, {y})")
    return await object_detect_process(file, x, y)

@router.post("/erase_object")
async def erasing_object(
    image: UploadFile, 
    object: UploadFile, 
    session: AsyncSession = Depends(get_session), 
    user = Depends(get_current_user),
):
    # Inpainting, key/url 반환
    resp = await erase_object(image, object)
    content = json.loads(resp.body.decode())
    items = content.get("results", [])

    # 요청 메타 저장
    gen = GenerationRequest(
        user_id=user.id,
        generation_type="inpainting",
        sub_type="erase",
        extra_params={},
    )
    session.add(gen)
    await session.commit()
    await session.refresh(gen)

    # 생성된 inpaint 이미지 저장
    for idx, item in enumerate(items):
        img = GeneratedImage(
            request_id=gen.id,
            index=idx,
            s3_key=item["key"],
            s3_url=item["url"],
        )
        session.add(img)
    await session.commit()

    # 원본 JSONResponse 반환
    return resp

@router.post("/redraw_object")
async def redraw_object(
    # metadata를 한 번에 묶어서 받는다
    request: GenerateRequest = Depends(GenerateRequest.as_form_json),
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
    user = Depends(get_current_user),
):
    # Inpainting, key/url 반환
    resp = redraw_image(request, image, mask)
    content = json.loads(resp.body.decode())
    items = content.get("results", [])

    # 요청 메타 저장
    gen = GenerationRequest(
        user_id=user.id,
        generation_type="inpainting",
        prompt=request.prompt,
        model_name=request.model,
        lora=request.lora,
        negative_prompt=request.negative_prompt,
        inference_steps=request.inference_steps,
        guidance_scale=request.guidance_scale,
        clip_skip=request.clip_skip,
        sub_type="redraw",
        extra_params={},
    )
    session.add(gen)
    await session.commit()
    await session.refresh(gen)

    # 생성된 inpaint 이미지 저장
    for idx, item in enumerate(items):
        img = GeneratedImage(
            request_id=gen.id,
            index=idx,
            s3_key=item["key"],
            s3_url=item["url"],
        )
        session.add(img)
    await session.commit()

    return resp

@router.get("")
async def ignore_get_generate():
    return {"message": "GET 요청은 지원되지 않습니다. POST 요청을 사용하세요."}