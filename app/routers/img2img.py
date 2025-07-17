from app.models.request_models import GenerateRequest
from fastapi import APIRouter, UploadFile, File, Depends, Form
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.responses import JSONResponse
import json
from PIL import Image

from app.db import get_session
from app.auth import get_current_user
from app.models.db_models import GenerationRequest, GeneratedImage
from app.services.img2img.edge_copy import edge_copy
from app.services.img2img.pose_copy import pose_copy
from app.services.img2img.style_copy import style_copy
from app.services.img2img.filter_copy import filter_copy
from app.utils.s3 import upload_image_to_s3

router = APIRouter()

# 헬퍼 함수: DB에 생성 요청과 결과를 기록
async def save_generation_results(
    response: JSONResponse,
    request: GenerateRequest,
    sub_type: str,
    session: AsyncSession,
    user_id: int,
    input_key: str = None,
    input_url: str = None,
) -> list:
    # 응답 파싱
    content = json.loads(response.body.decode())
    items = content.get("results", [])
    
    # 생성 요청 저장
    gen = GenerationRequest(
        user_id=user_id,
        generation_type="img2img",
        prompt=request.prompt,
        model_name=request.model,
        lora=",".join(request.loras or []),
        negative_prompt=request.negative_prompt,
        inference_steps=request.inference_steps,
        guidance_scale=request.guidance_scale,
        clip_skip=request.clip_skip,
        sub_type=sub_type,
        extra_params={},
        input_image_s3_key=input_key,     # 추가: 원본 이미지 S3 key 저장
        input_image_s3_url=input_url,     # 추가: 원본 이미지 S3 URL 저장
    )
    session.add(gen)
    await session.commit()
    await session.refresh(gen)
    
    # 생성된 이미지 저장
    for idx, item in enumerate(items):
        img = GeneratedImage(
            request_id=gen.id,
            index=idx,
            s3_key=item["key"],
            s3_url=item["url"],
        )
        session.add(img)
    await session.commit()
    
    return items


@router.post("/edge") # 형태
async def edge(
    request: GenerateRequest = Depends(GenerateRequest.as_form_json),
    image: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
    user = Depends(get_current_user),
):
    # 원본 이미지 S3 업로드
    img = Image.open(image.file).convert("RGB")
    input_key, input_url = upload_image_to_s3(img, folder="input_images")
    # 필터 처리
    resp = edge_copy(request, image)
    await save_generation_results(resp, request, "edge", session, user.id, input_key, input_url)
    return resp

@router.post("/pose") # 포즈
async def pose(
    request: GenerateRequest = Depends(GenerateRequest.as_form_json),
    image: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
    user = Depends(get_current_user),
):
    img = Image.open(image.file).convert("RGB")
    input_key, input_url = upload_image_to_s3(img, folder="input_images")
    resp = pose_copy(request, image)
    await save_generation_results(resp, request, "pose", session, user.id, input_key, input_url)
    return resp

@router.post("/style") # 화풍
async def style(
    request: GenerateRequest = Depends(GenerateRequest.as_form_json),
    image: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
    user = Depends(get_current_user),
):
    img = Image.open(image.file).convert("RGB")
    input_key, input_url = upload_image_to_s3(img, folder="input_images")
    resp = style_copy(request, image)
    await save_generation_results(resp, request, "style", session, user.id, input_key, input_url)
    return resp

@router.post("/filter") # 대상
async def filter(
    filter: str = Form(...),
    imgNum: int = Form(...),
    image: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
    user = Depends(get_current_user),
):
    img = Image.open(image.file).convert("RGB")
    input_key, input_url = upload_image_to_s3(img, folder="input_images")
    resp = filter_copy(filter, imgNum, image)
    
    content = json.loads(resp.body.decode())
    items = content.get("results", [])

    gen = GenerationRequest(
        user_id=user.id,
        generation_type="img2img",
        sub_type="filter",
        extra_params={},
        input_image_s3_key=input_key,    # 추가: 원본 이미지 S3 key 저장
        input_image_s3_url=input_url,    # 추가: 원본 이미지 S3 URL 저장
    )
    session.add(gen)
    await session.commit()
    await session.refresh(gen)

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