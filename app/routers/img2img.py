from app.models.request_models import GenerateRequest
from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.responses import JSONResponse
import json

from app.db import get_session
from app.auth import get_current_user
from app.models.db_models import GenerationRequest, GeneratedImage
from app.services.img2img.edge_copy import edge_copy
from app.services.img2img.pose_copy import pose_copy
from app.services.img2img.style_copy import style_copy

router = APIRouter()

# 헬퍼 함수: DB에 생성 요청과 결과를 기록
async def save_generation_results(
    response: JSONResponse,
    request: GenerateRequest,
    sub_type: str,
    session: AsyncSession,
    user_id: int
) -> list:
    """DB에 생성 요청 및 결과 저장"""
    # 응답 파싱
    content = json.loads(response.body.decode())
    items = content.get("results", [])
    
    # 1. 생성 요청 저장
    gen = GenerationRequest(
        user_id=user_id,
        generation_type="img2img",
        prompt=request.prompt,
        model_name=request.model,
        lora=request.lora,
        negative_prompt=request.negative_prompt,
        inference_steps=request.inference_steps,
        guidance_scale=request.guidance_scale,
        clip_skip=request.clip_skip,
        sub_type=sub_type,
        extra_params={},
    )
    session.add(gen)
    await session.commit()
    await session.refresh(gen)
    
    # 2. 생성된 이미지 저장
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

@router.post("/edge")
async def edge(
    request: GenerateRequest = Depends(GenerateRequest.as_form_json),
    image: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
    user = Depends(get_current_user),
):
    resp = edge_copy(request, image)
    await save_generation_results(resp, request, "edge", session, user.id)
    return resp

@router.post("/pose")
async def pose(
    request: GenerateRequest = Depends(GenerateRequest.as_form_json),
    image: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
    user = Depends(get_current_user),
):
    resp = pose_copy(request, image)
    await save_generation_results(resp, request, "pose", session, user.id)
    return resp

@router.post("/style")
async def style(
    request: GenerateRequest = Depends(GenerateRequest.as_form_json),
    image: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
    user = Depends(get_current_user),
):
    resp = style_copy(request, image)
    await save_generation_results(resp, request, "style", session, user.id)
    return resp

@router.post("/face")
async def face(
    request: GenerateRequest = Depends(GenerateRequest.as_form_json),
    image: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
    user = Depends(get_current_user),
):
    resp = style_copy(request, image) 
    await save_generation_results(resp, request, "face", session, user.id)
    return resp