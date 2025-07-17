from fastapi import HTTPException, APIRouter, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select, delete
from app.db import get_session
from app.models.db_models import GenerationRequest, GeneratedImage
from app.auth import get_current_user

router = APIRouter()

@router.get("")
async def list_logs(
    session: AsyncSession = Depends(get_session),
    user = Depends(get_current_user),
):
    stmt = (
        select(GenerationRequest)
        .where(GenerationRequest.user_id == user.id)
        .order_by(GenerationRequest.created_at.desc())
    )
    gens = (await session.execute(stmt)).scalars().all()

    result = []
    for g in gens:
        imgs = (await session.execute(
            select(GeneratedImage).where(GeneratedImage.request_id == g.id)
        )).scalars().all()
        result.append({
            "id": g.id,
            "type": g.generation_type,
            "subType": g.sub_type,
            "prompt": g.prompt,
            "model": g.model_name,
            "loras": g.lora.split(",") if g.lora else [],
            "negativePrompt": g.negative_prompt,
            "inferenceSteps": g.inference_steps,
            "guidanceScale": g.guidance_scale,
            "clipSkip": g.clip_skip,
            "width": g.width,
            "height": g.height,
            "originalImage": g.input_image_s3_url,
            "createdAt": g.created_at.isoformat(),
            "images": [{"url": i.s3_url, "key": i.s3_key} for i in imgs]
        })
    return result

@router.delete("/{request_id}", status_code=204)
async def delete_log(
    request_id: str,
    session: AsyncSession = Depends(get_session),
    user = Depends(get_current_user),
):
    stmt = select(GenerationRequest).where(
        GenerationRequest.id == request_id,
        GenerationRequest.user_id == user.id
    )
    result = await session.execute(stmt)
    gen = result.scalar_one_or_none()
    if not gen:
        raise HTTPException(status_code=404, detail="Log not found")

    await session.execute(delete(GeneratedImage).where(
        GeneratedImage.request_id == request_id
    ))
    await session.execute(delete(GenerationRequest).where(
        GenerationRequest.id == request_id
    ))
    await session.commit()

    return Response(status_code=204)