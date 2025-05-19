from datetime import datetime
from typing import Any, Dict, Optional, List
from uuid import uuid4

from dotenv import load_dotenv
from sqlalchemy import Column
from sqlmodel import SQLModel, Field, Relationship, JSON

load_dotenv()  # .env의 DATABASE_URL 등 불러오기
from app.models.db_models import SQLModel
target_metadata = SQLModel.metadata

class GenerationRequest(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    user_id: str  # 사용자 식별자 (추후 인증 시스템 연동)
    generation_type: str = Field(index=True)  # 'txt2img', 'img2img', 'inpaint'
    sub_type: Optional[str] = Field(default=None, index=True)
    # 텍스트 기반 생성
    prompt: Optional[str] = None
    model_name: Optional[str] = None
    lora: Optional[str] = None
    negative_prompt: Optional[str] = None
    inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    clip_skip: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    # 필터 기반 img2img
    filter_name: Optional[str] = None
    # 입력 이미지/마스크 참조 -> 근데 쓰기 어려울듯
    input_image_s3_key: Optional[str] = None
    input_image_s3_url: Optional[str] = None
    mask_s3_key: Optional[str] = None
    mask_s3_url: Optional[str] = None
    # 그 외 동적 파라미터
    extra_params: Dict[str, Any] = Field(
        sa_column=Column(JSON),      # JSON 컬럼으로 지정
        default_factory=dict         # 기본값을 빈 dict 로
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)

    images: List["GeneratedImage"] = Relationship(back_populates="request")


class GeneratedImage(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    request_id: str = Field(foreign_key="generationrequest.id")
    index: int  # 요청 내 이미지 순서
    s3_key: str
    s3_url: str
    width: Optional[int] = None
    height: Optional[int] = None

    request: GenerationRequest = Relationship(back_populates="images")
