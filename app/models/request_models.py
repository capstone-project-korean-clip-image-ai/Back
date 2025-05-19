from typing import Optional
from pydantic import BaseModel
from fastapi import Form
import json

class GenerateRequest(BaseModel):
    prompt: Optional[str] = None
    model: Optional[str] = None
    lora: Optional[str] = None
    imgNum: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    negative_prompt: Optional[str] = None
    inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    clip_skip: Optional[int] = None

    @classmethod
    def as_form_json(cls, data: str = Form(...)):
        # data에는 JSON.stringify(params)가 들어옵니다.
        return cls(**json.loads(data))

