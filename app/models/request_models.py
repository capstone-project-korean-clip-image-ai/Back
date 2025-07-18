from typing import Optional, List
from pydantic import BaseModel
from fastapi import Form
import json

class GenerateRequest(BaseModel):
    prompt: Optional[str] = None
    model: Optional[str] = None
    loras: Optional[List[str]] = None
    imgNum: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    negative_prompt: Optional[str] = None
    inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    clip_skip: Optional[int] = None

    @classmethod
    def as_form_json(cls, data: str = Form(...)):
        # data : JSON.stringify(params)
        return cls(**json.loads(data))

