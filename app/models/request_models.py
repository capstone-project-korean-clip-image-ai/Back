from pydantic import BaseModel

class GenerateRequest(BaseModel):
    prompt: str
    model: str
    lora: str
    negative_prompt: str
    inference_steps: int
    guidance_scale: float
    clip_skip: int
