from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import base64
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

class TextInput(BaseModel):
    text: str

def generate_image_from_text(text: str) -> BytesIO:
    img = Image.new('RGB', (256, 256), color=(255, 255, 255))
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

@app.post("/generate-image/")
async def generate_image(input_data: TextInput):
    img_byte_arr = generate_image_from_text(input_data.text)
    encoded_img = base64.b64encode(img_byte_arr.read()).decode('utf-8')
    return JSONResponse(content={"image_data": encoded_img})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)