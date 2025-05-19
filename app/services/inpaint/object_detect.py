from fastapi.responses import JSONResponse
from app.config import BUCKET_NAME, MODEL_PATHS, LORA_PATHS, DEVICE
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import torch
import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image

from app.utils.s3 import upload_image_to_s3

app = FastAPI()

# 정적 파일 경로 설정
UPLOAD_DIR = Path("app/uploaded_images")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 파일 이름에서 확장자 제거하는 함수
def remove_extension(file_name: str) -> str:
    return Path(file_name).stem

# 모델 해제 함수
def unload_model(model):
    try:
        model.to("cpu")
        del model
        torch.cuda.empty_cache()
        print("모델 메모리 해제 완료")
    except Exception as e:
        print(f"모델 메모리 해제 중 오류 발생: {e}")

# 마스킹 영역을 확장하는 함수 (cv2 : dilated)
def expand_mask(mask: np.ndarray, kernel_size) -> np.ndarray:
    mask_uint8 = (mask * 255).astype("uint8")
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
    return (dilated > 127).astype("uint8")  # 다시 binary mask

# 전역 변수로 SAM 모델 캐싱
sam_checkpoint = "sam_vit_l_0b3195.pth"
sam_model_type = "vit_l"

async def object_detect_process(file: UploadFile, x: int, y: int):
    # 이미지 파일 저장 경로 설정
    file_path = UPLOAD_DIR / file.filename
    print(f"파일 저장 경로: {file_path}")

    print("SAM 모델 로드 중...")
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(DEVICE)
    predictor = SamPredictor(sam)
    print("SAM 모델 로드 완료")

    try:
        # 이미지 저장
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # 이미지 로드
        image = Image.open(file_path).convert("RGB")
        image_np = np.array(image)

        # SAM 모델로 객체 탐지
        predictor.set_image(image_np)

        # 클릭된 위치를 중심으로 전경 감지
        input_points = np.array([
            [x, y],  # 사용자가 클릭한 좌표
            [x - 30, y],
            [x + 30, y],
            [x, y - 30],
            [x, y + 30]
        ])
        input_labels = np.array([1, 1, 1, 1, 1])

        masks, _, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )

        file_path.unlink()  # 원본 이미지 파일 삭제

        # 객체 마스크를 S3에 저장
        items = []

        for i, mask in enumerate(masks):
            expanded_mask = expand_mask(mask, 25)
            mask_image = Image.fromarray((expanded_mask * 255).astype("uint8"))
            s3_key, mask_url = upload_image_to_s3(mask_image, folder="object_masks")
            print(f"생성된 마스크 경로: {mask_url}")
            items.append({"key": s3_key, "url": mask_url})

        return JSONResponse(content={"results": items})
    except Exception as e:
        raise RuntimeError(f"이미지 업로드 오류: {str(e)}")
