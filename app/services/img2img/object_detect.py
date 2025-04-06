from app.config import MODEL_PATHS, LORA_PATHS, DEVICE
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import torch
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image

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

# 전역 변수로 SAM 모델 캐싱
sam_checkpoint = "sam_vit_l_0b3195.pth"
sam_model_type = "vit_l"

print("SAM 모델 로드 중...")
sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
sam.to(DEVICE)
predictor = SamPredictor(sam)
print("SAM 모델 로드 완료")

async def object_detect_process(file: UploadFile):
    # 이미지 파일 저장 경로 설정
    file_path = UPLOAD_DIR / file.filename
    print(f"파일 저장 경로: {file_path}")
    try:
        # 이미지 저장
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # 이미지 로드
        image = Image.open(file_path).convert("RGB")
        image_np = np.array(image)

        # SAM 모델로 객체 탐지
        predictor.set_image(image_np)

        # 중심점 기준으로 전경 감지
        image_height, image_width, _ = image_np.shape
        center_x = image_width // 2
        center_y = image_height // 2

        input_points = np.array([
            [center_x, center_y],
            [center_x - image_width // 4, center_y],
            [center_x + image_width // 4, center_y],
            [center_x, center_y - image_height // 4],
            [center_x, center_y + image_height // 4]
        ])
        input_labels = np.array([1, 1, 1, 1, 1])

        masks, _, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )

        # 객체 마스크를 이미지 파일로 저장
        object_paths = []

        for i, mask in enumerate(masks):
            mask_image = Image.fromarray((mask * 255).astype("uint8"))
            object_path = UPLOAD_DIR / f"{remove_extension(file.filename)}_object_{i+1}.jpg"
            print(f"생성된 객체 경로: {object_path}")
            mask_image.save(object_path)
            object_paths.append(object_path)

        return file_path, object_paths
    except Exception as e:
        raise RuntimeError(f"이미지 업로드 오류: {str(e)}")
