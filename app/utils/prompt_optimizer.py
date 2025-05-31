from typing import Tuple

# 프롬프트 강화 : 아직 미적용

# 1) 토큰 단위로 관리
POSITIVE_ENHANCERS = {
    "hanbok": [
        "고해상도", "성별에 맞는 한복", "정밀한 묘사", "자연스러운 옷 주름", "부드러운 명암 대비",
        "전통 의상", "선명한 얼굴", "또렷한 눈동자", "균형 잡힌 얼굴", "아름다운 눈", "머리에 아무것도 없는",
    ],
    "hansik": [
        "음식 사진 스타일", "먹음직스러운", "음식 중심 구도", "한국 음식"
    ],
    "hanok": [
        "고화질", 
    ],
}

NEGATIVE_ENHANCERS = {
    "hanbok": [
        "못생긴", "여자 한복을 입은 남자", "흐릿한", "저화질", "비현실적인", "얼굴 왜곡", "이상한 눈",
        "너무 많은 장신구", "과도한 액세서리", "장식이 많은 머리",
        "이상한 옷 구조", "왜곡된 카라", "부자연스러운 옷고름",
        "흐릿한 얼굴", "이상한 손가락", "깨진 옷 디테일", "기괴한 자세", "전통 머리 장식", "노이즈"
    ],
    "hansik": [
        "말려있는"
    ],
    "hanok": [
        "흐릿한 이미지", "저해상도",
    ],
}

def enhance_prompt(
    prompt: str,
    negative_prompt: str = None,
    lora: str = None,
) -> Tuple[str, str, str]:
    # 2) 선택된 lora가 키에 없으면 기본값으로 대체
    pos = POSITIVE_ENHANCERS.get(lora, ["최고 화질", "선명한 화질", "고해상도"])
    neg = NEGATIVE_ENHANCERS.get(lora, ["저해상도", "흐릿한 이미지", "세부 묘사 부족"])

    prompt         = f"{prompt}, {', '.join(pos)}"
    negative_prompt = f"{negative_prompt}, {', '.join(neg)}" if negative_prompt else ", ".join(neg)
    return prompt, negative_prompt, lora

# 테스트 코드
if __name__ == "__main__":
    test_prompt = "a woman in red dress walking in the city"
    
    # 기본 강화
    enhanced, neg, lora = enhance_prompt(test_prompt)
    print(f"원본: {test_prompt}")
    print(f"강화됨: {enhanced}")
    print("-" * 50)