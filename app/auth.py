from typing import NamedTuple

class User(NamedTuple):
    id: str

async def get_current_user() -> User:
    # 개발용 더미 ID. 나중에 실제 인증 로직으로 교체
    return User(id="dev-user")