from langchain_core.pydantic_v1 import BaseSettings


class Settings(BaseSettings):
    LLM_MODEL: str = "claude-3-haiku-20240307"
    MAX_ITERATIONS: int = 10
    ENABLE_HUMAN_FEEDBACK: bool = True

    class Config:
        env_file = ".env"


settings = Settings()
