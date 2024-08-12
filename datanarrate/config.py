from environs import Env

env = Env()
env.read_env(override=True)  # read .env file, if it exists


class Config:
    # Database configurations
    MYSQL_HOST = env.str("MYSQL_HOST", "localhost")
    MYSQL_USER = env.str("MYSQL_USER")
    MYSQL_PASSWORD = env.str("MYSQL_PASSWORD")
    MYSQL_DATABASE = env.str("MYSQL_DATABASE")
    SQLALCHEMY_DATABASE_URI = env.str("SQLALCHEMY_DATABASE_URI",
                                      f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DATABASE}")

    # Elasticsearch configurations
    ELASTICSEARCH_HOST = env.str("ELASTICSEARCH_HOST", "http://localhost:9200")
    ELASTICSEARCH_USERNAME = env.str("ELASTICSEARCH_USERNAME")
    ELASTICSEARCH_PASSWORD = env.str("ELASTICSEARCH_PASSWORD")
    ELASTICSEARCH_INDEX_PATTERN = env.str("ELASTICSEARCH_INDEX_PATTERN")

    # LLM configurations
    LLM_MODEL_NAME = env.str("LLM_MODEL_NAME", "deepseek-chat")
    OPENAI_API_BASE = env.str("OPENAI_API_BASE", "https://api.deepseek.com")
    OPENAI_API_KEY = env.str("OPENAI_API_KEY")

    # Other configurations
    LOG_LEVEL = env.str("LOG_LEVEL", "INFO")


config = Config()
