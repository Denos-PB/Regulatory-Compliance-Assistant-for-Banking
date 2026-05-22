import json
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from ..config import load_indexing_config

load_dotenv()
logger = logging.getLogger(__name__)


def _client(model: str | None = None, cfg: dict | None = None) -> OpenAIEmbeddings:
    c = {**load_indexing_config(), **(cfg or {})}
    model = model or c["embedding_model"]
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env")
    return OpenAIEmbeddings(model=model, api_key=api_key)
