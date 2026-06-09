import sys
from types import ModuleType


def ensure_ragas_importable() -> None:
    name = "langchain_community.chat_models.vertexai"
    if name in sys.modules:
        return
    stub = ModuleType(name)

    class ChatVertexAI:
        pass

    stub.ChatVertexAI = ChatVertexAI
    sys.modules[name] = stub
