from fastapi import APIRouter, Depends

from app.schemas.chat import ChatEntrada, ChatSalida
from app.services.llm import get_llm
from app.services.orquestador import responder

router = APIRouter()


# El LLM llega por Depends(get_llm) y no se crea aquí dentro. Así los tests pueden
# cambiarlo por el simulado con app.dependency_overrides sin tocar este código.
@router.post("/chat", response_model=ChatSalida)
def chat(entrada: ChatEntrada, llm=Depends(get_llm)) -> ChatSalida:
    return responder(entrada.mensajes, llm)
