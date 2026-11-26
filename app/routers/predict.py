from fastapi import APIRouter

from app.schemas.prestamo import PrediccionSalida, PrestamoEntrada
from app.services.modelo import predecir

router = APIRouter()


# FastAPI valida el cuerpo con PrestamoEntrada antes de entrar en la función.
# Si un dato está fuera de rango o sobra un campo, responde 422 sin llegar al modelo.
@router.post("/predict", response_model=PrediccionSalida)
def predict(entrada: PrestamoEntrada) -> PrediccionSalida:
    return predecir(entrada)
