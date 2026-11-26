from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class PrestamoEntrada(BaseModel):
    """Datos de una solicitud de préstamo, en el orden en que los espera el modelo."""

    # extra="forbid": si llega un campo que no está aquí (por ejemplo, uno que se ha
    # inventado el LLM), Pydantic da error en vez de ignorarlo.
    model_config = ConfigDict(extra="forbid")

    # Las descripciones pasan al JSON Schema de la tool, así que el LLM también las lee.
    # edad va antes que antiguedad_laboral_anios porque el validador de abajo la necesita.
    edad: int = Field(ge=18, le=75, description="Edad del solicitante en años")
    ingresos_mensuales: float = Field(gt=0, description="Ingresos netos al mes, en euros")
    importe_prestamo: float = Field(gt=0, description="Importe que se pide, en euros")
    plazo_meses: int = Field(ge=6, le=84, description="Plazo de devolución en meses")
    antiguedad_laboral_anios: int = Field(ge=0, le=50, description="Años trabajados")
    impagos_previos: int = Field(ge=0, le=20, description="Número de impagos anteriores")

    @field_validator("antiguedad_laboral_anios")
    @classmethod
    def antiguedad_coherente_con_edad(cls, valor: int, info: ValidationInfo) -> int:
        # Se toma 16 años como la edad mínima para empezar a trabajar.
        # Si la edad ya ha fallado su propia validación no está en info.data;
        # en ese caso no se comprueba nada aquí y queda solo el error de la edad.
        edad = info.data.get("edad")
        if edad is not None and valor > edad - 16:
            raise ValueError(
                f"la antigüedad laboral ({valor}) no puede ser mayor que edad - 16 ({edad - 16})"
            )
        return valor


class PrediccionSalida(BaseModel):
    """Respuesta de /predict y resultado de la tool."""

    probabilidad_impago: float = Field(ge=0, le=1)
    decision: Literal["aprobar", "revisar", "rechazar"]
