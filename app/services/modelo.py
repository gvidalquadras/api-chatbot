import joblib

from app.config import MODEL_PATH
from app.schemas.prestamo import PrediccionSalida

# Orden en el que el modelo espera las variables
VARIABLES = [
    "edad",
    "ingresos_mensuales",
    "importe_prestamo",
    "plazo_meses",
    "antiguedad_laboral_anios",
    "impagos_previos",
]

# Umbrales del enunciado: por debajo de 0.2 se aprueba, por debajo de 0.5 se revisa.
UMBRAL_APROBAR = 0.2
UMBRAL_REVISAR = 0.5

# Se carga al importar el módulo, así que se lee del disco una vez al arrancar la API
# y no en cada petición.
modelo = joblib.load(MODEL_PATH)


def decidir(probabilidad: float) -> str:
    if probabilidad < UMBRAL_APROBAR:
        return "aprobar"
    if probabilidad < UMBRAL_REVISAR:
        return "revisar"
    return "rechazar"


def predecir(entrada) -> PrediccionSalida:
    """Recibe un PrestamoEntrada ya validado y devuelve la probabilidad de impago y la decisión."""
    # La fila se construye recorriendo VARIABLES y no el orden de los campos de Pydantic,
    # para que no dependa de cómo estén declarados en el esquema.
    fila = [getattr(entrada, nombre) for nombre in VARIABLES]
    # predict_proba devuelve [[P(no impago), P(impago)]] para una sola fila.
    probabilidad = float(modelo.predict_proba([fila])[0][1])
    return PrediccionSalida(probabilidad_impago=probabilidad, decision=decidir(probabilidad))
