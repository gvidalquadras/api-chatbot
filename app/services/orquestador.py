import json

from pydantic import ValidationError

from app.schemas.chat import ChatSalida, HerramientaUsada
from app.schemas.prestamo import PrestamoEntrada
from app.services.modelo import predecir

NOMBRE_TOOL = "calcular_riesgo_impago"
MAX_VUELTAS = 4

SISTEMA = (
    "Eres un asistente de riesgo de un banco. Cuando tengas todos los datos de una solicitud de "
    "prestamo, llama a la herramienta calcular_riesgo_impago. Si falta algun dato, pidelo. "
    "Nunca inventes valores. Responde en espanol y de forma breve."
)


def leer_argumentos(texto):
    """YA HECHO: convierte el texto JSON de la tool en dict ({} si no es JSON valido)."""
    try:
        datos = json.loads(texto)
    except (json.JSONDecodeError, TypeError):
        return {}
    if isinstance(datos, dict):
        return datos
    return {}


def definir_tool():
    """Definición de la tool en el formato de OpenAI, que es el que entiende LiteLLM."""
    return {
        "type": "function",
        "function": {
            "name": NOMBRE_TOOL,
            "description": (
                "Calcula la probabilidad de impago de una solicitud de préstamo y devuelve "
                "la decisión (aprobar, revisar o rechazar). Necesita los seis datos."
            ),
            # Los parámetros salen del propio esquema de Pydantic. Si se cambia un campo o
            # un rango en PrestamoEntrada, el LLM recibe la definición ya actualizada.
            "parameters": PrestamoEntrada.model_json_schema(),
        },
    }


def _resumir_errores(error):
    """Resume un ValidationError en "campo: mensaje" separados por punto y coma, más corto que el texto de Pydantic."""
    partes = []
    for detalle in error.errors():
        campo = ".".join(str(trozo) for trozo in detalle["loc"])
        partes.append(f"{campo}: {detalle['msg']}")
    return "; ".join(partes)


def ejecutar_tool(nombre, argumentos_json):
    """Valida los argumentos y llama al modelo.

    Si la tool no existe, el JSON está roto o los datos no pasan la validación, devuelve
    {"ok": False, "error": ...} en vez de lanzar una excepción. Así el LLM lee el error y
    puede corregir la llamada o pedirle el dato al usuario.
    """
    if nombre != NOMBRE_TOOL:
        return {"ok": False, "error": f"la herramienta '{nombre}' no existe"}

    argumentos = leer_argumentos(argumentos_json)
    if not argumentos:
        return {"ok": False, "error": "los argumentos no son un objeto JSON con datos"}

    try:
        entrada = PrestamoEntrada(**argumentos)
    except ValidationError as error:
        return {"ok": False, "error": _resumir_errores(error)}

    salida = predecir(entrada)
    return {"ok": True, **salida.model_dump()}


def responder(mensajes, llm):
    """Bucle de conversación: LLM -> tool -> LLM, hasta que el LLM conteste con texto.

    Se corta a las MAX_VUELTAS llamadas al LLM para que un LLM que repite una llamada
    errónea no deje la petición colgada.
    """
    historial = [{"role": "system", "content": SISTEMA}]
    for mensaje in mensajes:
        historial.append(mensaje.model_dump())

    herramientas = [definir_tool()]
    # Última tool ejecutada. Es la que se devuelve al frontend para que muestre los
    # argumentos y el resultado.
    herramienta_usada = None

    for _ in range(MAX_VUELTAS):
        respuesta = llm.completar(historial, herramientas)
        llamadas = respuesta.get("tool_calls")

        if not llamadas:
            texto = respuesta.get("content") or "No tengo respuesta para eso."
            return ChatSalida(respuesta=texto, herramienta=herramienta_usada)

        # El mensaje del asistente con las tool_calls tiene que ir en el historial antes
        # de los mensajes "tool"; si no, la API del LLM rechaza la conversación.
        historial.append(respuesta)
        for llamada in llamadas:
            nombre = llamada["function"]["name"]
            argumentos_json = llamada["function"]["arguments"]
            resultado = ejecutar_tool(nombre, argumentos_json)
            historial.append(
                {
                    "role": "tool",
                    "tool_call_id": llamada["id"],
                    "content": json.dumps(resultado),
                }
            )
            herramienta_usada = HerramientaUsada(
                nombre=nombre,
                argumentos=leer_argumentos(argumentos_json),
                resultado=resultado,
            )

    return ChatSalida(
        respuesta="No he podido completar el cálculo. Revisa los datos de la solicitud e inténtalo de nuevo.",
        herramienta=herramienta_usada,
    )
