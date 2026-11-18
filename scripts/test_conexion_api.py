import os
import requests
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("EMT_CLIENT_ID", "").strip()
PASSKEY = os.getenv("EMT_PASSKEY", "").strip()

BASE_URL = "https://" + "openapi." + "emtmadrid.es/v2"


def probar_conexion_emt():
    assert CLIENT_ID and PASSKEY, "Faltan EMT_CLIENT_ID o EMT_PASSKEY en el archivo .env"

    # PASO 1: Autenticación para obtener el pase temporal (accessToken)
    url_login = f"{BASE_URL}/mobilitylabs/user/login/"
    headers_login = {
        "X-ClientId": CLIENT_ID,
        "passKey": PASSKEY,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ProyectoPuntualidadEMT/1.0",
        "Accept": "application/json"
    }

    resp_login = requests.get(url_login, headers=headers_login, timeout=15)
    resp_login.raise_for_status()
    datos_login = resp_login.json()

    if datos_login.get("code") not in ("00", "01"):
        raise RuntimeError(f"Error en login MobilityLabs: {datos_login}")

    bloque_login = next(iter(datos_login.get("data", [])), {})
    access_token = bloque_login["accessToken"]
    print("[OK] Paso 1 completado: Autenticación correcta en MobilityLabs EMT.")

    # PASO 2: Consultar tiempos reales de llegada en la Parada 78 (Neptuno)
    stop_prueba = "78"
    url_arrives = f"{BASE_URL}/transport/busemtmad/stops/{stop_prueba}/arrives/"
    headers_api = {
        "accessToken": access_token,
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ProyectoPuntualidadEMT/1.0"
    }
    payload = {
        "cultureInfo": "ES",
        "Text_StopRequired_YN": "Y",
        "Text_EstimationsRequired_YN": "Y",
        "Text_IncidencesRequired_YN": "N",
        "DateTime_Referenced_Incidencies_YYYYMMDD": ""
    }

    resp_arrives = requests.post(url_arrives, headers=headers_api, json=payload, timeout=15)
    resp_arrives.raise_for_status()
    datos_arrives = resp_arrives.json()

    bloque_arrives = next(iter(datos_arrives.get("data", [])), {})
    llegadas = bloque_arrives.get("Arrive", [])
    print(f"[OK] Paso 2 completado: {len(llegadas)} autobuses aproximándose ahora mismo a Neptuno:\n")

    for bus in llegadas[:5]:
        linea = bus.get("line")
        destino = bus.get("destination")
        segundos = bus.get("estimateArrive")
        distancia = bus.get("DistanceBus")
        print(f"  -> Línea {linea:<4} | Destino: {destino:<25} | Llegada en: {segundos:>4} s | Distancia: {distancia:>5} m")


if __name__ == "__main__":
    probar_conexion_emt()
