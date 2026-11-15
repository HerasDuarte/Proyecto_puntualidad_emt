"""
Módulo de Funciones Auxiliares y Utilidades Matemáticas del Proyecto.
Centraliza algoritmos genéricos reutilizables en múltiples fases del pipeline.
"""

import re
import numpy as np

def punto_en_poligono(x, y, poligono):
    """
    Algoritmo matemático nativo de Ray-Casting para determinar si un punto (x, y)
    se encuentra dentro de un polígono irregular de N vértices.
    Evita la importación de librerías gráficas pesadas en entornos de producción.
    """
    dentro = False
    n = len(poligono)
    if n == 0:
        return dentro
    p1x, p1y = poligono[0]
    for i in range(n + 1):
        p2x, p2y = poligono[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        dentro = not dentro
        p1x, p1y = p2x, p2y
    return dentro

def clasificar_regimen_horario(codigo_corto):
    """
    Segmenta las líneas de la EMT de Madrid en tres regímenes de operación
    basándose en las convenciones nomencláteras del estándar comercial GTFS.
    """
    codigo_str = str(codigo_corto).strip().upper()
    if codigo_str.startswith('N'):
        return 'Servicio Nocturno (Búho)'
    elif codigo_str == '203' or 'AEROPUERTO' in codigo_str:
        return 'Servicio 24 Horas'
    else:
        return 'Servicio Diurno Regular'

def extraer_calle_principal(nombre_parada):
    """
    Aísla la arteria vial principal del literal comercial de la parada de la EMT
    mediante expresiones regulares (Regex), eliminando cruces secundarios y portales.
    """
    txt = str(nombre_parada).strip()
    partes = re.split(r'\s*-\s*|\s+AND\s+|\s+Y\s+', txt, maxsplit=1)
    calle = partes[0].strip()
    calle = re.sub(r'\s*\d+$', '', calle).strip()
    return calle


def categorizar_tramos(total_marquesinas):
    """
    Segmenta las arterias viales de Madrid en tramos discretos según 
    la densidad y frecuencia de marquesinas que soportan.
    """
    if total_marquesinas <= 3:
        return 'Capilaridad Local\n(1-3 paradas)'
    elif total_marquesinas <= 10:
        return 'Ejes Secundarios\n(4-10 paradas)'
    else:
        return 'Grandes Arterias\n(>10 paradas)'

def calcular_distancia_haversine(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia en línea recta en metros entre dos puntos geográficos
    utilizando la fórmula del Haversine, optimizada para matrices de NumPy.
    """
    R = 6371000.0  # Radio medio de la Tierra en metros
    rad_lat1, rad_lon1 = np.radians(lat1), np.radians(lon1)
    rad_lat2, rad_lon2 = np.radians(lat2), np.radians(lon2)
    dlat, dlon = rad_lat2 - rad_lat1, rad_lon2 - rad_lon1
    a = np.sin(dlat/2.0)**2 + np.cos(rad_lat1) * np.cos(rad_lat2) * np.sin(dlon/2.0)**2
    return R * (2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))

def clasificar_extension(km):
    """
    Clasifica los trayectos comerciales de la EMT según su extensión kilométrica total.
    """
    if km <= 5.0: 
        return 'Líneas Cortas / Barrio\n(0-5 km)'
    elif km <= 15.0: 
        return 'Líneas Medias / Regulares\n(5-15 km)'
    else: 
        return 'Largo Recorrido\n(>15 km)'

def clasificar_territorio_avanzado(ratio):
    """
    Determina el nivel de complejidad vial y fricción urbana de un trayecto 
    combinando su índice de tortuosidad y la densidad de su muestreo geométrico.
    """
    if ratio == 1.0: return 'Interior Puro\n(Centro)'
    elif ratio == 0.0: return 'Exterior Puro\n(Periferia)'
    elif ratio <= 0.30: return 'Transversal\nPeriférica (0-30%)'
    elif ratio <= 0.60: return 'Transversal\nEquilibrada (30-60%)'
    else: return 'Transversal\nInvasiva (>60%)'
