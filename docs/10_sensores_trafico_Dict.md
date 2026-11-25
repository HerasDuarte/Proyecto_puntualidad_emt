# 📘 Diccionario de datos de `sensores`

A continuación se incluye un resumen para referencia rápida:

| Bloque | Variable | Tipo | Descripción Operativa |
| :--- | :--- | :--- | :--- |
| **Feed Bruto** | `timestamp_feed` | `String` | Fecha y hora oficial de publicación del boletín XML del Ayuntamiento. |
| **Feed Bruto** | `idelem` | `String` | Identificador único del punto de medida (espira electromagnética / estación). |
| **Feed Bruto** | `descripcion` | `String` | Ubicación vial descriptiva del sensor en el callejero de Madrid. |
| **Feed Bruto** | `acceso_asociado` | `String` | Código del eje o enlace viario asociado (especialmente en M-30). |
| **Feed Bruto** | `intensidad` | `Float64` | Flujo vehicular medido en vehículos por hora (`veh/h`). |
| **Feed Bruto** | `ocupacion` | `Float64` | Porcentaje de tiempo (`0–100%`) que el sensor permanece ocupado por vehículos. |
| **Feed Bruto** | `carga` | `Float64` | Grado de saturación vial (`0–126%`) estimado por el Centro de Gestión de Movilidad. |
| **Feed Bruto** | `nivel_servicio` | `Float64` | Indicador discreto de fluidez (`0`: Fluido, `1`: Lento, `2`: Retenciones, `3`: Congestionado). |
| **Feed Bruto** | `intensidad_sat` | `Float64` | Capacidad máxima teórica (`veh/h`) de la vía donde se ubica el sensor. |
| **Feed Bruto** | `velocidad_trafico` | `Float64` | Velocidad media medida en km/h (disponible principalmente en túneles y M-30). |
| **Feed Bruto** | `error_sensor` | `String` | Bandera de estado técnico (`'N'` = Operativo normal; `'S'` / `'E'` = Incidencia o sin dato). |
| **Feed Bruto** | `subarea` | `String` | Código de sector de regulación semafórica de Madrid. |
| **Feed Bruto** | `lat`, `lon` | `Float64` | Coordenadas geográficas transformadas desde `EPSG:25830` a `WGS84` (`EPSG:4326`). |
| **Cruce Espacial EMT** | `distancia_sensor_mas_cercano_m` | `Float64` | Distancia geodésica en metros desde la parada EMT hasta el sensor operativo más próximo. |
| **Cruce Espacial EMT** | `num_sensores_500m` | `Int64` | Número total de sensores operativos ubicados dentro del radio principal de 500 metros. |
| **Cruce Espacial EMT** | `tiene_cobertura_trafico_500m` | `Boolean` | Indicador de cobertura directa (`True` si hay ≥ 1 sensor a ≤ 500 m; `False` si usa respaldo ≤ 1.000 m). |
| **Cruce Espacial EMT** | `trafico_carga_entorno` | `Float64` | Saturación vial media ponderada por inversa de la distancia (*IDW*, `0–126%`) alrededor de la parada. |
| **Cruce Espacial EMT** | `trafico_ocupacion_entorno` | `Float64` | Porcentaje medio ponderado (*IDW*, `0–100%`) de ocupación del carril en el entorno de la parada. |
| **Cruce Espacial EMT** | `trafico_intensidad_entorno` | `Float64` | Flujo vehicular medio ponderado (*IDW*, `veh/h`) en el entorno inmediato de la parada. |
| **Cruce Espacial EMT** | `trafico_nivel_servicio_max` | `Float64` | Peor nivel de servicio semafórico (`0` a `3`) registrado dentro del radio operativo de la parada. |
| **Cruce Espacial EMT** | `trafico_regimen` | `String` | Régimen de tráfico categórico del entorno (`"Fluido"`, `"Saturado_Con_Movimiento"`, `"Colapso_Estatico"`, `"Libre"`). |
| **Cruce Espacial EMT** | `trafico_indice_friccion` | `Float64` | Índice de fricción vial ponderado (`(Ocupación × Carga) / max(Intensidad, 1)`). |
| **Cruce Espacial EMT** | `trafico_presion_entorno` | `Float64` | Presión de congestión ponderada con decaimiento espacial reforzado a corto alcance ($\le 250\text{ m}$). |
