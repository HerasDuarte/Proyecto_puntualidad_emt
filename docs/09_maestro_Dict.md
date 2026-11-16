# 📘 Diccionario de datos de `maestro`

A continuación se incluye un resumen para referencia rápida:

| Bloque Origen | Variable | Tipo de Dato | Descripción Operativa y Regla de Negocio |
| :--- | :--- | :--- | :--- |
| **Malla Horaria** (`stop_times`) | `trip_id` | `String` | Identificador único de la expedición comercial (`route_service_trip`). |
| **Malla Horaria** (`stop_times`) | `arrival_time` | `String` | Hora programada de llegada del autobús a la marquesina (`HH:MM:SS`). |
| **Malla Horaria** (`stop_times`) | `departure_time` | `String` | Hora programada de salida desde la marquesina (`HH:MM:SS`). |
| **Malla Horaria** (`stop_times`) | `stop_id` | `String` | Identificador único de la parada física en la red EMT. |
| **Malla Horaria** (`stop_times`) | `stop_sequence` | `Int32` | Orden correlativo de paso de la parada dentro del viaje (`1, 2, ..., N`). |
| **Malla Horaria** (`stop_times`) | `shape_dist_traveled` | `Int32` | Distancia acumulada en metros desde la cabecera de origen hasta la parada. |
| **Malla Horaria** (`stop_times`) | `tiempo_tramo_teorico_seg` | `Int32` | Duración planificada en segundos desde la parada inmediatamente anterior. |
| **Malla Horaria** (`stop_times`) | `distancia_tramo_m` | `Int32` | Longitud real del tramo en metros entre la parada anterior y la actual. |
| **Malla Horaria** (`stop_times`) | `hora_llegada_decimal` | `Float32` | Hora de llegada convertida a formato numérico continuo (`0.0` a `30.0+`). |
| **Malla Horaria** (`stop_times`) | `progreso_viaje_pct` | `Float32` | Fracción normalizada de avance del trayecto (`0.0` en origen a `1.0` en destino). |
| **Malla Horaria** (`stop_times`) | `densidad_paradas_acumuladas` | `Float32` | Ratio de paradas acumuladas por kilómetro recorrido en el viaje. |
| **Expediciones** (`trips`) | `route_id` | `String` | Código identificador normalizado a 3 dígitos de la línea (`001`, `203`, etc.). |
| **Expediciones** (`trips`) | `service_id` | `String` | Patrón de calendario operativo semanal (`LA`, `SA`, `FE`, `VV`, `LJ`). |
| **Expediciones** (`trips`) | `trip_headsign` | `String` | Letrero luminoso frontal de destino que muestra el autobús. |
| **Expediciones** (`trips`) | `direction_id` | `Int8` | Sentido de marcha del trayecto (`0`: Ida / `1`: Vuelta). |
| **Expediciones** (`trips`) | `block_id` | `String` | Identificador del turno físico del vehículo encadenado durante la jornada. |
| **Expediciones** (`trips`) | `shape_id` | `String` | Identificador del trazado geométrico asignado a la expedición. |
| **Expediciones** (`trips`) | `volumen_viajes_bloque` | `Int32` | Número total de expediciones encadenadas dentro del mismo turno (`block_id`). |
| **Líneas** (`routes`) | `route_short_name` | `String` | Denominación comercial pública de la línea (`1`, `27`, `C1`, `N1`, `SE721`). |
| **Líneas** (`routes`) | `cabecera_ruta` | `String` | Nombre normalizado de la terminal de origen de la línea. |
| **Líneas** (`routes`) | `destino_ruta` | `String` | Nombre normalizado de la terminal de destino de la línea. |
| **Líneas** (`routes`) | `regimen_horario` | `String` | Clasificación operativa (`Servicio Diurno Regular`, `Servicio Nocturno (Búho)`, `Servicio 24 Horas`). |
| **Infraestructura** (`stops`) | `stop_code` | `String` | Código público visible en el poste/marquesina para información al viajero. |
| **Infraestructura** (`stops`) | `stop_name` | `String` | Denominación oficial de la parada en el callejero de Madrid. |
| **Infraestructura** (`stops`) | `stop_lat` | `Float32` | Latitud geográfica en sistema `WGS84` (`EPSG:4326`). |
| **Infraestructura** (`stops`) | `stop_lon` | `Float32` | Longitud geográfica en sistema `WGS84` (`EPSG:4326`). |
| **Infraestructura** (`stops`) | `calle_principal` | `String` | Arteria vial principal extraída mediante expresiones regulares (`Regex`). |
| **Infraestructura** (`stops`) | `en_alta_densidad_real` | `Boolean` | Indicador espacial de contención dentro del perímetro oficial de la **M-30**. |
| **Infraestructura** (`stops`) | `ubicacion_territorial` | `String` | Etiqueta territorial (`Interior M-30 (Alta Densidad)` / `Exterior M-30 (Periferia)`). |
| **Geometría** (`shapes`) | `longitud_total_ruta_m` | `Int32` | Distancia total en metros del trazado geométrico completo de la ruta. |
| **Geometría** (`shapes`) | `indice_sinuosidad` | `Float32` | Ratio entre la distancia real por calle y la distancia geodésica en línea recta. |
| **Geometría** (`shapes`) | `puntos_por_km` | `Float32` | Densidad de vértices GPS por kilómetro que definen la curvatura del trazado. |
| **Frecuencias** (`frequencies`) | `headway_medio_secs` | `Float32` | Intervalo medio programado en segundos entre autobuses consecutivos. |
| **Frecuencias** (`frequencies`) | `headway_min_secs` | `Int32` | Intervalo mínimo en segundos alcanzado en la franja de máxima frecuencia. |
| **Frecuencias** (`frequencies`) | `autobuses_hora_medio` | `Float32` | Oferta media de expediciones por hora de la línea a lo largo del día. |
| **Frecuencias** (`frequencies`) | `autobuses_hora_max` | `Float32` | Oferta máxima de expediciones por hora en hora punta para esa línea. |
| **Frecuencias** (`frequencies`) | `cobertura_total_franjas_secs` | `Int32` | Amplitud horaria diaria acumulada en segundos en la que opera la línea. |
| **Feature Eng. Relacional** | `hora_entera` | `Int32` | Hora del reloj en punto normalizada (`0` a `23`) a partir de `hora_llegada_decimal`. |
| **Feature Eng. Relacional** | `tramo_15min` | `Int32` | Ventana operativa de 15 minutos del día (`0` a `95`) para alta resolución temporal. |
| **Feature Eng. Relacional** | `tipo_jornada` | `String` | Régimen semanal de demanda (`LABORABLE`, `VIERNES_VISPERA`, `FIN_DE_SEMANA`). |
| **Feature Eng. Relacional** | `indice_sinuosidad_normalizado` | `Float32` | Índice de sinuosidad acotado al P95 lineal (`2,506`) en rutas circulares reales. |
| **Feature Eng. Relacional** | `velocidad_teorica_kmh` | `Float32` | Velocidad comercial programada del tramo (`km/h`), acotada a `[0, 80]`. |
| **Feature Eng. Relacional** | `saturacion_marquesina_15min` | `Int32` | Expediciones simultáneas en el mismo poste, sentido, jornada y tramo de 15 min. |
| **Feature Eng. Relacional** | `friccion_acumulada_m30` | `Float32` | Arrastre progresivo dentro de la Almendra Central (`en_alta_densidad_real` × `progreso_viaje_pct`). |
| **Feature Eng. Relacional** | `hora_seno` | `Float32` | Componente seno de la proyección trigonométrica cíclica de las 24 horas del día. |
| **Feature Eng. Relacional** | `hora_coseno` | `Float32` | Componente coseno de la proyección trigonométrica cíclica de las 24 horas del día. |