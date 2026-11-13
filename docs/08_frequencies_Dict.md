# 📘 Diccionario de datos de `frequencies`

A continuación se incluye un resumen para referencia rápida:

  
| Columna | Tipo de Dato (Pandas) | Descripción de Negocio | Rol en el Modelo / Importancia |
| :--- | :--- | :--- | :--- |
| **`trip_id`** | `String` a `string[pyarrow]` | Identificador único de la expedición o viaje individual al que aplica la regla de frecuencia. | **Clave Primaria / Foránea**. Vincula la cadencia teórica de paso con la expedición en `trips_limpio.csv`. |
| **`start_time`** | `String` a `string[pyarrow]` | Hora de inicio de vigencia de la franja de frecuencia (`H:MM:SS` o `HH:MM:SS`, hasta `30:00:00`). | **Descriptor Temporal Bruto**. Marca el arranque del intervalo; se transforma a segundos absolutos para cálculo matemático. |
| **`end_time`** | `String` a `string[pyarrow]` | Hora de finalización de vigencia de la franja de frecuencia (`H:MM:SS` o `HH:MM:SS`, hasta `30:00:00`). | **Descriptor Temporal Bruto**. Delimita el cierre de la ventana de regularidad del servicio. |
| **`headway_secs`** | `Int64` a `int32[pyarrow]` | Intervalo teórico programado en segundos entre dos autobuses consecutivos de la misma línea en esa franja. | **Variable Predictora**. Métrica base de regularidad; mide el tiempo de espera teórico entre vehículos. |
| **`start_time_secs`** | `Nueva` a `int32[pyarrow]` | Conversión numérica exacta de `start_time` a segundos absolutos transcurridos desde medianoche. | **Descriptor Cronológico**. Permite ordenar y validar franjas horarias nocturnas sin errores alfabéticos. |
| **`end_time_secs`** | `Nueva` a `int32[pyarrow]` | Conversión numérica exacta de `end_time` a segundos absolutos transcurridos desde medianoche. | **Descriptor Cronológico**. Garantiza la verificación matemática de que el cierre sea posterior al inicio. |
| **`autobuses_por_hora`** | `Nueva` a `float32[pyarrow]` | Tasa directa de presión de oferta en el carril calculada como `3600.0 / headway_secs`. | **Feature Engineering / Predictor de Saturación**. Cuantifica el flujo horario de flota (de 1 a 20 buses/h) para modelar el riesgo de efecto acordeón. |
| **`duracion_franja_secs`** | `Nueva` a `int32[pyarrow]` | Amplitud total en segundos de la ventana horaria evaluada (`end_time_secs - start_time_secs + 1`). | **Feature Engineering / Cobertura**. Discrimina entre ventanas estrechas de refuerzo en hora punta frente a bloques valle o nocturnos. |
| **`hora_central_franja`** | `Nueva` a `float32[pyarrow]` | Punto medio cronológico continuo de la ventana operativa (`((start_time_secs + end_time_secs) / 2.0) / 3600.0`). | **Feature Engineering / Estacionalidad**. Sitúa la regla de regularidad dentro del ciclo diario de congestión de Madrid. |