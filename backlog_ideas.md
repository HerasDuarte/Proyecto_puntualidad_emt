# 🚀 Backlog de Ideas y Feature Engineering Avanzado

Este documento opera como el registro estratégico de hipótesis de negocio e ingeniería de características complejas identificadas durante las fases exploratorias (EDA). Estas ideas quedan congeladas temporalmente para preservar el aislamiento de los pipelines de limpieza y se reactivarán de forma unificada durante la fase de integración relacional y entrenamiento del modelo predictivo.

---

## 📌 [IDE-001] Segmentación de Paradas Espejo por Sentido de Marcha (`direction_id`)

* **Origen de la Idea:** Identificado durante la auditoría visual de dispersión cartográfica en el Notebook de `stops`. Se observa la presencia recurrente de paradas gemelas físicamente enfrentadas en aceras opuestas que comparten el mismo literal nominal (`stop_name`) pero operan bajo identificadores únicos (`stop_id`) independientes.

* **Fase de Activación:** Fase de Integración y Enriquecimiento Relacional (Post-limpieza de todas las tablas GTFS).

### Justificación de Negocio e Insight Analítico:
El transporte de superficie de la EMT opera bajo patrones de flujo pendular altamente asimétricos según la hora punta. Por ejemplo, en una gran avenida lineal (como la Calle de Alcalá o el Paseo de la Castellana), las marquesinas de la acera de entrada al centro absorben la máxima congestión de tráfico y detenciones por transbordo durante la franja matinal, mientras que las paradas de la acera opuesta (salida hacia la periferia) registran un flujo continuo y estable a esa misma hora. Agrupar ambas aceras bajo el mismo nombre sin discriminar el sentido induce un sesgo analítico de promediado que diluye la verdadera varianza del tráfico.

### Propuesta de Implementación Técnica:
1. **Bloqueo de Cruce Actual:** Mantener la tabla `stops` aislada en su fase de limpieza actual para evitar la contaminación con el archivo `trips.txt` que aún se encuentra en estado bruto.
2. **Estrategia Relacional Posterior:** Una vez saneado el archivo de viajes, realizar un cruce relacional (*merge*) multi-tabla intermedio: `stops` ➡️ `stop_times` ➡️ `trips`.
3. **Estructura del Dato:** Heredar e inyectar de forma binaria el campo original **`direction_id`** (Tipo de dato: `Integer / int64`), codificado estrictamente bajo el estándar GTFS:
   * **`0`**: Trayecto de Ida (Eje de absorción pendular matinal/central).

   * **`1`**: Trayecto de Vuelta (Eje de dispersión residencial/periférica).

### Impacto Esperado en el Modelo Predictivo:
Aportará al algoritmo de Machine Learning (XGBoost/LightGBM) una variable predictora de contexto crítico con alta densidad de información. El modelo podrá aislar el riesgo de retraso de un viaje condicionándolo no solo a la hora del día y la parada literal, sino al vector específico de la acera en la que se realiza la detención, eliminando el ruido estadístico de comportamientos erráticos y disparando la precisión del cálculo de puntualidad.

## 📌 [IDE-002] Normalización del Índice de Sinuosidad en Líneas Circulares Reales

* **Origen de la Idea:** Identificado durante la auditoría del resumen estadístico descriptivo en el Notebook de `shapes`. Se observa que las líneas con topología circular o bucles cerrados (donde la cabecera origen y destino coinciden geográficamente en el mismo espacio) registran distancias rectas próximas a cero, distorsionando artificialmente el valor máximo del `indice_sinuosidad` en esta fase aislada.

* **Fase de Activación:** Fase de Integración Relacional Multi-Tabla (Post-limpieza y normalización de la tabla `routes.txt` y `trips.txt`).

### Justificación y Resolución Técnica:
Para evitar el uso de umbrales métricos arbitrarios e inestables que induzcan a errores en líneas lineales cortas (falsos positivos), mantenemos el cálculo de sinuosidad directa y transparente durante el procesamiento aislado de `shapes`. El tratamiento y corrección definitiva de este sesgo se pospone a la fase de integración del modelo: una vez saneada la tabla `routes.txt`, cruzaremos las trayectorias para identificar unívocamente aquellas líneas tipificadas explícitamente por las reglas de negocio de la EMT como circulares. Para estas rutas específicas, el denominador de la sinuosidad se recalculará sustituyendo la distancia origen-destino por el diámetro máximo real de dispersión del vector (apogeo geométrico), garantizando un descriptor homogéneo y libre de anomalías para el algoritmo de Machine Learning.

## 📌 [IDE-003] Ingeniería de Características Temporales Cíclicas a partir de la Matriz de Calendario

* **Origen de la Idea:** Identificado durante la caracterización estadística del Notebook de `calendar`. Las banderas booleanas semanales de operación constituyen la base para modelar el comportamiento del tráfico pendular en Madrid.

* **Fase de Activación:** Fase de Integración Relacional y Enriquecimiento de la Matriz Maestra de Viajes (Post-limpieza de `trips` y `stop_times`).

### Justificación de Negocio e Hipótesis:
El comportamiento de la congestión circulatoria y, por ende, los retrasos en las líneas de la EMT de Madrid, responden a ciclos semanales rígidos vinculados a la actividad socioeconómica. Un martes a las 08:00 AM experimenta picos de saturación masivos debido a la movilidad corporativa pendular; por el contrario, un domingo a la misma hora exhibe un flujo libre. Aislar estas realidades temporales es indispensable para que el algoritmo asimile las fluctuaciones de la varianza horaria.

### Propuestas de Implementación Técnica en la Matriz General:

1. **Inyección de la Carga de Demanda Comercial (`tipo_jornada`):**
   Mapear las claves categóricas extraídas de `service_id` para colapsar las banderas en una variable cualitativa simplificada de tres niveles:
   * `LABORABLE` (Ejes asociados a `LA` y `LJ`): Máximo estrés circulatorio.

   * `VIERNES_VISPERA` (Asociado a `VV`): Patrón asimétrico de salida de la capital con congestión adelantada al mediodía.

   * `FIN_DE_SEMANA` (Asociado a `SA` y `FE`): Malla fluida de baja fricción vehicular.

2. **Codificación Cíclica de la Estacionalidad Anual (Seno/Coseno de Fecha):**
   Utilizando las columnas saneadas `start_date` y `end_date`, el modelo transformará el componente temporal del viaje en coordenadas circulares continuas. Al calcular el `np.sin()` y `np.cos()` del día del año, el algoritmo de Machine Learning comprenderá de forma nativa que el 31 de diciembre y el 1 de enero están cronológicamente adyacentes, permitiéndole asimilar patrones estacionales (como la caída de tráfico en vacaciones navideñas o el aumento de retrasos por inclemencias climáticas invernales en Madrid) de manera puramente matemática.

## 📌 [IDE-004] Mitigación del Sesgo por Escasez de Histórico en Líneas Residuales (Series 700/800)

* **Origen de la Idea:** Identificado durante la auditoría forense de la cola de la distribución en `trips.ipynb`. Las líneas de servicios especiales temporales registran volumetrías críticamente bajas (< 30 expediciones totales por trimestre).
* **Fase de Activación:** Fase de Entrenamiento del Modelo Predictivo y Selección de Algoritmos.

### Justificación de Negocio e Hipótesis:
Líneas de la EMT de Madrid como la 704 o la 718 sufren de una escasez severa de observaciones históricas para que los algoritmos asimilen sus patrones de tráfico. Intentar entrenar un modelo aislado por ruta para estos pasillos inducirá a un severo sobreajuste (*overfitting*) o a un colapso en la varianza predictiva.

### Propuesta de Implementación Técnica:
Para blindar la estabilidad de las predicciones en líneas de cobertura mínima, se descarta el enfoque de modelos unilineales compartimentados. En su lugar, el pipeline implementará un **Algoritmo Global de Árboles de Decisión Generalizados (e.g., LightGBM o XGBoost)**. Al inyectar la variable de ingeniería `volumen_viajes_bloque` junto con los metadatos cruzados de `routes` y `shapes`, las líneas masivas de alta frecuencia transferirán el aprendizaje de las leyes generales del callejero madrileño hacia las rutas residuales, garantizando estimaciones de puntualidad estables para la totalidad del catálogo.

## 📌 [IDE-005] Reproducibilidad e Ingesta Automatizada
- [ ] **Script de descarga automática del catálogo GTFS (`scripts/descargar_gtfs.py`):** Crear un script que descargue automáticamente el paquete oficial `gtfs_emt.zip` desde el portal Open Data de la EMT de Madrid y lo descomprima en `data/raw/gtfs/`, permitiendo reproducir el pipeline completo desde cero tras clonar el repositorio sin subir ficheros pesados a GitHub.
