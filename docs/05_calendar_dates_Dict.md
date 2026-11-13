# 📘 Diccionario de datos de `calendar_dates`

A continuación se incluye un resumen para referencia rápida:

| Columna | Tipo de Dato (Pandas) | Descripción de Negocio | Rol en el Modelo / Importancia |
| :--- | :--- | :--- | :--- |
| **`service_id`** | `String` a `string[pyarrow]` | Identificador alfabético o numérico mnemotécnico que parametriza el patrón semanal regular modificado (e.g., `LA`: Laborables, `841`: Línea especial). | **Clave Primaria / Foránea**. Indexador temporal compuesto; mapea la regla relacional de origen para interceptar y alterar los viajes comerciales. |
| **`date`** | `Int32` a `timestamp[us][pyarrow]` | Fecha de calendario específica en la que se ejecuta la alteración del servicio, cargada como entero comprimido (`AAAAMMDD`) y transformada a marca de tiempo. | **Descriptor Cronológico**. Variable temporal raíz; delimita el día exacto de la excepción y actúa como la base para el parseo y validación de vigencia. |
| **`exception_type`** | `Int64` a `int8[pyarrow]` | Código numérico estandarizado que parametriza la regla de negocio de la excepción (`1`: Adición de servicio, `2`: Supresión de servicio). | **Indicador de Modificación**. Variable categórica predictora; delata si la línea opera bajo un refuerzo excepcional o si el trayecto regular se cancela. |