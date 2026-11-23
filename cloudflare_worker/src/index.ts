import { PARADAS_ESTRATIFICADAS } from "./paradas";

export interface Env {
  DB: D1Database;
  EMT_CLIENT_ID: string;
  EMT_PASSKEY: string;
  SYNC_SECRET_TOKEN: string;
  AEMET_API_KEY: string;
}

const BASE_URL_EMT = "https://" + "openapi." + "emtmadrid.es/v2";
const URL_TRAFICO_PRIMARIA = "https://" + "informo." + "madrid.es/informo/tmadrid/pm.xml";
const URL_TRAFICO_RESPALDO = "https://" + "datos." + "madrid.es/egob/catalogo/202087-0-trafico-intensidad.xml";
const URL_OPEN_METEO =
  "https://" + "api." + "://open-meteo.com" +
  "&current=temperature_2m,apparent_temperature,relative_humidity_2m,is_day," +
  "precipitation,rain,snowfall,visibility,weather_code,wind_speed_10m,wind_gusts_10m" +
  "&timezone=Europe%2FMadrid";

const LIMITE_DIARIO_EMT = 4900;
const TAMANO_LOTE_EMT = 10;

async function comprimirGzipBase64(texto: string): Promise<string> {
  const bytes = new TextEncoder().encode(texto);
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream("gzip"));
  const buffer = await new Response(stream).arrayBuffer();
  const uint8 = new Uint8Array(buffer);
  let binario = "";
  const chunk = 8192;
  for (let i = 0; i < uint8.length; i += chunk) {
    binario += String.fromCharCode(...uint8.subarray(i, i + chunk));
  }
  return btoa(binario);
}

async function capturarMeteorologia(env: Env, ahoraIso: string): Promise<string> {
  const cabeceras = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ProyectoPuntualidadEMT/1.0",
    "Accept": "application/json",
  };

  // 1. Fuente Primaria: AEMET OpenData - Estación 3195 (Madrid-Retiro, observación cada 10 min)
  if (env.AEMET_API_KEY) {
    try {
      const urlMeta =
        "https://" + "opendata." + "aemet.es/opendata/api/observacion/convencional/datos/estacion/3195" +
        `?api_key=${encodeURIComponent(env.AEMET_API_KEY)}`;
      const respMeta = await fetch(urlMeta, { headers: cabeceras });
      if (respMeta.ok) {
        const meta: any = await respMeta.json();
        if (meta?.estado === 200 && meta?.datos) {
          const respDatos = await fetch(meta.datos, { headers: cabeceras });
          if (respDatos.ok) {
            const listaObs: any = await respDatos.json();
            const obs = Array.isArray(listaObs) && listaObs.length > 0 ? listaObs.at(-1) : null;
            if (obs) {
              const precMm = obs.prec != null ? Number(obs.prec) : 0;
              const vientoKmh = obs.vv != null ? Number(obs.vv) * 3.6 : null;
              const rachasKmh = obs.vmax != null ? Number(obs.vmax) * 3.6 : null;
              const visKm = obs.vis != null ? Number(obs.vis) : null;
              const horaLocal = new Date().getHours();

              await env.DB.prepare(
                `INSERT INTO meteo_madrid (
                  timestamp_captura_utc, timestamp_api_madrid, temperatura_c, sensacion_termica_c,
                  humedad_relativa_pct, es_de_dia, precipitacion_mm, lluvia_mm, nevadas_cm,
                  visibilidad_m, codigo_tiempo_wmo, viento_velocidad_kmh, viento_rachas_kmh
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
              )
                .bind(
                  ahoraIso,
                  obs.fint ?? ahoraIso,
                  obs.ta != null ? Number(obs.ta) : null,
                  obs.ta != null ? Number(obs.ta) : null,
                  obs.hr != null ? Number(obs.hr) : null,
                  horaLocal >= 7 && horaLocal <= 20 ? 1 : 0,
                  precMm,
                  precMm,
                  0,
                  visKm != null ? visKm * 1000 : 10000,
                  null,
                  vientoKmh,
                  rachasKmh
                )
                .run();
              return `METEO OK AEMET-Retiro (${obs.ta}°C, prec=${precMm}mm, obs=${obs.fint})`;
            }
          }
        }
      }
    } catch (err) {
      console.warn("Aviso AEMET, activando respaldo 7Timer:", err);
    }
  }

  // 2. Respaldo automático: 7Timer!
  const url7Timer = "https://" + "www." + "7timer.info/bin/civil.php?lon=-3.7038&lat=40.4168&ac=0&unit=metric&output=json&tzshift=0";
  const resp7 = await fetch(url7Timer, { headers: cabeceras });
  if (!resp7.ok) throw new Error(`HTTP 7Timer: ${resp7.status}`);
  const d7: any = await resp7.json();
  const serie = Array.isArray(d7?.dataseries) ? d7.dataseries : [];
  const p = serie.at(0) || {};
  const precTipo = String(p.prec_type || "none");
  const lluviaMm = precTipo === "rain" ? Math.max(0.2, Number(p.prec_amount || 0) * 0.5) : 0;
  const nieveCm = precTipo === "snow" ? Math.max(0.2, Number(p.prec_amount || 0) * 0.5) : 0;
  const vientoKmh = p.wind10m?.speed != null ? Number(p.wind10m.speed) * 3.6 : null;
  const horaLocal = new Date().getHours();

  await env.DB.prepare(
    `INSERT INTO meteo_madrid (
      timestamp_captura_utc, timestamp_api_madrid, temperatura_c, sensacion_termica_c,
      humedad_relativa_pct, es_de_dia, precipitacion_mm, lluvia_mm, nevadas_cm,
      visibilidad_m, codigo_tiempo_wmo, viento_velocidad_kmh, viento_rachas_kmh
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
  )
    .bind(
      ahoraIso,
      d7?.init ?? ahoraIso,
      p.temp2m != null ? Number(p.temp2m) : null,
      p.temp2m != null ? Number(p.temp2m) : null,
      p.rh2m != null ? Number(String(p.rh2m).replace("%", "")) : null,
      horaLocal >= 7 && horaLocal <= 20 ? 1 : 0,
      lluviaMm + nieveCm,
      lluviaMm,
      nieveCm,
      10000,
      p.cloudcover != null ? Number(p.cloudcover) : null,
      vientoKmh,
      null
    )
    .run();

  return `METEO OK 7Timer-Backup (${p.temp2m}°C, prec=${precTipo})`;
}


async function capturarTrafico(env: Env, ahoraIso: string): Promise<string> {
  const cabeceras = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/xml, text/xml, */*;q=0.9",
    "Accept-Language": "es-ES,es;q=0.9",
    "Referer": "https://madrid.es",
  };

  let resp = await fetch(URL_TRAFICO_PRIMARIA, { headers: cabeceras });
  if (!resp.ok) {
    resp = await fetch(URL_TRAFICO_RESPALDO, { headers: cabeceras });
  }
  if (!resp.ok) throw new Error(`HTTP Tráfico Madrid: ${resp.status}`);

  const xmlTexto = await resp.text();
  const comprimidoB64 = await comprimirGzipBase64(xmlTexto);

  await env.DB.prepare(
    `INSERT INTO trafico_xml_raw (timestamp_captura_utc, xml_gzip_base64, tamano_bytes_original)
     VALUES (?, ?, ?)`
  )
    .bind(ahoraIso, comprimidoB64, xmlTexto.length)
    .run();

  return `TRAFICO OK (${xmlTexto.length} bytes -> gzip)`;
}

async function obtenerTokenEMT(env: Env): Promise<string> {
  const resp = await fetch(`${BASE_URL_EMT}/mobilitylabs/user/login/`, {
    headers: {
      "X-ClientId": env.EMT_CLIENT_ID,
      passKey: env.EMT_PASSKEY,
      Accept: "application/json",
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ProyectoPuntualidadEMT/1.0",
    },
  });
  const texto = await resp.text();
  let datos: any;
  try {
    datos = JSON.parse(texto);
  } catch {
    throw new Error(`Login EMT respuesta no JSON (${resp.status}): ${texto.slice(0, 120)}`);
  }
  const lista = Array.isArray(datos?.data) ? datos.data : [];
  const primero = lista.at(0);
  if (!primero?.accessToken) {
    throw new Error(`Fallo login EMT (code=${datos?.code}): ${JSON.stringify(datos).slice(0, 150)}`);
  }
  return primero.accessToken;
}

async function capturarLoteEMT(env: Env, ahora: Date): Promise<string> {
  const hoyUtc = ahora.toISOString().slice(0, 10);
  const ahoraIso = ahora.toISOString();

  let estado: any = await env.DB.prepare("SELECT * FROM estado_cuota WHERE id = 1").first();
  if (!estado || estado.fecha_cuota_utc !== hoyUtc) {
    const nuevoSessionId = `SES_CF_${hoyUtc.replace(/-/g, "")}`;
    const indicePrevio = estado ? Number(estado.indice_parada_actual) : 0;
    await env.DB.prepare(
      `INSERT INTO estado_cuota (
        id, fecha_cuota_utc, session_id, peticiones_realizadas,
        indice_parada_actual, en_pausa_cuota, access_token, token_expira_utc, ultima_actualizacion_utc
      ) VALUES (1, ?, ?, 0, ?, 0, NULL, NULL, ?)
      ON CONFLICT(id) DO UPDATE SET
        fecha_cuota_utc = excluded.fecha_cuota_utc,
        session_id = excluded.session_id,
        peticiones_realizadas = 0,
        en_pausa_cuota = 0,
        ultima_actualizacion_utc = excluded.ultima_actualizacion_utc`
    )
      .bind(hoyUtc, nuevoSessionId, indicePrevio, ahoraIso)
      .run();
    estado = await env.DB.prepare("SELECT * FROM estado_cuota WHERE id = 1").first();
  }

  if (Number(estado.en_pausa_cuota) === 1 || Number(estado.peticiones_realizadas) >= LIMITE_DIARIO_EMT) {
    return `EMT STANDBY (${estado.peticiones_realizadas}/${LIMITE_DIARIO_EMT})`;
  }

  let token = estado.access_token;
  const expiraMs = estado.token_expira_utc ? Date.parse(estado.token_expira_utc) : 0;
  if (!token || Date.now() >= expiraMs - 300_000) {
    token = await obtenerTokenEMT(env);
    const nuevaExpira = new Date(Date.now() + 4 * 3600 * 1000).toISOString();
    await env.DB.prepare(
      "UPDATE estado_cuota SET access_token = ?, token_expira_utc = ? WHERE id = 1"
    )
      .bind(token, nuevaExpira)
      .run();
  }

  const restantes = LIMITE_DIARIO_EMT - Number(estado.peticiones_realizadas);
  const nConsultas = Math.min(TAMANO_LOTE_EMT, restantes);
  const totalParadas = PARADAS_ESTRATIFICADAS.length;
  let idx = Number(estado.indice_parada_actual) % totalParadas;

  const consultas = Array.from({ length: nConsultas }, (_, k) => {
    const stopId = PARADAS_ESTRATIFICADAS[(idx + k) % totalParadas];
    return fetch(`${BASE_URL_EMT}/transport/busemtmad/stops/${stopId}/arrives/`, {
      method: "POST",
      headers: {
        accessToken: token,
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ProyectoPuntualidadEMT/1.0",
      },
      body: JSON.stringify({
        cultureInfo: "ES",
        Text_StopRequired_YN: "N",
        Text_EstimationsRequired_YN: "Y",
        Text_IncidencesRequired_YN: "N",
        DateTime_Referenced_Incidencies_YYYYMMDD: "",
      }),
    })
      .then((r) => r.json())
      .then((json: any) => {
        const lista = Array.isArray(json?.data) ? json.data : [];
        const primero = lista.at(0);
        return { stopId, arrives: primero?.Arrive || [] };
      })
      .catch(() => ({ stopId, arrives: [] }));
  });

  const resultados = await Promise.all(consultas);
  const inserts: D1PreparedStatement[] = [];
  const stmtInsert = env.DB.prepare(
    `INSERT INTO emt_llegadas (
      session_id, timestamp_captura_utc, stop_id, line, destination,
      bus_id, estimate_arrive_sec, distance_bus_m, position_type_bus, is_head
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
  );

  for (const res of resultados) {
    for (const bus of res.arrives) {
      inserts.push(
        stmtInsert.bind(
          estado.session_id,
          ahoraIso,
          String(res.stopId),
          String(bus.line ?? ""),
          String(bus.destination ?? ""),
          String(bus.bus ?? ""),
          bus.estimateArrive != null ? Number(bus.estimateArrive) : null,
          bus.DistanceBus != null ? Number(bus.DistanceBus) : null,
          bus.positionTypeBus != null ? String(bus.positionTypeBus) : null,
          bus.isHead != null ? String(bus.isHead) : null
        )
      );
    }
  }

  if (inserts.length > 0) {
    await env.DB.batch(inserts);
  }

  const nuevasReqs = Number(estado.peticiones_realizadas) + nConsultas;
  const nuevoIdx = (idx + nConsultas) % totalParadas;
  const enPausa = nuevasReqs >= LIMITE_DIARIO_EMT ? 1 : 0;

  await env.DB.prepare(
    `UPDATE estado_cuota
     SET peticiones_realizadas = ?, indice_parada_actual = ?, en_pausa_cuota = ?, ultima_actualizacion_utc = ?
     WHERE id = 1`
  )
    .bind(nuevasReqs, nuevoIdx, enPausa, ahoraIso)
    .run();

  return `EMT OK (${nuevasReqs}/${LIMITE_DIARIO_EMT} reqs | ${inserts.length} llegadas)`;
}

export default {
  async scheduled(event: ScheduledEvent, env: Env, ctx: ExecutionContext): Promise<void> {
    const ahora = new Date();
    const ahoraIso = ahora.toISOString();

    const tareas: Promise<string>[] = [];
    if (event.cron === "*/3 * * * *") tareas.push(capturarLoteEMT(env, ahora));
    if (event.cron === "*/5 * * * *") tareas.push(capturarTrafico(env, ahoraIso));
    if (event.cron === "*/15 * * * *") tareas.push(capturarMeteorologia(env, ahoraIso));

    const resultados = await Promise.allSettled(tareas);
    const resumen = resultados.map((r) =>
      r.status === "fulfilled" ? { ok: r.value } : { error: String(r.reason?.message || r.reason) }
    );
    console.log(`[${ahoraIso}] [${event.cron}]`, JSON.stringify(resumen));
  },

  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const token = url.searchParams.get("token");

    if (url.pathname === "/api/export" || url.pathname === "/api/purge") {
      if (!env.SYNC_SECRET_TOKEN || token !== env.SYNC_SECRET_TOKEN) {
        return new Response("No autorizado", { status: 401 });
      }
    }

    if (url.pathname === "/api/export") {
      const tabla = url.searchParams.get("table") || "emt_llegadas";
      const permitidas = ["emt_llegadas", "trafico_xml_raw", "meteo_madrid"];
      if (!permitidas.includes(tabla)) {
        return new Response("Tabla no válida", { status: 400 });
      }
      const limite = tabla === "trafico_xml_raw" ? 30 : 5000;
      const rows = await env.DB.prepare(`SELECT * FROM ${tabla} ORDER BY id ASC LIMIT ?`)
        .bind(limite)
        .all();
      return Response.json({ table: tabla, count: rows.results.length, rows: rows.results });
    }

    if (url.pathname === "/api/purge" && request.method === "POST") {
      const body: any = await request.json();
      const tabla = body.table;
      const maxId = Number(body.max_id);
      const permitidas = ["emt_llegadas", "trafico_xml_raw", "meteo_madrid"];
      if (!permitidas.includes(tabla) || !Number.isFinite(maxId) || maxId <= 0) {
        return new Response("Parámetros no válidos", { status: 400 });
      }
      const res = await env.DB.prepare(`DELETE FROM ${tabla} WHERE id <= ?`).bind(maxId).run();
      return Response.json({ deleted: true, table: tabla, max_id: maxId, meta: res.meta });
    }

        if (url.pathname === "/api/live") {
      const estadoLive: any = await env.DB.prepare(
        "SELECT fecha_cuota_utc, peticiones_realizadas, indice_parada_actual, en_pausa_cuota, ultima_actualizacion_utc FROM estado_cuota WHERE id = 1"
      ).first();

      const meteoActual: any = await env.DB.prepare(
        "SELECT timestamp_captura_utc, temperatura_c, sensacion_termica_c, humedad_relativa_pct, es_de_dia, precipitacion_mm, lluvia_mm, visibilidad_m, codigo_tiempo_wmo, viento_velocidad_kmh FROM meteo_madrid ORDER BY id DESC LIMIT 1"
      ).first();

      const traficoActual: any = await env.DB.prepare(
        "SELECT id, timestamp_captura_utc, tamano_bytes_original FROM trafico_xml_raw ORDER BY id DESC LIMIT 1"
      ).first();

      const resumenLineas: any = await env.DB.prepare(
        `SELECT line,
                COUNT(*) AS lecturas,
                ROUND(AVG(estimate_arrive_sec) / 60.0, 1) AS espera_media_min,
                ROUND(MAX(estimate_arrive_sec) / 60.0, 1) AS espera_max_min
         FROM emt_llegadas
         WHERE id > (SELECT COALESCE(MAX(id) - 300, 0) FROM emt_llegadas)
           AND line IS NOT NULL
         GROUP BY line
         ORDER BY espera_media_min DESC
         LIMIT 10`
      ).all();

      const ultimasLlegadas: any = await env.DB.prepare(
        "SELECT timestamp_captura_utc, stop_id, line, destination, bus_id, estimate_arrive_sec, distance_bus_m, is_head FROM emt_llegadas ORDER BY id DESC LIMIT 25"
      ).all();

      return new Response(
        JSON.stringify({
          generado_utc: new Date().toISOString(),
          atribucion: "Powered by EMT de Madrid (http://emtmadrid.es)",
          estado_cuota_emt: estadoLive || null,
          meteo_actual: meteoActual || null,
          trafico_actual: traficoActual || null,
          top_lineas_espera: resumenLineas.results || [],
          ultimas_llegadas: ultimasLlegadas.results || [],
        }),
        {
          headers: {
            "Content-Type": "application/json; charset=utf-8",
            "Cache-Control": "public, max-age=120",
          },
        }
      );
    }

    const estado: any = await env.DB.prepare("SELECT id, fecha_cuota_utc, session_id, peticiones_realizadas, indice_parada_actual, en_pausa_cuota, token_expira_utc, ultima_actualizacion_utc FROM estado_cuota WHERE id = 1").first();
    const cEmt: any = await env.DB.prepare("SELECT COUNT(*) as total FROM emt_llegadas").first();
    const cTraf: any = await env.DB.prepare("SELECT COUNT(*) as total FROM trafico_xml_raw").first();
    const cMet: any = await env.DB.prepare("SELECT COUNT(*) as total FROM meteo_madrid").first();

    return Response.json({
      servicio: "Orquestador Tiempo Real EMT Madrid (Cloudflare Workers)",
      atribucion: "Powered by EMT de Madrid (http://emtmadrid.es)",
      estado_cuota_emt: estado || "Esperando primer cron...",
      filas_pendientes_descarga: {
        emt_llegadas: cEmt?.total ?? 0,
        trafico_xml_raw: cTraf?.total ?? 0,
        meteo_madrid: cMet?.total ?? 0,
      },
    });
  },
};
