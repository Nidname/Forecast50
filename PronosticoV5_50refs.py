# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from openai import OpenAI
import os

# --------------------------------------------------------------------------
# CONFIGURACIÓN DE LA APP
# --------------------------------------------------------------------------
st.set_page_config(page_title="Pronóstico de la Demanda", layout="centered")

st.markdown("""
    <h1 style='text-align: center; margin-top: -50px;'>Herramienta de Pronóstico de la Demanda</h1>
    """, unsafe_allow_html=True)

st.markdown("""
    ### ¿Cómo te ayudará este pronóstico?
    Esta herramienta te permite anticipar la demanda y optimizar los niveles de inventario, 
    facilitando la toma de decisiones en producción, compras y ventas.
""")

# --------------------------------------------------------------------------
# DESCARGA DE PLANTILLA
# --------------------------------------------------------------------------
st.markdown("""
1. Descarga la plantilla de demanda histórica:  
[📊 Plantilla](https://docs.google.com/spreadsheets/d/1ahEceNTmowyUnXssWv6eY_ftV8FeyZpf/edit?usp=drive_link&ouid=114619176855631654417&rtpof=true&sd=true)

2. Carga el archivo Excel con el historial de la demanda mensual.  
Debe contener al menos 8 meses de datos y la columna **Fecha** en formato dd/mm/aaaa.

3. La hoja debe llamarse **Historico** y tener esta estructura:
- Columna **Fecha**
- Hasta **50 columnas** adicionales (una por referencia), por ejemplo: Ref_1, Ref_2, ..., Ref_50
""")

# --------------------------------------------------------------------------
# CARGA DEL ARCHIVO
# --------------------------------------------------------------------------
uploaded_file = st.file_uploader("📤 Subir archivo Excel", type=["xlsx"])

def _safe_to_datetime(series: pd.Series) -> pd.Series:
    # Intenta parseo robusto (dd/mm/aaaa y variantes). Si falla, intenta parseo general.
    try:
        return pd.to_datetime(series, dayfirst=True, errors="coerce")
    except Exception:
        return pd.to_datetime(series, errors="coerce")

if uploaded_file is not None:
    # ----------------------------------------------------------------------
    # Lectura robusta de la hoja "Historico"
    # ----------------------------------------------------------------------
    try:
        df_his = pd.read_excel(uploaded_file, sheet_name="Historico")
    except Exception:
        st.error("⚠️ No pude leer la hoja 'Historico'. Verifica que el Excel tenga una hoja llamada exactamente 'Historico'.")
        st.stop()

    # Validaciones mínimas
    if "Fecha" not in df_his.columns:
        st.error("⚠️ El archivo debe incluir una columna llamada 'Fecha' en la hoja 'Historico'.")
        st.stop()

    # Detectar columnas de referencias (todas menos Fecha)
    columnas_referencias = [c for c in df_his.columns if str(c).strip().lower() != "fecha"]

    if len(columnas_referencias) == 0:
        st.error("⚠️ No encontré columnas de referencias. Debes tener 'Fecha' y al menos una columna adicional (una por referencia).")
        st.stop()

    # Limitar a máximo 50 referencias
    if len(columnas_referencias) > 50:
        st.warning(f"Encontré {len(columnas_referencias)} referencias. Esta versión soporta hasta 50; usaré las primeras 50.")
        columnas_referencias = columnas_referencias[:50]

    st.write("### Datos Históricos (hoja 'Historico'):")
    st.dataframe(df_his)

    # ----------------------------------------------------------------------
    # Limpieza / indexación por fecha
    # ----------------------------------------------------------------------
    df_his["Fecha"] = _safe_to_datetime(df_his["Fecha"])
    if df_his["Fecha"].isna().any():
        st.error("⚠️ Hay fechas que no se pudieron convertir. Revisa el formato de la columna 'Fecha' (ideal dd/mm/aaaa).")
        st.stop()

    df_his = df_his.sort_values("Fecha").set_index("Fecha")

    # Convertir referencias a numérico (por si vienen como texto) y completar vacíos por tiempo
    for c in columnas_referencias:
        df_his[c] = pd.to_numeric(df_his[c], errors="coerce")
    df_his = df_his.interpolate(method="time")

    if len(df_his) < 8:
        st.error("⚠️ Necesitas al menos 8 meses (filas) de datos para poder pronosticar.")
        st.stop()

    st.markdown("### ✅ Selección de referencias a pronosticar (hasta 50)")
    refs_seleccionadas = st.multiselect(
        "Elige una o varias referencias:",
        options=columnas_referencias,
        default=[columnas_referencias[0]] if columnas_referencias else [],
        max_selections=50
    )

    if not refs_seleccionadas:
        st.info("Selecciona al menos una referencia para continuar.")
        st.stop()
    # Parámetros del modelo (fijos para SaaS: sin p,d,q visibles)
    ORDER_ARIMA = (1, 1, 0)

    # ----------------------------------------------------------------------
    # Pronóstico por referencia
    # ----------------------------------------------------------------------
    st.markdown("## 📈 Resultados del Pronóstico")

    progress = st.progress(0)
    resultados_metricas = []
    pronosticos_12m = {}

    for i, ref in enumerate(refs_seleccionadas, start=1):
        progress.progress(int((i / len(refs_seleccionadas)) * 100))
        st.markdown(f"### Referencia: **{ref}**")

        serie = df_his[ref].dropna()
        if len(serie) < 8:
            st.warning(f"⚠️ '{ref}' tiene menos de 8 datos útiles. Se omite.")
            continue

        # División de datos (misma lógica: mitad y mitad)
        fecha_corte = serie.index[len(serie) // 2]
        y_test = serie.loc[serie.index >= fecha_corte].copy()

        # Ajuste del modelo
        try:
            model = ARIMA(serie, order=ORDER_ARIMA)
            model_fit = model.fit()
        except Exception as e:
            st.error(f"❌ No pude ajustar el modelo para '{ref}'. Error: {e}")
            continue

        # Predicción para prueba
        try:
            y_pred = model_fit.predict(start=y_test.index[0], end=y_test.index[-1])
        except Exception as e:
            st.error(f"❌ No pude predecir para '{ref}'. Error: {e}")
            continue

        # Alinear índices
        y_test_filtered = y_test.loc[y_test.index.isin(y_pred.index)]
        y_pred_filtered = y_pred.loc[y_pred.index.isin(y_test.index)]

        if y_test_filtered.empty or y_pred_filtered.empty:
            st.warning("⚠️ No se pueden calcular métricas: no coinciden índices entre prueba y predicción.")
            continue

        # Métricas
        mse = mean_squared_error(y_test_filtered, y_pred_filtered)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_filtered, y_pred_filtered)

        resultados_metricas.append({
            "Referencia": ref,
            "MSE": float(mse),
            "RMSE": float(rmse),
            "MAE": float(mae)
        })

        st.markdown(f"- **MSE:** {mse:.2f}")
        st.markdown(f"- **RMSE:** {rmse:.2f}")
        st.markdown(f"- **MAE:** {mae:.2f}")

        # Pronóstico futuro (12 meses)
        try:
            future_pred = model_fit.forecast(steps=12)
        except Exception as e:
            st.error(f"❌ No pude generar pronóstico futuro para '{ref}'. Error: {e}")
            continue

        future_dates = pd.date_range(start=serie.index[-1] + pd.DateOffset(months=1), periods=12, freq="M")
        future_df = pd.DataFrame({"Fecha": future_dates, ref: future_pred}).set_index("Fecha")
        pronosticos_12m[ref] = future_df[ref]

        # Gráfica
        plt.figure(figsize=(10, 6))
        plt.plot(y_test_filtered.index, y_test_filtered, label="Datos Reales (Prueba)")
        plt.plot(y_pred_filtered.index, y_pred_filtered, label="Predicción (Prueba)", linestyle="--")
        plt.plot(future_dates, future_pred, label="Pronóstico Futuro", linestyle="--")
        plt.title(f"Pronóstico de la Demanda Mensual - {ref}")
        plt.xlabel("Fecha")
        plt.ylabel("Demanda")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

        # Tabla futuro
        st.write("**🔮 Pronóstico para los próximos 12 meses:**")
        st.dataframe(future_df)

        st.divider()

    progress.empty()

    # ----------------------------------------------------------------------
    # Resumen: métricas y pronóstico combinado
    # ----------------------------------------------------------------------
    if resultados_metricas:
        st.markdown("## 🧾 Resumen de métricas (referencias procesadas)")
        df_metricas = pd.DataFrame(resultados_metricas).sort_values("RMSE", ascending=True)
        st.dataframe(df_metricas)

    if pronosticos_12m:
        st.markdown("## 📊 Pronóstico combinado (12 meses) — Todas las referencias seleccionadas")
        df_pronostico_combinado = pd.DataFrame(pronosticos_12m)
        st.dataframe(df_pronostico_combinado)

        # Descarga del pronóstico combinado
        st.download_button(
            label="⬇️ Descargar pronóstico combinado (CSV)",
            data=df_pronostico_combinado.to_csv(index=True).encode("utf-8"),
            file_name="pronostico_12m_combinado.csv",
            mime="text/csv"
        )

    # ----------------------------------------------------------------------
    # INTERPRETACIÓN AUTOMÁTICA CON IA (solo para 1 referencia a la vez)
    # ----------------------------------------------------------------------
    if pronosticos_12m:
        st.markdown("## 🧠 Interpretación Estratégica con IA (elige 1 referencia)")

        ref_ia = st.selectbox("Referencia para interpretación IA:", list(pronosticos_12m.keys()))

        perfil = st.selectbox(
            "Selecciona tu perfil para recibir un análisis adaptado:",
            ["Gerente General", "Gerente Comercial", "Gerente de Producción", "Logística", "Compras", "Ventas"]
        )

        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            st.info("ℹ️ Para usar la interpretación con IA, configura la variable de entorno OPENAI_API_KEY en tu hosting.")
        else:
            client = OpenAI(api_key=api_key)

            # Buscar métricas de esa referencia
            met = next((m for m in resultados_metricas if m["Referencia"] == ref_ia), None)
            mse_txt = f"{met['MSE']:.2f}" if met else "N/A"
            rmse_txt = f"{met['RMSE']:.2f}" if met else "N/A"
            mae_txt = f"{met['MAE']:.2f}" if met else "N/A"

            future_df_ia = pd.DataFrame({ref_ia: pronosticos_12m[ref_ia]})

            prompt = f"""
Asume el rol de un analista senior en inteligencia comercial y operativa especializado en planeación de la demanda y gestión de inventarios. 
Tu tarea es asistir exclusivamente al siguiente perfil: **{perfil}**.

La referencia analizada es: **{ref_ia}**.

Genera un análisis estratégico y accionable del pronóstico de demanda para apoyar las decisiones de este perfil en su ámbito de gestión.

### Instrucciones principales:
- Enfócate únicamente en el perfil seleccionado ({perfil}). 
- No incluyas interpretaciones o recomendaciones para otros roles.
- Adapta el lenguaje, los indicadores y la profundidad técnica al contexto de este perfil.
- No menciones algoritmos ni términos estadísticos.
- Prioriza el impacto financiero, operativo y de servicio.
- Si faltan datos relevantes, sugiere cuáles serían necesarios.

### Estructura esperada del resultado:
1. **Análisis del Contexto**
2. **Interpretación para {perfil}**
3. **Recomendaciones Estratégicas**
4. **Conclusión Ejecutiva**

### Métricas (si están disponibles):
- MSE: {mse_txt}
- RMSE: {rmse_txt}
- MAE: {mae_txt}

### Pronóstico de demanda (12 meses):
{future_df_ia.tail(12).to_markdown()}
"""

            if st.button("🪄 Generar Interpretación con IA"):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "Eres un analista experto en planeación de la demanda e inteligencia comercial."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    st.success("✅ Análisis generado exitosamente.")
                    st.markdown(response.choices[0].message.content)
                except Exception as e:
                    if "quota" in str(e).lower():
                        st.error("⚠️ Tu cuenta de OpenAI ha superado la cuota de uso. Revisa tu panel de facturación.")
                    else:
                        st.error(f"Ocurrió un error al conectar con ChatGPT: {e}")

# --------------------------------------------------------------------------
# SECCIÓN DE CONTACTO / CTA
# --------------------------------------------------------------------------
st.markdown("""
    <div class="landing-page">
        <h2>¿Listo para optimizar tu inventario y producción?</h2>
        <p>Regístrate para recibir una asesoría personalizada sobre cómo usar estos pronósticos en tu empresa.</p>
        <a href="https://forms.gle/oJ84oWinHXuMqY3v5" class="cta-button" target="_blank">¡Regístrate!</a>
    </div>
    """, unsafe_allow_html=True)
