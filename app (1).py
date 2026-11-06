import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

st.set_page_config(page_title="Riesgo de diabetes", page_icon="🩺")

@st.cache_resource
def entrenar_modelo():
    """Entrena la configuración seleccionada en el experimento (SVM lineal, C=10)
    con la misma partición 80/20 y semilla 42 del informe."""
    df = pd.read_csv("dataset_final.csv")
    X, y = df.drop(columns="Resultado"), df["Resultado"]
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    modelo = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("m", SVC(kernel="linear", C=10, probability=True, random_state=42)),
    ])
    return modelo.fit(X_tr, y_tr)

modelo = entrenar_modelo()

st.title("🩺 Predicción de riesgo de diabetes")
st.caption("Herramienta de apoyo a la decisión y de tamizaje. No reemplaza el diagnóstico "
           "médico ni las pruebas de laboratorio.")

with st.sidebar:
    st.header("Ajustes")
    umbral = st.slider("Umbral de alerta", 0.10, 0.90, 0.50, 0.05,
        help="Con un umbral más bajo el sistema alerta a más pacientes (mayor sensibilidad, más falsas alarmas).")
    st.info("Modelo: SVM lineal entrenada con el dataset Pima Indians Diabetes "
            "(mujeres de al menos 21 años).")

st.subheader("Datos del paciente")
c1, c2 = st.columns(2)
with c1:
    embarazos = st.number_input("Número de embarazos", 0, 20, 1)
    glucosa = st.number_input("Glucosa (mg/dL)", 40, 300, 100)
    presion = st.number_input("Presión arterial diastólica (mm Hg)", 30, 140, 70)
    edad = st.number_input("Edad (años)", 21, 90, 35)
with c2:
    peso = st.number_input("Peso (kg)", 30.0, 200.0, 70.0, 0.5)
    talla = st.number_input("Talla (m)", 1.20, 2.10, 1.60, 0.01)
    pedigri = st.number_input("Antecedentes familiares (función de pedigrí)", 0.0, 2.5, 0.4, 0.01,
        help="Valor entre 0 y 2,5 aprox. Cuanto mayor, más antecedentes familiares de diabetes.")
imc = peso / talla**2
st.write(f"**IMC calculado:** {imc:.1f} kg/m²")

st.markdown("**Datos opcionales** (si no se tienen, marca la casilla y el sistema usa un valor típico)")
c3, c4 = st.columns(2)
with c3:
    sin_pliegue = st.checkbox("No dispongo del pliegue cutáneo", value=True)
    pliegue = np.nan if sin_pliegue else st.number_input("Pliegue del tríceps (mm)", 5, 100, 25)
with c4:
    sin_insulina = st.checkbox("No dispongo de la insulina", value=True)
    insulina = np.nan if sin_insulina else st.number_input("Insulina a las 2 h (µU/mL)", 10, 900, 100)

if st.button("Calcular riesgo", type="primary"):
    fila = pd.DataFrame([[embarazos, glucosa, presion, pliegue, insulina, imc, pedigri, edad]],
        columns=["Embarazos","Glucosa","PresionArterial","PliegueCutaneo","Insulina","IMC","Pedigri","Edad"])
    prob = float(modelo.predict_proba(fila)[0, 1])
    st.metric("Probabilidad estimada de diabetes", f"{prob*100:.1f} %")
    st.progress(min(prob, 1.0))
    if prob >= umbral:
        st.error("⚠️ **Riesgo ALTO**: se recomienda derivar a pruebas de laboratorio "
                 "(glucosa en ayunas o hemoglobina glicosilada) y evaluación médica.")
    else:
        st.success("✅ **Riesgo BAJO** según el modelo. Mantener controles preventivos, "
                   "alimentación saludable y actividad física.")
    st.caption("El resultado es una estimación estadística; la decisión final corresponde al profesional de salud. "
               "Los datos ingresados no se almacenan.")
