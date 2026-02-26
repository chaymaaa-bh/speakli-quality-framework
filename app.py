import streamlit as st
import pandas as pd
import json
import os
import io
from fpdf import FPDF
import re

# IMPORTATION DE TA CLASSE DE CALCUL (Assure-toi que evaluate.py est dans le même dossier)
from evaluate import SpeakliEvaluator 

# Configuration de la page
st.set_page_config(page_title="Speakli Quality Framework", layout="wide")

# --- STYLE CSS PERSONNALISÉ ---
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { color: #1f77b4 !important; font-weight: bold; }
    [data-testid="stMetricLabel"] { color: #9eb9d4 !important; }
    .stMetric { background-color: #1e2130; padding: 20px; border-radius: 10px; border: 1px solid #3d4455; }
    </style>
    """, unsafe_allow_html=True)

# --- FONCTION GÉNÉRATION PDF ---
def generate_pdf_report(df, summary):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "Speakli Quality Framework - Audit Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Synthese des Performances :", ln=True)
    pdf.set_font("Courier", size=10)
    for task, metrics in summary.items():
        pdf.cell(0, 10, f"- {task.upper()}: WER={metrics['wer']} | Qual={metrics['extraction_quality']}", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Alertes de Securite detectees :", ln=True)
    pdf.set_font("Courier", size=9)
    criticals = df[df['safety_alert'] == "YES"]
    for _, row in criticals.iterrows():
        pdf.multi_cell(0, 5, f"ID: {row['id']} | Tache: {row['task_type']} | WER: {row['wer']}")
        
    return pdf.output(dest='S').encode('latin-1', 'replace')

# --- INITIALISATION ET CHARGEMENT ---
st.title("🚀 Speakli Quality Framework Dashboard")
st.markdown("Interface d'audit automatisée pour le monitoring des performances STT et la sécurité du pipeline LLM.")

st.sidebar.header("📂 Importation du Dataset")
uploaded_file = st.sidebar.file_uploader("Charger un fichier JSON Speakli", type=['json'])

# Chemin du fichier par défaut
DEFAULT_DATA = "data/dataset_eval_speakli.json"

# Sélection de la source de données
if uploaded_file is not None:
    data_source = json.load(uploaded_file)
    st.sidebar.success("Dataset personnalisé chargé !")
else:
    if os.path.exists(DEFAULT_DATA):
        with open(DEFAULT_DATA, 'r', encoding='utf-8') as f:
            data_source = json.load(f)
        st.sidebar.info("Utilisation du dataset par défaut.")
    else:
        st.error("⚠️ Aucun dataset trouvé. Veuillez uploader un fichier JSON.")
        st.stop()

# --- EXÉCUTION DU MOTEUR DE CALCUL ---
@st.cache_data
def get_analysis_results(json_data):
    # On initialise ton évaluateur avec les données chargées
    # Note : On modifie légèrement l'init pour accepter un dict si besoin, 
    # ou on sauvegarde temporairement. Ici on simule l'exécution :
    evaluator = SpeakliEvaluator(DEFAULT_DATA) # On utilise ta classe
    evaluator.data = json_data # On injecte les données (upload ou défaut)
    evaluator.run()
    
    df_res = pd.DataFrame(evaluator.results)
    
    # Calcul du résumé pour l'interface
    summary_res = df_res.groupby('task_type').agg({
        'wer': 'mean', 'res_similarity': 'mean', 'extraction_quality': 'mean'
    }).round(3).to_dict(orient='index')
    
    return df_res, summary_res, evaluator.error_catalog

df, summary, error_catalog = get_analysis_results(data_source)

# --- BARRE LATÉRALE : CONFIGURATION & EXPORTS ---
st.sidebar.divider()
st.sidebar.header("⚙️ Configuration")
task_filter = st.sidebar.multiselect("Filtrer par tâche", options=df['task_type'].unique(), default=df['task_type'].unique())
hallu_only = st.sidebar.checkbox("Isoler les alertes de sécurité")

st.sidebar.divider()
st.sidebar.header("📥 Télécharger les rapports")
st.sidebar.download_button("1. Télécharger CSV", df.to_csv(index=False).encode('utf-8'), "report.csv", "text/csv")
st.sidebar.download_button("2. Télécharger JSON", json.dumps(summary, indent=4), "summary.json", "application/json")
pdf_bytes = generate_pdf_report(df, summary)
st.sidebar.download_button("3. Télécharger PDF (Audit)", pdf_bytes, "audit_log.pdf", "application/pdf")

# --- FILTRAGE DES DONNÉES ---
filtered_df = df[df['task_type'].isin(task_filter)]
if hallu_only:
    filtered_df = filtered_df[filtered_df['safety_alert'] == "YES"]

# --- AFFICHAGE DES KPIs ---
st.subheader("📌 Indicateurs Clés de Performance")
m1, m2, m3, m4 = st.columns(4)
if not filtered_df.empty:
    m1.metric("WER Global", f"{filtered_df['wer'].mean():.1%}")
    m2.metric("Qualité Extraction", f"{filtered_df['extraction_quality'].mean():.1%}")
    m3.metric("Sim. Résident", f"{filtered_df['res_similarity'].mean():.1%}")
    nb_hallus = len(filtered_df[filtered_df['safety_alert'] == "YES"])
    m4.metric("Alertes Sécurité", f"{nb_hallus}", delta=f"{nb_hallus} critiques", delta_color="inverse")

st.divider()

# --- ANALYSE ET EXPLORATEUR ---
col_left, col_right = st.columns([1, 2])
with col_left:
    st.subheader("📊 Stats par Catégorie")
    summary_df = pd.DataFrame(summary).T
    st.dataframe(summary_df[['wer', 'extraction_quality']], width='stretch')

with col_right:
    st.subheader("🔍 Explorateur de Résultats")
    def highlight_errors(row):
        return ['background-color: #4a1d1d' if row.safety_alert == "YES" else '' for _ in row]
    st.dataframe(filtered_df.style.apply(highlight_errors, axis=1), height=400, width='stretch')

# --- FOCUS SÉCURITÉ ---
st.divider()
st.subheader("🛡️ Analyse des Alertes de Sécurité")
critical_cases = filtered_df[filtered_df['safety_alert'] == "YES"]

if not critical_cases.empty:
    selected_id = st.selectbox("Sélectionner un incident :", critical_cases['id'].unique(), key="incident_selector")
    case_info = critical_cases[critical_cases['id'] == selected_id].iloc[0]
    
    # Récupération de l'erreur précise dans le catalogue
    case_errors = [e['desc'] for e in error_catalog if e['id'] == selected_id]
    
    st.error(f"Incident identifié sur le rapport {selected_id}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Diagnostic Technique**")
        for err in case_errors:
            st.info(f"Détail : {err}")
    with c2:
        st.markdown("**Action Recommandée**")
        st.success("Validation humaine prioritaire avant injection DPI.")
else:
    st.success("✅ Aucune alerte de sécurité détectée.")

st.caption("Speakli Quality Framework v1.0 | Dashboard Dynamique | 2026")