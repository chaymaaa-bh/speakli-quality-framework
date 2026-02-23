import streamlit as st
import pandas as pd
import json
import os
import subprocess
from fpdf import FPDF

# Configuration de la page
st.set_page_config(page_title="Speakli Quality Framework", layout="wide")

# --- ÉTAPE 0 : EXÉCUTION DU MOTEUR & CAPTURE CONSOLE ---
@st.cache_resource
def run_evaluation_engine():
    """Lance le calcul et capture la sortie console pour le PDF."""
    try:
        # Capture stdout pour récupérer le tableau ASCII de la console
        result = subprocess.run(["python", "evaluate.py"], capture_output=True, text=True, check=True)
        return result.stdout
    except Exception as e:
        st.error(f"Erreur moteur : {e}")
        return None

# --- STYLE CSS (Correction Contraste & Thème Sombre) ---
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { color: #1f77b4 !important; font-weight: bold; }
    [data-testid="stMetricLabel"] { color: #9eb9d4 !important; }
    .stMetric { background-color: #1e2130; padding: 20px; border-radius: 10px; border: 1px solid #3d4455; }
    </style>
    """, unsafe_allow_html=True)

# --- FONCTION GÉNÉRATION PDF SÉCURISÉE ---
def generate_pdf_report(text_logs):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "Speakli Quality Framework - Audit Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Courier", size=9)
    
    # NETTOYAGE STRICT : On ne garde que les caractères ASCII imprimables pour éviter l'UnicodeEncodeError
    clean_text = "".join(i for i in text_logs if ord(i) < 128)
    clean_text = clean_text.replace('[?]', '-').replace('=', '=')
    
    pdf.multi_cell(0, 5, clean_text)
    return pdf.output(dest='S').encode('latin-1', 'replace')

st.title("🚀 Speakli Quality Framework Dashboard")
st.markdown("Interface d'audit automatisée pour le monitoring des performances STT et la sécurité du pipeline LLM.")

# Exécution automatique du backend
console_output = run_evaluation_engine()

if console_output:
    # Chargement des fichiers générés
    csv_path, json_path = 'outputs/report.csv', 'outputs/summary.json'
    
    if os.path.exists(csv_path) and os.path.exists(json_path):
        df = pd.read_csv(csv_path)
        with open(json_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)

        # --- BARRE LATÉRALE : CONFIGURATION & MULTI-EXPORTS ---
        st.sidebar.header("⚙️ Configuration")
        task_filter = st.sidebar.multiselect("Filtrer par tâche", options=df['task_type'].unique(), default=df['task_type'].unique())
        hallu_only = st.sidebar.checkbox("Isoler les alertes de sécurité")

        st.sidebar.divider()
        st.sidebar.header("📥 Télécharger les rapports")
        
        # Export CSV (Détails)
        st.sidebar.download_button("1. Télécharger CSV (Détails)", df.to_csv(index=False).encode('utf-8'), "report_details.csv", "text/csv")
        
        # Export JSON (Synthèse)
        st.sidebar.download_button("2. Télécharger JSON (Synthèse)", json.dumps(summary_data, indent=4), "summary.json", "application/json")
        
        # Export PDF (Audit Console)
        pdf_bytes = generate_pdf_report(console_output)
        st.sidebar.download_button("3. Télécharger PDF (Logs Console)", pdf_bytes, "audit_log.pdf", "application/pdf")

        # --- FILTRAGE DES DONNÉES ---
        filtered_df = df[df['task_type'].isin(task_filter)]
        if hallu_only:
            filtered_df = filtered_df[filtered_df['safety_alert'] == "YES"]

        # --- KPI GÉNÉRAUX ---
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
            summary_df = pd.DataFrame(summary_data).T
            st.dataframe(summary_df[['wer', 'extraction_quality']], width='stretch')

        with col_right:
            st.subheader("🔍 Explorateur de Résultats")
            def highlight_errors(row):
                return ['background-color: #4a1d1d' if row.safety_alert == "YES" else '' for _ in row]
            st.dataframe(filtered_df.style.apply(highlight_errors, axis=1), height=400, width='stretch')

        # --- FOCUS SÉCURITÉ (INTERACTIF) ---
        st.divider()
        st.subheader("🛡️ Analyse des Alertes de Sécurité")
        critical_cases = filtered_df[filtered_df['safety_alert'] == "YES"]
        
        if not critical_cases.empty:
            # Utilisation de key="incident_selector" pour assurer l'interactivité du widget
            selected_id = st.selectbox("Sélectionner un incident :", critical_cases['id'].unique(), key="incident_selector")
            case_info = critical_cases[critical_cases['id'] == selected_id].iloc[0]
            
            st.error(f"Incident identifié sur le rapport {selected_id}")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Diagnostic Technique**")
                st.info(f"Écart détecté pour la tâche : {case_info['task_type'].upper()}")
                st.write(f"WER spécifique au cas : **{case_info['wer']:.3f}**")
            with c2:
                st.markdown("**Action Recommandée**")
                st.success("Validation humaine prioritaire avant injection DPI.")
        else:
            st.success("✅ Aucune alerte de sécurité détectée dans le périmètre actuel.")

        st.divider()
        st.caption("Speakli Quality Framework v1.0 | Audit Pipeline Automatisé | 2026")
    else:
        st.error("⚠️ Fichiers de sortie introuvables. Vérifiez que evaluate.py a bien généré les rapports.")
else:
    st.error("Impossible de lancer l'interface : le moteur d'évaluation n'a pas pu s'exécuter.")