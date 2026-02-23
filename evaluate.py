import json
import pandas as pd
from jiwer import wer
from difflib import SequenceMatcher
import os
import re

class SpeakliEvaluator:
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier {file_path} introuvable.")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.results = []
        self.error_catalog = []

    def normalize_value(self, v):
        """Standardise les nombres (38,5 -> 38.5) pour éviter les faux négatifs."""
        try:
            cleaned = str(v).replace(',', '.').replace(' ', '').strip()
            return round(float(cleaned), 1)
        except (ValueError, TypeError):
            return str(v).lower().strip()

    def token_overlap(self, gt, pred):
        """Mesure le Rappel (Recall) des mots-clés essentiels."""
        def get_tokens(t): return set(re.sub(r"[^\w\s]", " ", str(t).lower()).split())
        gt_t, pr_t = get_tokens(gt), get_tokens(pred)
        if not gt_t: return 1.0
        return round(len(gt_t & pr_t) / len(gt_t), 4)

    def check_clinical_safety(self, gt_text, pred_obj):
        """Scanner de sécurité pour détecter les termes sensibles inventés."""
        critical_terms = ["stade 2", "stade 3", "chute", "laxatif", "eva 7", "39°", "escarre"]
        pred_text = str(pred_obj).lower()
        gt_text = str(gt_text).lower()
        return [p for p in critical_terms if p in pred_text and p not in gt_text]

    def evaluate_json_content(self, entry):
        gt = entry['json_gt']
        pred_raw = entry['json_pred']
        task_type = entry['task_type']
        
        try:
            pred = json.loads(pred_raw) if isinstance(pred_raw, str) else pred_raw

            if task_type == 'vitals':
                gt_list = gt.get('vitals', [])
                pred_list = pred.get('vitals', [])
                
                gt_v = { (v.get('name', '').lower(), self.normalize_value(v.get('value', ''))) for v in gt_list }
                pred_v = { (v.get('name', '').lower(), self.normalize_value(v.get('value', ''))) for v in pred_list }
                
                for gn, gv in gt_v:
                    matched = False
                    for pn, pv in pred_v:
                        if gn == pn:
                            matched = True
                            if gv != pv:
                                self.error_catalog.append({'id': entry['id'], 'type': 'VALUE_MISMATCH', 'desc': f"{gn}: {gv} vs {pv}"})
                    if not matched:
                        self.error_catalog.append({'id': entry['id'], 'type': 'MISSING_DATA', 'desc': f"Constante {gn} oubliée"})
                
                return len(gt_v & pred_v) / len(gt_v) if gt_v else 1.0
            else:
                return self.token_overlap(str(gt), str(pred))
                
        except Exception as e:
            self.error_catalog.append({'id': entry['id'], 'type': 'PARSE_ERROR', 'desc': str(e)})
            return 0.0

    def run(self):
        for entry in self.data:
            # 1. Qualité Audio (STT)
            stt_wer = round(wer(entry['transcript_gt'], entry['transcript_pred']), 4)
            
            # 2. Similarité Résident
            res_sim = round(SequenceMatcher(None, str(entry['resident_gt']).lower(), str(entry['resident_pred']).lower()).ratio(), 4)
            
            # 3. Qualité Extraction + Hallucinations
            json_qual = self.evaluate_json_content(entry)
            hallus = self.check_clinical_safety(entry['transcript_gt'], entry['json_pred'])
            
            if hallus:
                for h in hallus:
                    self.error_catalog.append({'id': entry['id'], 'type': 'CRITICAL_HALLUCINATION', 'desc': f"Invention: {h}"})

            self.results.append({
                "id": entry['id'],
                "task_type": entry['task_type'],
                "wer": stt_wer,
                "res_similarity": res_sim,
                "extraction_quality": json_qual, # NOM DE COLONNE FIXÉ POUR APP.PY
                "safety_alert": "YES" if hallus else "NO"
            })

    def generate_recommendations(self, summary):
        recs = []
        if 'vitals' in summary.index and summary.loc['vitals', 'extraction_quality'] < 0.7:
            recs.append("🎯 PRIORITÉ : Fiabiliser l'extraction numérique (Vitals).")
        if summary['wer'].mean() > 0.4:
            recs.append("🎙️ STT : WER élevé (>40%). Fine-tuner le modèle sur le vocabulaire médical.")
        if summary['res_similarity'].mean() < 0.95:
            recs.append("👤 PATIENT : Risque d'identification. Implémenter un référentiel patient (liste blanche).")
        return recs

    def generate_reports(self):
        df = pd.DataFrame(self.results)
        summary = df.groupby('task_type').agg({
            'wer': 'mean', 'res_similarity': 'mean', 'extraction_quality': 'mean'
        }).round(3)

        print("\n" + "═"*65 + "\n🚀 SPEAKLI QUALITY FRAMEWORK - ULTIMATE DASHBOARD\n" + "═"*65)
        print(summary)
        print("═"*65)

        if self.error_catalog:
            print("\n🚨 ALERTES DE SÉCURITÉ (TOP ERREURS) :")
            err_df = pd.DataFrame(self.error_catalog)
            print(err_df['type'].value_counts().to_string())

        print("\n💡 RECOMMANDATIONS STRATÉGIQUES :")
        for r in self.generate_recommendations(summary):
            print(f"- {r}")

        # EXPORTS : Harmonisation des noms de fichiers pour l'interface app.py
        os.makedirs('outputs', exist_ok=True)
        df.to_csv('outputs/report.csv', index=False) # Changé ultimate_report en report.csv
        with open('outputs/summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary.to_dict(orient='index'), f, indent=4)
        print(f"\n✅ Analyse terminée. Fichiers disponibles dans /outputs/")

if __name__ == "__main__":
    evaluator = SpeakliEvaluator('data/dataset_eval_speakli.json')
    evaluator.run()
    evaluator.generate_reports()