import json
import pandas as pd
from jiwer import wer, cer
from difflib import SequenceMatcher
import os
import numpy as np

class SpeakliEvaluator:
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier {file_path} introuvable.")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.results = []
        self.error_catalog = [] # Pour stocker les détails des erreurs critiques

    def normalize_value(self, v):
        """Nettoyage pour éviter les faux négatifs (38,5 -> 38.5)."""
        try:
            cleaned = str(v).replace(',', '.').replace(' ', '').strip()
            return round(float(cleaned), 1)
        except (ValueError, TypeError):
            return str(v).lower().strip()

    def calculate_stt_metrics(self, gt, pred):
        if not pred or str(pred).strip() == "":
            return 1.0, 1.0
        gt_c, pred_c = str(gt).lower(), str(pred).lower()
        return round(wer(gt_c, pred_c), 4), round(cer(gt_c, pred_c), 4)

    def evaluate_json_content(self, entry):
        """Analyse fine du JSON selon le type de tâche."""
        gt = entry['json_gt']
        pred_raw = entry['json_pred']
        task_type = entry['task_type']
        
        try:
            # 1. Tentative de réparation/parsing
            if isinstance(pred_raw, str):
                clean_pred = pred_raw.strip()
                if not clean_pred.endswith('}'): clean_pred += '}'
                pred = json.loads(clean_pred)
            else:
                pred = pred_raw

            # 2. Logique spécifique par tâche
            if task_type == 'vitals':
                gt_list = gt.get('vitals', [])
                pred_list = pred.get('vitals', [])
                
                gt_v = { (v.get('name', '').lower(), self.normalize_value(v.get('value', ''))) for v in gt_list }
                pred_v = { (v.get('name', '').lower(), self.normalize_value(v.get('value', ''))) for v in pred_list }
                
                # Check des erreurs critiques (différence de valeur)
                for g_name, g_val in gt_v:
                    found = False
                    for p_name, p_val in pred_v:
                        if g_name == p_name:
                            found = True
                            if g_val != p_val:
                                self.error_catalog.append({
                                    'id': entry['id'], 'type': 'CRITICAL_VALUE_MISMATCH',
                                    'desc': f"{g_name}: GT={g_val} vs PRED={p_val}"
                                })
                    if not found:
                        self.error_catalog.append({'id': entry['id'], 'type': 'MISSING_DATA', 'desc': f"Constante {g_name} oubliée"})

                intersection = gt_v & pred_v
                return len(intersection) / len(gt_v) if gt_v else 1.0
            
            else:
                # Pour Narrative et Targeted : Similarité sémantique
                gt_str = json.dumps(gt, sort_keys=True)
                pred_str = json.dumps(pred, sort_keys=True)
                return SequenceMatcher(None, gt_str, pred_str).ratio()
                
        except Exception as e:
            self.error_catalog.append({'id': entry['id'], 'type': 'JSON_PARSE_ERROR', 'desc': str(e)})
            return 0.0

    def run(self):
        for entry in self.data:
            # Métriques STT
            stt_wer, stt_cer = self.calculate_stt_metrics(entry['transcript_gt'], entry['transcript_pred'])
            
            # Sécurité Patient (Levenshtein ratio)
            res_gt = str(entry['resident_gt']).strip().lower()
            res_pred = str(entry['resident_pred']).strip().lower()
            res_sim = round(SequenceMatcher(None, res_gt, res_pred).ratio(), 4)
            
            # Qualité Extraction
            json_qual = self.evaluate_json_content(entry)
            
            # Hallucination (basé sur la longueur et contenu)
            is_hallu = "Yes" if len(str(entry['json_pred'])) > len(str(entry['json_gt'])) * 1.5 else "No"
            if is_hallu == "Yes":
                self.error_catalog.append({'id': entry['id'], 'type': 'HALLUCINATION', 'desc': "Contenu généré trop verbeux"})

            self.results.append({
                "id": entry['id'],
                "task_type": entry['task_type'],
                "wer": stt_wer,
                "cer": stt_cer,
                "res_similarity": res_sim,
                "extraction_quality": round(json_qual, 4),
                "hallucination": is_hallu
            })

    def generate_recommendations(self, summary):
        recs = []
        if summary.loc['vitals', 'extraction_quality'] < 0.7:
            recs.append("🎯 Priorité : Améliorer l'extraction des constantes (Vitals). Vérifier le formatage des nombres.")
        if summary['wer'].mean() > 0.4:
            recs.append("🎙️ STT : Le taux d'erreur est élevé. Envisager un fine-tuning ou un meilleur micro soignant.")
        if summary['res_similarity'].mean() < 0.9:
            recs.append("👤 Patient : Risque d'erreur d'identification. Implémenter un Fuzzy Matching sur la base résidents.")
        return recs

    def generate_reports(self):
        df = pd.DataFrame(self.results)
        summary = df.groupby('task_type').agg({
            'wer': 'mean', 'cer': 'mean', 'res_similarity': 'mean', 'extraction_quality': 'mean'
        }).round(3)

        print("\n" + "═"*60)
        print("🚀 SPEAKLI AI EVALUATION - DASHBOARD")
        print("═"*60)
        print(summary)
        print("═"*60)

        # Top Erreurs
        if self.error_catalog:
            print("\n🔍 ANALYSE DES ERREURS CRITIQUES :")
            err_df = pd.DataFrame(self.error_catalog)
            print(err_df['type'].value_counts().to_string())

        # Recommandations
        print("\n💡 RECOMMANDATIONS STRATÉGIQUES :")
        for r in self.generate_recommendations(summary):
            print(f"- {r}")

        # Sauvegarde
        os.makedirs('outputs', exist_ok=True)
        df.to_csv('outputs/detailed_report.csv', index=False)
        with open('outputs/summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary.to_dict(orient='index'), f, indent=4)
        print(f"\n✅ Rapports générés dans /outputs/")

if __name__ == "__main__":
    evaluator = SpeakliEvaluator('data/dataset_eval_speakli.json')
    evaluator.run()
    evaluator.generate_reports()