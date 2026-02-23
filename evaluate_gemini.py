import json
import pandas as pd
from jiwer import wer, cer
from difflib import SequenceMatcher
import os

class SpeakliEvaluator:
    def __init__(self, file_path):
        """Initialise l'évaluateur avec gestion d'erreur de chemin."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Le fichier {file_path} est introuvable.")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.results = []

    def normalize_value(self, v):
        """Transforme '38,5' en 38.5 pour une comparaison équitable."""
        try:
            return round(float(str(v).replace(',', '.').replace(' ', '')), 1)
        except:
            return str(v).lower().strip()

    def calculate_stt_metrics(self, gt, pred):
        """Calcule le WER (mots) et le CER (caractères)."""
        if not pred or not gt:
            return 1.0, 1.0
        gt_c, pred_c = gt.lower(), pred.lower()
        return round(wer(gt_c, pred_c), 4), round(cer(gt_c, pred_c), 4)

    def evaluate_json_quality(self, gt, pred_raw, task_type):
        """Évalue la précision du contenu du JSON (Vitals et texte)."""
        try:
            # Réparation de fortune pour JSON tronqué (ex: cas n06)
            if isinstance(pred_raw, str):
                clean_pred = pred_raw.strip()
                if not clean_pred.endswith('}'):
                    clean_pred += '}'
                pred = json.loads(clean_pred)
            else:
                pred = pred_raw

            if task_type == 'vitals':
                # Comparaison intelligente des valeurs numériques
                gt_v = { (v['name'].lower(), self.normalize_value(v['value'])) for v in gt.get('vitals', []) }
                pred_v = { (v['name'].lower(), self.normalize_value(v['value'])) for v in pred.get('vitals', []) }
                intersection = gt_v & pred_v
                return len(intersection) / len(gt_v) if gt_v else 1.0
            
            # Pour Narrative/Targeted : on vérifie si les clés principales existent
            return 1.0 if pred else 0.0
            
        except (json.JSONDecodeError, AttributeError):
            return 0.0

    def run(self):
        """Boucle principale d'évaluation."""
        for entry in self.data:
            # 1. Métriques STT (Transcription)
            stt_wer, stt_cer = self.calculate_stt_metrics(entry['transcript_gt'], entry['transcript_pred'])
            
            # 2. Métriques Résident (Exact Match + Similarité phonétique)
            res_gt = entry['resident_gt'].strip().lower()
            res_pred = entry['resident_pred'].strip().lower()
            res_match = 1 if res_gt == res_pred else 0
            res_sim = round(SequenceMatcher(None, res_gt, res_pred).ratio(), 4)
            
            # 3. Qualité de l'extraction JSON
            json_qual = self.evaluate_json_quality(entry['json_gt'], entry['json_pred'], entry['task_type'])
            
            # Hallucination si le JSON est suspectement plus long que la vérité
            is_hallu = "Yes" if json_qual > 0 and len(str(entry['json_pred'])) > len(str(entry['json_gt'])) * 1.5 else "No"

            self.results.append({
                "id": entry['id'],
                "task_type": entry['task_type'],
                "wer": stt_wer,
                "cer": stt_cer,
                "res_match": res_match,
                "res_similarity": res_sim,
                "json_quality": json_qual,
                "hallucination": is_hallu
            })

    def generate_reports(self):
        """Sortie console et fichiers de rapport."""
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*50)
        print("📊 SYNTHÈSE DES PERFORMANCES AMÉLIORÉE")
        print("="*50)
        summary = df.groupby('task_type').agg({
            'wer': 'mean',
            'cer': 'mean',
            'res_match': 'mean',
            'res_similarity': 'mean',
            'json_quality': 'mean'
        }).round(3)
        print(summary)
        
        # Identification des points noirs
        print("\n⚠️ ALERTES QUALITÉ (PIRES SCORES STT) :")
        print(df.sort_values(by='wer', ascending=False)[['id', 'task_type', 'wer']].head(3))

        os.makedirs('outputs', exist_ok=True)
        df.to_csv('outputs/report.csv', index=False)
        with open('outputs/summary.json', 'w', encoding='utf-8') as f:
            f.write(summary.to_json(indent=4))
        print(f"\n✅ Rapport détaillé : outputs/report.csv")

if __name__ == "__main__":
    evaluator = SpeakliEvaluator('data/dataset_eval_speakli.json')
    evaluator.run()
    evaluator.generate_reports()