#!/usr/bin/env python3
# evaluate_grok.py
# Version robuste pour ton codespace – chemin par défaut + gestion string JSON

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

try:
    from jiwer import cer, wer
    from rouge_score import rouge_scorer
    import numpy as np
except ImportError as e:
    print("ERREUR: dépendances manquantes", file=sys.stderr)
    print("Exécutez : pip install jiwer rouge-score numpy", file=sys.stderr)
    sys.exit(1)

@dataclass
class Sample:
    id: str
    task_type: str
    resident_gt: str
    resident_pred: str
    transcript_gt: str
    transcript_pred: str
    json_gt: Any      # peut être dict ou str
    json_pred: Any    # peut être dict ou str

def safe_json_load(value: Any) -> Dict:
    """Transforme str → dict si besoin, gère les erreurs"""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            print(f"  → JSON invalide pour cet item : {e}", file=sys.stderr)
            print(f"    Valeur brute (tronquée) : {value[:180]}...", file=sys.stderr)
            return {}
    return {}

def load_dataset(path: Path) -> List[Sample]:
    if not path.is_file():
        print(f"ERREUR : fichier introuvable → {path}", file=sys.stderr)
        sys.exit(1)
    
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    
    samples = []
    for item in raw:
        # Normalisation des json_gt / json_pred si ce sont des strings
        item["json_gt"]   = safe_json_load(item.get("json_gt"))
        item["json_pred"] = safe_json_load(item.get("json_pred"))
        samples.append(Sample(**item))
    
    print(f"Dataset chargé → {len(samples)} exemples")
    return samples

# ─── Métriques STT ───────────────────────────────────────────────

def compute_stt_metrics(gt: str, pred: str) -> Dict[str, float]:
    gt_clean = gt.strip().lower()
    pred_clean = pred.strip().lower()
    return {
        "cer": cer(gt_clean, pred_clean),
        "wer": wer(gt_clean, pred_clean),
    }

def resident_accuracy(samples: List[Sample]) -> float:
    correct = sum(1 for s in samples if s.resident_gt.strip().lower() == s.resident_pred.strip().lower())
    return correct / len(samples) if samples else 0.0

# ─── Normalisation des valeurs numériques (tolérance légère) ─────

def normalize_value(v: Any) -> Any:
    if isinstance(v, (int, float)):
        return round(float(v), 1)
    if isinstance(v, str):
        try:
            cleaned = v.replace(",", ".").replace(" ", "").strip()
            return round(float(cleaned), 1)
        except:
            return v.strip().lower()
    return v

def vitals_exact_match(gt_vitals: List[Dict], pred_vitals: List[Dict]) -> float:
    if not gt_vitals:
        return 1.0 if not pred_vitals else 0.0
    
    gt_set = {
        (v["name"].lower().strip(), normalize_value(v["value"]), v.get("unit", "").lower().strip())
        for v in gt_vitals
    }
    pred_set = {
        (v["name"].lower().strip(), normalize_value(v["value"]), v.get("unit", "").lower().strip())
        for v in pred_vitals
    }
    
    tp = len(gt_set & pred_set)
    return tp / len(gt_set) if gt_set else 0.0

# ─── Évaluation structurée par type de tâche ─────────────────────

def rouge_on_fields(gt_dict: Dict, pred_dict: Dict, fields: List[str]) -> float:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    for field in fields:
        gt_text = str(gt_dict.get(field, "")).strip()
        pred_text = str(pred_dict.get(field, "")).strip()
        if gt_text:
            scores.append(scorer.score(gt_text, pred_text)["rougeL"].fmeasure)
    return np.mean(scores) if scores else 0.0

def evaluate_structured(sample: Sample) -> Dict[str, float]:
    tt = sample.task_type.lower()
    result = {}

    gt = sample.json_gt
    pred = sample.json_pred

    if tt == "vitals":
        score = vitals_exact_match(
            gt.get("vitals", []),
            pred.get("vitals", [])
        )
        result["vitals_match"] = round(score, 4)

    elif tt == "targeted":
        rouge = rouge_on_fields(gt, pred, ["data", "actions", "results"])
        target_ok = gt.get("target", "").lower().strip() == pred.get("target", "").lower().strip()
        result.update({
            "rougeL_targeted": round(rouge, 4),
            "target_exact": 1.0 if target_ok else 0.0
        })

    elif tt == "narrative":
        gt_secs = {
            s["title"].lower().strip(): s.get("content", "").strip()
            for s in gt.get("sections", [])
            if isinstance(s, dict) and "title" in s
        }
        pred_secs = {
            s["title"].lower().strip(): s.get("content", "").strip()
            for s in pred.get("sections", [])
            if isinstance(s, dict) and "title" in s
        }

        common = set(gt_secs) & set(pred_secs)
        recall = len(common) / len(gt_secs) if gt_secs else 1.0
        precision = len(common) / len(pred_secs) if pred_secs else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        rouge_scores = []
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        for title in common:
            score = scorer.score(gt_secs[title], pred_secs[title])["rougeL"].fmeasure
            rouge_scores.append(score)
        rouge_mean = np.mean(rouge_scores) if rouge_scores else 0.0

        result.update({
            "section_f1": round(f1, 4),
            "rougeL_narrative": round(rouge_mean, 4)
        })

    return result

# ─── Main ────────────────────────────────────────────────────────

def main():
    DEFAULT_DATASET = Path("data/dataset_eval_speakli.json")

    parser = argparse.ArgumentParser(description="Évaluation pipeline Speakli")
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=DEFAULT_DATASET,
        help=f"Chemin vers dataset_eval_speakli.json (défaut: {DEFAULT_DATASET})"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("report.json"),
        help="Fichier de sortie (défaut: report.json)"
    )
    args = parser.parse_args()

    print(f"Dataset utilisé : {args.dataset.resolve()}\n")

    try:
        samples = load_dataset(args.dataset)
    except Exception as e:
        print(f"Erreur chargement dataset : {e}", file=sys.stderr)
        sys.exit(1)

    results = []
    stt_cers, stt_wers = [], []
    resident_ok = 0

    for s in samples:
        stt = compute_stt_metrics(s.transcript_gt, s.transcript_pred)
        stt_cers.append(stt["cer"])
        stt_wers.append(stt["wer"])

        res_match = s.resident_gt.strip().lower() == s.resident_pred.strip().lower()
        if res_match:
            resident_ok += 1

        struct = evaluate_structured(s)

        results.append({
            "id": s.id,
            "task_type": s.task_type,
            **stt,
            "resident_ok": int(res_match),
            **struct
        })

    # Rapport agrégé
    report = {
        "global": {
            "nb_samples": len(samples),
            "cer_mean": round(float(np.mean(stt_cers)), 4),
            "wer_mean": round(float(np.mean(stt_wers)), 4),
            "resident_accuracy": round(resident_ok / len(samples), 4) if samples else 0.0,
        },
        "by_task": {}
    }

    for tt in ["vitals", "targeted", "narrative"]:
        subset = [r for r in results if r["task_type"].lower() == tt]
        if not subset:
            continue
        sub = {
            "count": len(subset),
            "cer_mean": round(float(np.mean([r["cer"] for r in subset])), 4),
            "wer_mean": round(float(np.mean([r["wer"] for r in subset])), 4),
        }
        if tt == "vitals":
            sub["vitals_match_mean"] = round(float(np.mean([r.get("vitals_match", 0) for r in subset])), 4)
        elif tt == "targeted":
            sub["rougeL_mean"] = round(float(np.mean([r.get("rougeL_targeted", 0) for r in subset])), 4)
        elif tt == "narrative":
            sub["section_f1_mean"] = round(float(np.mean([r.get("section_f1", 0) for r in subset])), 4)
            sub["rougeL_mean"] = round(float(np.mean([r.get("rougeL_narrative", 0) for r in subset])), 4)
        report["by_task"][tt] = sub

    # Sauvegarde
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump({"report": report, "details": results}, f, ensure_ascii=False, indent=2)

    print("\n" + "═" * 70)
    print("RÉSULTATS GLOBAUX")
    print(json.dumps(report["global"], indent=2, ensure_ascii=False))
    print("═" * 70)
    print(f"Rapport complet sauvegardé → {args.output.resolve()}\n")

if __name__ == "__main__":
    main()