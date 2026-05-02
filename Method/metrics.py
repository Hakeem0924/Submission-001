import json
import os

# =====================================================================
# Core Metrics 1 & 2: Text Degeneration (N-gram Repetition Ratio) and Length Expansion Ratio (LER)
# =====================================================================
def calculate_rep_n(words_list, n):
    """
    Calculate N-gram repetition ratio (rep-n)
    Formula: 1.0 - (Unique n-grams / Total n-grams)
    """
    if len(words_list) < n:
        return 0.0
        
    n_grams = [tuple(words_list[i:i+n]) for i in range(len(words_list)-n+1)]
    total_ngrams = len(n_grams)
    unique_ngrams = len(set(n_grams))
    
    rep_n_ratio = 1.0 - (unique_ngrams / total_ngrams)
    return rep_n_ratio

def evaluate_efficiency_and_degeneration(clean_text, adv_text):
    """Evaluate Length Expansion Ratio (LER) and N-gram degeneration rate."""
    clean_words = clean_text.lower().split() if clean_text else []
    adv_words = adv_text.lower().split() if adv_text else []
    
    len_clean = max(len(clean_words), 1)
    len_adv = len(adv_words)
    
    # 1. Length Expansion Ratio (LER)
    ler = len_adv / len_clean
    
    # 2. Calculate complete rep-n for clean text (Baseline)
    clean_rep_2 = calculate_rep_n(clean_words, n=2)
    clean_rep_3 = calculate_rep_n(clean_words, n=3)
    clean_rep_4 = calculate_rep_n(clean_words, n=4)
    
    # 3. Calculate complete rep-n for adversarial text
    adv_rep_2 = calculate_rep_n(adv_words, n=2)
    adv_rep_3 = calculate_rep_n(adv_words, n=3)
    adv_rep_4 = calculate_rep_n(adv_words, n=4)
    
    # rep-4 > 0.2 (20%) typically indicates catastrophic looping/degeneration
    is_severe_degeneration = adv_rep_4 > 0.20 

    return {
        "clean_length": len_clean,
        "adv_length": len_adv,
        "LER": round(ler, 4),
        "clean_rep_2": round(clean_rep_2, 4),
        "clean_rep_3": round(clean_rep_3, 4),
        "clean_rep_4": round(clean_rep_4, 4),
        "adv_rep_2": round(adv_rep_2, 4),
        "adv_rep_3": round(adv_rep_3, 4),
        "adv_rep_4": round(adv_rep_4, 4),
        "is_severe_degeneration": is_severe_degeneration
    }

# =====================================================================
# Core Metric 3: Targeted Adversarial Hallucination Rate (TAHR)
# =====================================================================
def evaluate_targeted_hallucination(clean_text, adv_text):
    """Detect specific digital artifact hallucinations induced by adversarial noise."""
    # Exclusive probe lexicon for adversarial attacks
    artifact_lexicon = [
        "pixel", "pixelated", "qr code", "mosaic", "glitch", 
        "digital artifact", "watermark", "colorful square", "overlay",
        "corrupted"
    ]
    
    clean_lower = clean_text.lower()
    adv_lower = adv_text.lower()
    
    hit_words = []
    for word in artifact_lexicon:
        # Strict validation: Appears in adversarial text but NOT in clean text 
        # (excluding artifacts already present in the original image)
        if word in adv_lower and word not in clean_lower:
            hit_words.append(word)
            
    return {
        "has_artifact_hallucination": len(hit_words) > 0,
        "hit_words": hit_words
    }

# =====================================================================
# Main Function: Pipeline processing and macro statistics
# =====================================================================
def main():
    INPUT_JSON = "blip_attack_results.json"
    OUTPUT_JSON = "blip_evaluation_new_metrics.json"

    if not os.path.exists(INPUT_JSON):
        print(f"[Error] Input file not found: {INPUT_JSON}")
        return

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[Success] Loaded {len(data)} items. Calculating academic evaluation metrics...\n")

    # Accumulators for macro statistics
    stats = {
        "total_samples": len(data),
        "total_ler": 0.0,
        "total_clean_rep2": 0.0,
        "total_clean_rep3": 0.0,
        "total_clean_rep4": 0.0,
        "total_adv_rep2": 0.0,
        "total_adv_rep3": 0.0,
        "total_adv_rep4": 0.0,
        "degeneration_count": 0,
        "hallucination_count": 0
    }

    for item in data:
        clean_text = item.get("clean_response", "")
        adv_text = item.get("adv_response", "")

        # 1. Calculate efficiency and degeneration metrics
        eff_metrics = evaluate_efficiency_and_degeneration(clean_text, adv_text)
        item["efficiency_metrics"] = eff_metrics
        
        # 2. Calculate targeted hallucination metrics
        hal_metrics = evaluate_targeted_hallucination(clean_text, adv_text)
        item["hallucination_metrics"] = hal_metrics

        # Accumulate statistics
        stats["total_ler"] += eff_metrics["LER"]
        stats["total_clean_rep2"] += eff_metrics["clean_rep_2"]
        stats["total_clean_rep3"] += eff_metrics["clean_rep_3"]
        stats["total_clean_rep4"] += eff_metrics["clean_rep_4"]
        stats["total_adv_rep2"] += eff_metrics["adv_rep_2"]
        stats["total_adv_rep3"] += eff_metrics["adv_rep_3"]
        stats["total_adv_rep4"] += eff_metrics["adv_rep_4"]
        
        if eff_metrics["is_severe_degeneration"]:
            stats["degeneration_count"] += 1
        if hal_metrics["has_artifact_hallucination"]:
            stats["hallucination_count"] += 1

    # Calculate averages
    N = stats["total_samples"]
    if N == 0:
        print("[Error] Sample count is 0, cannot calculate averages.")
        return

    avg_ler = stats["total_ler"] / N
    avg_clean_rep2 = (stats["total_clean_rep2"] / N) * 100
    avg_clean_rep3 = (stats["total_clean_rep3"] / N) * 100
    avg_clean_rep4 = (stats["total_clean_rep4"] / N) * 100
    avg_adv_rep2 = (stats["total_adv_rep2"] / N) * 100
    avg_adv_rep3 = (stats["total_adv_rep3"] / N) * 100
    avg_adv_rep4 = (stats["total_adv_rep4"] / N) * 100
    degeneration_rate = (stats["degeneration_count"] / N) * 100
    tahr_rate = (stats["hallucination_count"] / N) * 100

    # Print statistical table formatted for paper inclusion
    print("="*65)
    print(" 📊 VLM Adversarial Robustness Evaluation Results")
    print("="*65)
    print(f"Total samples: {N}")
    print("\n[I] Inference Efficiency Cost")
    print(f"  ▶ Avg Length Expansion Ratio (Avg LER)  : {avg_ler:.2f}x")
    print(f"  ▶ Catastrophic Degeneration Rate (rep-4 > 20%) : {degeneration_rate:.1f}%")
    
    print("\n[II] N-gram Repetition Ratio Analysis")
    print(f"  ▶ rep-2 (Phrase-level)  | Clean: {avg_clean_rep2:5.2f}%  ->  Adv: {avg_adv_rep2:5.2f}%")
    print(f"  ▶ rep-3 (Clause-level)  | Clean: {avg_clean_rep3:5.2f}%  ->  Adv: {avg_adv_rep3:5.2f}%")
    print(f"  ▶ rep-4 (Core Collapse) | Clean: {avg_clean_rep4:5.2f}%  ->  Adv: {avg_adv_rep4:5.2f}%")

    print("\n[III] Task Accuracy Degradation")
    print(f"  ▶ Targeted Adversarial Hallucination Rate (TAHR): {tahr_rate:.1f}%")
    print("    (Proportion of models manifesting noise as digital artifacts like 'pixels', 'qr codes', etc.)")
    print("="*65)

    # Save results
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
    print(f"\n[Success] Detailed evaluation results saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()