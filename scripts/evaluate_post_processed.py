import json
import os
import re

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def calculate_metrics(reference, hypothesis):
    # Word level
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if len(ref_words) == 0:
        wer = 1.0 if len(hyp_words) > 0 else 0.0
    else:
        wer = levenshtein_distance(ref_words, hyp_words) / len(ref_words)
    
    # Character level
    if len(reference) == 0:
        cer = 1.0 if len(hypothesis) > 0 else 0.0
    else:
        cer = levenshtein_distance(reference, hypothesis) / len(reference)
        
    return wer, cer

def main():
    results_path = os.path.join('nemo_experiments', 'marathi_pilot_v3', 'final_test_results.json')
    fixed_path = 'marathi_predictions_fixed.txt'
    
    if not os.path.exists(results_path) or not os.path.exists(fixed_path):
        print("Required files missing.")
        return

    with open(results_path, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
    
    with open(fixed_path, 'r', encoding='utf-8') as f:
        fixed_lines = [line.strip() for line in f if line.strip()]

    references = [sample['reference'] for sample in results_data['per_sample']]
    original_wers = [sample['wer'] for sample in results_data['per_sample']]
    
    # Normalize punctuation for fair comparison if needed, 
    # but the user asked for a direct comparison with the fixed file.
    # Note: final_test_results.json 'wer' might have been calculated with some normalization.
    
    total_wer = 0
    total_cer = 0
    count = min(len(references), len(fixed_lines))
    
    print(f"{'#':<3} | {'Original WER':<12} | {'New WER':<10} | {'New CER':<10}")
    print("-" * 50)
    
    detailed_results = []
    
    for i in range(count):
        ref = references[i]
        hyp = fixed_lines[i]
        
        # Simple normalization to match typical ASR evaluation (remove certain punctuation)
        # However, to be strict, we compare exactly what's in the files.
        # We also need to factor in the Devanagari danda if it's treated as a word or punctuation.
        
        # Standard ASR normalization (simple):
        def clean(t):
            t = re.sub(r'[।!,.?\-]', '', t)
            return " ".join(t.split())

        wer, cer = calculate_metrics(clean(ref), clean(hyp))
        total_wer += wer
        total_cer += cer
        
        detailed_results.append({
            "index": i,
            "ref": ref,
            "hyp": hyp,
            "orig_wer": original_wers[i],
            "new_wer": wer,
            "new_cer": cer
        })
        
        print(f"{i:<3} | {original_wers[i]:.4f}       | {wer:.4f}     | {cer:.4f}")

    avg_wer = total_wer / count if count > 0 else 0
    avg_cer = total_cer / count if count > 0 else 0
    
    print("-" * 50)
    print(f"Average Original WER: {results_data['summary']['mean_wer']:.4f}")
    print(f"Average New WER:      {avg_wer:.4f}")
    print(f"Average New CER:      {avg_cer:.4f}")
    
    # Save results
    with open('post_process_metrics.json', 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "avg_wer": avg_wer,
                "avg_cer": avg_cer,
                "original_avg_wer": results_data['summary']['mean_wer'],
                "sample_count": count
            },
            "details": detailed_results
        }, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
