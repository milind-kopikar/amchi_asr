import json
import os

def extract_predictions():
    results_path = os.path.join('nemo_experiments', 'marathi_pilot_v3', 'final_test_results.json')
    output_path = 'marathi_predictions.txt'
    
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return
        
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    predictions = [sample.get('prediction', '') for sample in data.get('per_sample', [])]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(pred + '\n')
            
    print(f"Successfully extracted {len(predictions)} predictions to {output_path}")

if __name__ == "__main__":
    extract_predictions()
