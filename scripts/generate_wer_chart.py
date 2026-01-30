import matplotlib.pyplot as plt
import os

def generate_chart():
    # Data
    models = ['Goan Konkani Base', 'Marathi Base', 'Marathi Base +\nPost-Processing']
    wer_scores = [0.7200, 0.3511, 0.2127]
    colors = ['#ff9999', '#66b3ff', '#99ff99']

    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, wer_scores, color=colors, edgecolor='grey')

    # Add titles and labels
    plt.title('Amchi Konkani ASR: Word Error Rate (WER) Comparison', fontsize=16, pad=20)
    plt.ylabel('Word Error Rate (Lower is Better)', fontsize=12)
    plt.ylim(0, 0.8)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add improvement arrows/text
    # Arrow from Goan to Marathi
    plt.annotate('', xy=(1, 0.36), xytext=(0, 0.72),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color='gray'))
    
    # Arrow from Marathi to Post-Processed
    plt.annotate('', xy=(2, 0.22), xytext=(1, 0.36),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color='gray'))
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save
    output_path = 'wer_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    generate_chart()
