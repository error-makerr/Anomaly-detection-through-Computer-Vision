"""Model Comparison and Analysis Utilities"""

import os
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class ModelComparison:
    """Compare multiple trained models"""
    
    def __init__(self, model_dirs):
        """
        Args:
            model_dirs: List of model output directories
        """
        self.model_dirs = [Path(d) for d in model_dirs]
        self.results = {}
        self.load_results()
    
    def load_results(self):
        """Load results from all models"""
        print("\n" + "="*70)
        print("LOADING MODEL RESULTS")
        print("="*70)
        
        for model_dir in self.model_dirs:
            results_file = model_dir / "logs" / "test_results.json"
            
            if results_file.exists():
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    model_name = model_dir.name.replace("_output", "")
                    self.results[model_name] = data
                    print(f"✓ Loaded results for {model_name}")
            else:
                print(f"⚠️  Results not found for {model_dir}")
    
    def generate_comparison_table(self):
        """Generate comparison table"""
        print("\n" + "="*70)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*70)
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Test Accuracy (%)': f"{results['test_accuracy']*100:.2f}",
                'Test Loss': f"{results['test_loss']:.4f}",
                'Top-5 Accuracy (%)': f"{results.get('test_top5_accuracy', 0)*100:.2f}",
                'Number of Classes': results['num_classes']
            })
        
        # Create DataFrame
        df = pd.DataFrame(comparison_data)
        
        # Sort by accuracy
        df['Accuracy_Numeric'] = df['Test Accuracy (%)'].astype(float)
        df = df.sort_values('Accuracy_Numeric', ascending=False)
        df = df.drop('Accuracy_Numeric', axis=1)
        
        print("\n" + "="*70)
        print(df.to_string(index=False))
        print("="*70)
        
        # Save to CSV
        df.to_csv('model_comparison.csv', index=False)
        print(f"\n✓ Comparison table saved to: model_comparison.csv")
        
        # Save to JSON
        with open('model_comparison.json', 'w') as f:
            json.dump(comparison_data, f, indent=2)
        print(f"✓ Comparison data saved to: model_comparison.json")
        
        return df
    
    def plot_comparison_charts(self):
        """Generate comparison visualizations"""
        print("\n📊 Generating comparison charts...")
        
        if len(self.results) == 0:
            print("⚠️  No results to compare")
            return
        
        models = list(self.results.keys())
        accuracies = [self.results[m]['test_accuracy'] * 100 for m in models]
        losses = [self.results[m]['test_loss'] for m in models]
        top5_accs = [self.results[m].get('test_top5_accuracy', 0) * 100 for m in models]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Color scheme
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        # 1. Test Accuracy Comparison
        bars1 = axes[0].bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        axes[0].set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0].set_ylim([0, 100])
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Test Loss Comparison
        bars2 = axes[1].bar(models, losses, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        axes[1].set_title('Test Loss Comparison', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, loss in zip(bars2, losses):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{loss:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Top-5 Accuracy Comparison
        bars3 = axes[2].bar(models, top5_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        axes[2].set_title('Top-5 Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Top-5 Accuracy (%)', fontsize=12)
        axes[2].set_ylim([0, 100])
        axes[2].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars3, top5_accs):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('model_comparison_charts.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Comparison charts saved to: model_comparison_charts.png")
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n📝 Generating summary report...")
        
        report_file = 'model_comparison_report.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("DEFECT DETECTION MODEL COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Individual model results
            f.write("-"*80 + "\n")
            f.write("INDIVIDUAL MODEL RESULTS\n")
            f.write("-"*80 + "\n\n")
            
            for model_name, results in self.results.items():
                f.write(f"{model_name}:\n")
                f.write(f"  Test Accuracy: {results['test_accuracy']*100:.2f}%\n")
                f.write(f"  Test Loss: {results['test_loss']:.4f}\n")
                f.write(f"  Top-5 Accuracy: {results.get('test_top5_accuracy', 0)*100:.2f}%\n")
                f.write(f"  Number of Classes: {results['num_classes']}\n\n")
            
            # Best model
            f.write("-"*80 + "\n")
            f.write("BEST PERFORMING MODEL\n")
            f.write("-"*80 + "\n\n")
            
            best_model = max(self.results.items(), key=lambda x: x[1]['test_accuracy'])
            f.write(f"🏆 Winner: {best_model[0]}\n")
            f.write(f"   Accuracy: {best_model[1]['test_accuracy']*100:.2f}%\n")
            f.write(f"   Loss: {best_model[1]['test_loss']:.4f}\n\n")
            
            # Recommendations
            f.write("-"*80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("-"*80 + "\n\n")
            
            accuracies = {name: results['test_accuracy'] for name, results in self.results.items()}
            
            if max(accuracies.values()) < 0.80:
                f.write("⚠️  All models show accuracy < 80%\n")
                f.write("   Consider:\n")
                f.write("   - Training for more epochs\n")
                f.write("   - Fine-tuning pre-trained layers\n")
                f.write("   - Increasing model complexity\n")
                f.write("   - Data augmentation strategies\n\n")
            
            if max(accuracies.values()) >= 0.80:
                f.write("✓ Good performance achieved!\n")
                f.write(f"  Best model ({best_model[0]}) is recommended for deployment.\n\n")
            
            f.write("="*80 + "\n")
        
        print(f"✓ Summary report saved to: {report_file}")
