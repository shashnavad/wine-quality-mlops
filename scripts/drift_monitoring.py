# scripts/drift_monitoring.py
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

def track_drift_metrics():
    """Track and visualize data drift metrics over time"""
    # Path to Great Expectations validation results
    results_dir = "great_expectations/uncommitted/validations/wine_quality_suite/"
    
    # Collect metrics from all validation runs
    metrics_over_time = []
    
    for result_file in os.listdir(results_dir):
        if result_file.endswith('.json'):
            with open(os.path.join(results_dir, result_file), 'r') as f:
                result = json.load(f)
                
                # Extract timestamp
                timestamp = result.get('meta', {}).get('run_id', 'unknown')
                if timestamp == 'unknown':
                    timestamp = result_file.split('.')[0]
                
                # Extract key metrics
                metrics = {
                    'timestamp': timestamp,
                    'success': result.get('success', False),
                    'evaluated_expectations': result.get('statistics', {}).get('evaluated_expectations', 0),
                    'successful_expectations': result.get('statistics', {}).get('successful_expectations', 0)
                }
                
                metrics_over_time.append(metrics)
    
    # Convert to DataFrame
    df = pd.DataFrame(metrics_over_time)
    if not df.empty:
        df['success_rate'] = df['successful_expectations'] / df['evaluated_expectations']
        
        # Plot drift metrics
        plt.figure(figsize=(10, 6))
        plt.plot(df['timestamp'], df['success_rate'], marker='o')
        plt.title('Data Drift Monitoring')
        plt.ylabel('Expectation Success Rate')
        plt.xlabel('Validation Run')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        os.makedirs('reports/figures', exist_ok=True)
        plt.savefig(f'reports/figures/drift_metrics_{datetime.now().strftime("%Y%m%d")}.png')
        
        return df
    return None

if __name__ == "__main__":
    track_drift_metrics()