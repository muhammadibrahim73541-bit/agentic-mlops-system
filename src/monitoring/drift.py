import pandas as pd
import numpy as np
import glob
import os
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

def detect_drift_simple():
    """Simple drift detection without evidently (using KS test)"""
    
    ref_files = glob.glob("data/processed/features_*.csv")
    current_files = glob.glob("data/raw/energy_demand_*.csv")
    
    if not ref_files or not current_files:
        print("Missing data files")
        return
    
    ref_data = pd.read_csv(max(ref_files, key=os.path.getctime))
    current = pd.read_csv(max(current_files, key=os.path.getctime))
    
    # Add features to current
    current['temp_squared'] = current['temp'] ** 2
    current['temp_humidity'] = current['temp'] * current['humidity']
    current['hour_sin'] = np.sin(2 * np.pi * current['hour'] / 24)
    current['hour_cos'] = np.cos(2 * np.pi * current['hour'] / 24)
    
    current_features = current[['temp', 'humidity', 'pressure', 'wind_speed']]
    
    # Kolmogorov-Smirnov test for drift
    drift_report = []
    for col in ['temp', 'humidity']:
        stat, p_value = stats.ks_2samp(ref_data[col], current[col])
        drift_detected = p_value < 0.05
        drift_report.append({
            'feature': col,
            'drift_detected': drift_detected,
            'p_value': p_value,
            'statistic': stat
        })
    
    # Save report as HTML
    html = "<h1>Data Drift Report</h1>"
    html += "<table border='1'><tr><th>Feature</th><th>Drift?</th><th>P-Value</th></tr>"
    for item in drift_report:
        color = "red" if item['drift_detected'] else "green"
        html += f"<tr style='color:{color}'><td>{item['feature']}</td><td>{item['drift_detected']}</td><td>{item['p_value']:.4f}</td></tr>"
    html += "</table>"
    html += "<p><small>P < 0.05 indicates significant drift</small></p>"
    
    with open("data/drift_report.html", "w") as f:
        f.write(html)
    
    print("✅ Drift report saved to data/drift_report.html")
    
    # Also save SHAP
    try:
        model = joblib.load("models/demand_model.pkl")
        X = ref_data.drop('energy_demand_mw', axis=1)
        
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, show=False)
        plt.savefig("data/shap_summary.png", bbox_inches='tight', dpi=150)
        print("✅ SHAP plot saved to data/shap_summary.png")
    except Exception as e:
        print(f"SHAP skipped: {e}")

if __name__ == "__main__":
    detect_drift_simple()