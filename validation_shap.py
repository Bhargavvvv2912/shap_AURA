import sys
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
import shap

def test_shap_compilation():
    print("--- Starting SHAP (NeurIPS 2017) JIT Verification ---")
    
    try:
        # 1. Create a tiny dataset
        X, y = shap.datasets.california(n_points=100)
        model = RandomForestRegressor(n_estimators=10, max_depth=2, random_state=42)
        model.fit(X, y)
        print("--> Model trained.")

        # 2. TRIGGER THE DEP-DRIFT TRAP
        # TreeExplainer triggers Numba-accelerated C++ code.
        # Mismatched Numba/LLVM/NumPy versions will crash here.
        print("--> Initializing TreeExplainer (Triggers Numba JIT)...")
        explainer = shap.TreeExplainer(model)
        
        print("--> Calculating SHAP values...")
        shap_values = explainer.shap_values(X.iloc[:5])
        
        if shap_values is not None:
            print("    [✓] SHAP values computed successfully.")
            
        print("--- SMOKE TEST PASSED ---")

    except ImportError as ie:
        print(f"CRITICAL DEPENDENCY ERROR: {str(ie)}")
        sys.exit(1)
    except Exception as e:
        print(f"CRITICAL VALIDATION FAILURE: {str(e)}")
        # Expecting Numba-related errors here if versions are mismatched.
        sys.exit(1)

if __name__ == "__main__":
    test_shap_compilation()