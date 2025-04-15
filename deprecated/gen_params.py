# alabebm/data/gen_params.py
import json
import numpy as np 
import math

base_params = {
  "MMSE": {
    "theta_mean": 22,
    "theta_std": 2.6666666666666665,
    "phi_mean": 28,
    "phi_std": 0.6666666666666666,
  },
  "ADAS": {
    "theta_mean": -20,
    "theta_std": 4.0,
    "phi_mean": -6,
    "phi_std": 1.3333333333333333,
  },
  "AB": {
    "theta_mean": 150,
    "theta_std": 16.666666666666668,
    "phi_mean": 250,
    "phi_std": 50.0,
  },
  "P-Tau": {
    "theta_mean": -50,
    "theta_std": 33.333333333333336,
    "phi_mean": -25,
    "phi_std": 16.666666666666668,
  },
  "HIP-FCI": {
    "theta_mean": -5,
    "theta_std": 6.666666666666667,
    "phi_mean": 5,
    "phi_std": 1.6666666666666667,
  },
  "HIP-GMI": {
    "theta_mean": 0.3,
    "theta_std": 0.3333333333333333,
    "phi_mean": 0.4,
    "phi_std": 0.2333333333333333,
  },
  "AVLT-Sum": {
    "theta_mean": 20,
    "theta_std": 6.666666666666667,
    "phi_mean": 40,
    "phi_std": 15.0,
  },
  "PCC-FCI": {
    "theta_mean": 5,
    "theta_std": 3.3333333333333335,
    "phi_mean": 12,
    "phi_std": 4.0,
  },
  "FUS-GMI": {
    "theta_mean": 0.5,
    "theta_std": 0.06666666666666667,
    "phi_mean": 0.6,
    "phi_std": 0.06666666666666667,
  },
  "FUS-FCI": {
    "theta_mean": -20,
    "theta_std": 6.0,
    "phi_mean": -10,
    "phi_std": 3.3333333333333335,
  }
}

def calculate_rho(theta_mean, theta_std, phi_mean, phi_std):
    R_i = theta_mean - phi_mean
    denominator = math.sqrt(theta_std**2 + phi_std**2)
    rho_i = max(1, abs(R_i) / denominator)
    return R_i, rho_i

def generate_params():
    new_params = {}
    
    for biomarker, params in base_params.items():
        # Calculate R_i and rho_i
        R_i, rho_i = calculate_rho(
            params["theta_mean"], 
            params["theta_std"],
            params["phi_mean"],
            params["phi_std"]
        )
        
        # Create new entry
        new_entry = {
            "theta_mean": params["theta_mean"],
            "theta_std": params["theta_std"],
            "phi_mean": params["phi_mean"],
            "phi_std": params["phi_std"],
            "R_i": R_i,
            "rho_i": rho_i,
        }
        
        # Convert numpy types to native Python types
        for k, v in new_entry.items():
            if isinstance(v, np.generic):
                new_entry[k] = v.item()
                
        new_params[biomarker] = new_entry
    
    return new_params

if __name__ == "__main__":
    updated_params = generate_params()
    
    with open("params.json", "w") as f:
        json.dump(updated_params, f, indent=2)
        
    print("params.json generated successfully!")

