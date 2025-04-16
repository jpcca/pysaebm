import numpy as np
import base64
import io
from typing import Dict, Any

def kde_dict_to_json_serializable(kde_dict):
    """
    Convert a KDE dictionary to a JSON-serializable format.
    
    Args:
        kde_dict: Dictionary containing FastKDE objects
        
    Returns:
        Dictionary that can be directly saved to JSON with json.dump()
    """
    serializable_dict = {}
    
    for biomarker, kde_data in kde_dict.items():
        # Serialize theta KDE
        theta_kde = kde_data['theta_kde']
        theta_data_buffer = io.BytesIO()
        np.save(theta_data_buffer, theta_kde.data)
        theta_weights_buffer = io.BytesIO()
        np.save(theta_weights_buffer, theta_kde.weights)
        
        # Serialize phi KDE
        phi_kde = kde_data['phi_kde']
        phi_data_buffer = io.BytesIO()
        np.save(phi_data_buffer, phi_kde.data)
        phi_weights_buffer = io.BytesIO()
        np.save(phi_weights_buffer, phi_kde.weights)
        
        # Store in serializable dictionary
        serializable_dict[biomarker] = {
            'theta_kde': {
                'data': base64.b64encode(theta_data_buffer.getvalue()).decode('utf-8'),
                'weights': base64.b64encode(theta_weights_buffer.getvalue()).decode('utf-8'),
                'bandwidth': float(theta_kde.bandwidth)
            },
            'phi_kde': {
                'data': base64.b64encode(phi_data_buffer.getvalue()).decode('utf-8'),
                'weights': base64.b64encode(phi_weights_buffer.getvalue()).decode('utf-8'),
                'bandwidth': float(phi_kde.bandwidth)
            }
        }
    
    return serializable_dict

def json_serializable_to_kde_dict(serializable_dict):
    """
    Convert a JSON-serializable dictionary back to a KDE dictionary.
    
    Args:
        serializable_dict: Dictionary loaded from JSON
        
    Returns:
        Dictionary containing FastKDE objects
    """
    from your_module import FastKDE  # Import your FastKDE class
    
    kde_dict = {}
    
    for biomarker, kde_data in serializable_dict.items():
        # Deserialize theta KDE
        theta_data = np.load(io.BytesIO(base64.b64decode(kde_data['theta_kde']['data'])))
        theta_weights = np.load(io.BytesIO(base64.b64decode(kde_data['theta_kde']['weights'])))
        theta_bandwidth = kde_data['theta_kde']['bandwidth']
        
        # Deserialize phi KDE
        phi_data = np.load(io.BytesIO(base64.b64decode(kde_data['phi_kde']['data'])))
        phi_weights = np.load(io.BytesIO(base64.b64decode(kde_data['phi_kde']['weights'])))
        phi_bandwidth = kde_data['phi_kde']['bandwidth']
        
        # Create FastKDE objects
        theta_kde = FastKDE(data=theta_data, weights=theta_weights, bandwidth=theta_bandwidth)
        phi_kde = FastKDE(data=phi_data, weights=phi_weights, bandwidth=phi_bandwidth)
        
        # Store in KDE dictionary
        kde_dict[biomarker] = {
            'theta_kde': theta_kde,
            'phi_kde': phi_kde
        }
    
    return kde_dict