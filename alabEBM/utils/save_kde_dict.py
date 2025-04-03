import numpy as np
import base64
import io
from typing import Dict, Any

def kde_dict_to_json_serializable(kde_dict):
    """
    Convert a KDE dictionary to a JSON-serializable format.
    
    Args:
        kde_dict: Dictionary containing FastKDE objects and weights
        
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
        
        # Serialize the biomarker's weights
        bio_theta_weights_buffer = io.BytesIO()
        np.save(bio_theta_weights_buffer, kde_data['theta_weights'])
        bio_phi_weights_buffer = io.BytesIO()
        np.save(bio_phi_weights_buffer, kde_data['phi_weights'])
        
        # Store in serializable dictionary
        serializable_dict[biomarker] = {
            'theta_kde': {
                'data': base64.b64encode(theta_data_buffer.getvalue()).decode('utf-8'),
                'weights': base64.b64encode(theta_weights_buffer.getvalue()).decode('utf-8'),
                'bandwidth': float(theta_kde.bandwidth)
            },
            'theta_weights': base64.b64encode(bio_theta_weights_buffer.getvalue()).decode('utf-8'),
            'phi_kde': {
                'data': base64.b64encode(phi_data_buffer.getvalue()).decode('utf-8'),
                'weights': base64.b64encode(phi_weights_buffer.getvalue()).decode('utf-8'),
                'bandwidth': float(phi_kde.bandwidth)
            },
            'phi_weights': base64.b64encode(bio_phi_weights_buffer.getvalue()).decode('utf-8')
        }
    
    return serializable_dict
