"""
Mapeo de clases para el modelo Iris ONNX.
Convierte entre índices numéricos y nombres de especies.
"""

# Mapeo de índices a nombres de especies (orden típico en dataset Iris)
CLASS_MAPPING = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

# Mapeo inverso
REVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}


def decode_prediction(prediction):
    """
    Convierte una predicción numérica a nombre de clase.
    
    Args:
        prediction: float o int - índice de clase o probabilidad
    
    Returns:
        str: nombre de la clase
    """
    # Si es un número flotante, redondearlo al índice más cercano
    if isinstance(prediction, float):
        class_idx = int(round(prediction))
    else:
        class_idx = int(prediction)
    
    # Asegurar que el índice está en rango
    if class_idx not in CLASS_MAPPING:
        class_idx = max(0, min(class_idx, len(CLASS_MAPPING) - 1))
    
    return CLASS_MAPPING[class_idx]


def encode_class(class_name):
    """
    Convierte un nombre de clase a su índice numérico.
    
    Args:
        class_name: str - nombre de la clase
    
    Returns:
        int: índice de la clase
    """
    class_name_lower = class_name.lower()
    return REVERSE_CLASS_MAPPING.get(class_name_lower, 0)


def get_all_classes():
    """Retorna lista de todos los nombres de clases."""
    return list(CLASS_MAPPING.values())


def get_all_class_indices():
    """Retorna lista de todos los índices de clases."""
    return list(CLASS_MAPPING.keys())
