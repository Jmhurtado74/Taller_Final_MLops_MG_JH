#!/usr/bin/env python
"""
Script de Pre-entrenamiento (Pre-training Validation)
Carga los datos de prueba CSV y valida/entrena el modelo ONNX.
Este script se ejecuta antes de las pruebas unitarias en el pipeline CI/CD.
"""

import os
import sys
import pandas as pd
import numpy as np
import onnxruntime as ort
from dotenv import load_dotenv
import logging
from class_mapper import decode_prediction

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Variables de configuración
MODEL_FILE = os.getenv("MODEL_FILE", "iris.onnx")
ENV = os.getenv("ENV", "dev")

# Rutas
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILENAME = os.path.basename(MODEL_FILE)
MODEL_PATH = os.path.join(PROJECT_ROOT, MODEL_FILENAME)
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, 'test_data')


def load_training_data():
    """Carga todos los datos CSV de la carpeta test_data."""
    
    logger.info("="*60)
    logger.info("CARGANDO DATOS DE ENTRENAMIENTO")
    logger.info("="*60)
    logger.info(f"Directorio: {TEST_DATA_DIR}")
    
    if not os.path.exists(TEST_DATA_DIR):
        logger.error(f"[✗] Directorio {TEST_DATA_DIR} no existe")
        return None
    
    csv_files = [f for f in os.listdir(TEST_DATA_DIR) if f.endswith('.csv')]
    
    if not csv_files:
        logger.warning(f"[!] No se encontraron archivos CSV en {TEST_DATA_DIR}")
        return None
    
    logger.info(f"[✓] Encontrados {len(csv_files)} archivo(s) CSV")
    
    # Combinar todos los archivos CSV
    all_data = []
    total_rows = 0
    
    for csv_file in csv_files:
        file_path = os.path.join(TEST_DATA_DIR, csv_file)
        logger.info(f"\n  [*] Cargando: {csv_file}")
        
        try:
            df = pd.read_csv(file_path)
            all_data.append(df)
            total_rows += len(df)
            
            logger.info(f"      - Filas: {len(df)}")
            logger.info(f"      - Columnas: {len(df.columns)}")
            logger.info(f"      - Nombres: {list(df.columns)}")
            logger.info(f"      - Tipos de datos:")
            for col, dtype in df.dtypes.items():
                logger.info(f"        · {col}: {dtype}")
            logger.info(f"      - Valores nulos por columna:")
            for col, nulls in df.isnull().sum().items():
                if nulls > 0:
                    logger.info(f"        · {col}: {nulls}")
            
        except Exception as e:
            logger.error(f"      [✗] Error cargando {csv_file}: {e}")
            continue
    
    if not all_data:
        logger.error("[✗] No se pudieron cargar datos de ningún archivo")
        return None
    
    # Combinar todos los dataframes
    combined_data = pd.concat(all_data, ignore_index=True)
    
    logger.info("\n" + "="*60)
    logger.info("RESUMEN DE DATOS COMBINADOS")
    logger.info("="*60)
    logger.info(f"[✓] Total de filas: {combined_data.shape[0]}")
    logger.info(f"[✓] Total de columnas: {combined_data.shape[1]}")
    logger.info(f"[✓] Nombres de columnas: {list(combined_data.columns)}")
    logger.info(f"[✓] Forma (shape): {combined_data.shape}")
    logger.info(f"[✓] Memoria usada: {combined_data.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # Estadísticas descriptivas
    logger.info("\n[*] Estadísticas descriptivas:")
    logger.info(f"\n{combined_data.describe()}")
    
    return combined_data


def extract_features_and_labels(df):
    """Extrae features y labels del dataframe."""
    
    logger.info("="*60)
    logger.info("EXTRAYENDO FEATURES Y LABELS")
    logger.info("="*60)
    
    # Identificar columnas de features (todas excepto la última que suele ser target)
    feature_cols = [col for col in df.columns if col.lower() not in ['target', 'label', 'species', 'class']]
    label_col = [col for col in df.columns if col.lower() in ['target', 'label', 'species', 'class']]
    
    if not feature_cols:
        logger.error("[✗] No se encontraron columnas de features")
        return None, None
    
    logger.info(f"[✓] Features identificadas: {len(feature_cols)}")
    for i, col in enumerate(feature_cols, 1):
        logger.info(f"    {i}. {col}")
    
    X = df[feature_cols].values.astype(np.float32)
    
    logger.info(f"\n[*] Conversión de features a numpy array:")
    logger.info(f"    - Shape: {X.shape}")
    logger.info(f"    - Dtype: {X.dtype}")
    logger.info(f"    - Rango de valores:")
    logger.info(f"      · Min: {X.min():.6f}")
    logger.info(f"      · Max: {X.max():.6f}")
    logger.info(f"      · Mean: {X.mean():.6f}")
    logger.info(f"      · Std: {X.std():.6f}")
    
    # Label es opcional
    y = None
    if label_col:
        y = df[label_col[0]].values
        logger.info(f"\n[✓] Label identificada: {label_col[0]}")
        logger.info(f"    - Shape: {y.shape}")
        logger.info(f"    - Dtype: {y.dtype}")
        logger.info(f"    - Valores únicos: {np.unique(y)}")
        
        # Distribución de clases
        unique, counts = np.unique(y, return_counts=True)
        logger.info(f"    - Distribución de clases:")
        for cls, count in zip(unique, counts):
            pct = (count / len(y)) * 100
            logger.info(f"      · Clase '{cls}': {count} muestras ({pct:.2f}%)")
    else:
        logger.warning("[!] No se encontró columna de label/target")
    
    logger.info("\n[✓] Extracción completada exitosamente")
    
    return X, y


def validate_model_with_data(X, y=None):
    """Valida el modelo ONNX con los datos de entrenamiento."""
    
    logger.info("\n" + "="*60)
    logger.info("VALIDACIÓN DEL MODELO CON DATOS DE ENTRENAMIENTO")
    logger.info("="*60)
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Modelo no encontrado en: {MODEL_PATH}")
        return False
    
    try:
        # Cargar modelo ONNX
        logger.info(f"Cargando modelo desde: {MODEL_PATH}")
        ort_session = ort.InferenceSession(MODEL_PATH)
        
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        
        logger.info(f"  Input: {input_name}")
        logger.info(f"  Output: {output_name}")
        
        # Validar dimensiones
        input_shape = ort_session.get_inputs()[0].shape
        logger.info(f"  Input shape esperado: {input_shape}")
        logger.info(f"  Features disponibles: {X.shape[1]}")
        
        if X.shape[1] != input_shape[1]:
            logger.error(f"Mismatch en features: esperado {input_shape[1]}, obtenido {X.shape[1]}")
            return False
        
        # Realizar predicciones (una por una, ya que el modelo espera dim 1)
        logger.info("\n[*] Realizando predicciones...")
        predictions = []
        
        for i in range(len(X)):
            sample = X[i:i+1]  # Mantener dimensión (1, num_features)
            result = ort_session.run(None, {input_name: sample})
            
            # Extraer predicciones
            if isinstance(result[0], np.ndarray):
                pred = result[0].flatten()[0]  # Obtener primer valor escalar
            else:
                pred = result[0]
            
            predictions.append(pred)
        
        predictions = np.array(predictions)
        logger.info(f"[✓] Predicciones completadas: {len(predictions)} muestras")
        
        # Análisis de predicciones
        logger.info("\n[*] Análisis de predicciones:")
        logger.info(f"  - Min: {predictions.min():.6f}")
        logger.info(f"  - Max: {predictions.max():.6f}")
        logger.info(f"  - Mean: {predictions.mean():.6f}")
        logger.info(f"  - Std: {predictions.std():.6f}")
        
        # Validar que no hay NaN o Inf
        if np.any(np.isnan(predictions)):
            logger.error("[✗] Se encontraron valores NaN en predicciones")
            return False
        
        if np.any(np.isinf(predictions)):
            logger.error("[✗] Se encontraron valores Inf en predicciones")
            return False
        
        logger.info("[✓] Todas las predicciones son válidas (sin NaN/Inf)")
        
        # Análisis por clase si hay labels
        if y is not None:
            logger.info("\n[*] Análisis por clase:")
            unique_classes = np.unique(y)
            
            for cls in unique_classes:
                mask = (y == cls)
                class_preds = predictions[mask]
                class_preds_decoded = [decode_prediction(p) for p in class_preds]
                
                # Contar predicciones correctas e incorrectas
                correct = sum(1 for p in class_preds_decoded if p == cls)
                incorrect = len(class_preds_decoded) - correct
                accuracy = (correct / len(class_preds_decoded)) * 100 if len(class_preds_decoded) > 0 else 0
                
                logger.info(f"  Clase '{cls}' ({np.sum(mask)} muestras):")
                logger.info(f"    - Predicción numérica media: {class_preds.mean():.6f}")
                logger.info(f"    - Rango numérico: [{class_preds.min():.6f}, {class_preds.max():.6f}]")
                logger.info(f"    - Predicciones correctas: {correct}/{len(class_preds_decoded)} ({accuracy:.2f}%)")
                logger.info(f"    - Distribución de predicciones: {dict(pd.Series(class_preds_decoded).value_counts())}")
        
        logger.info("\n[✓] Validación del modelo completada exitosamente")
        return True
        
    except Exception as e:
        logger.error(f"Error durante validación: {e}")
        import traceback
        traceback.print_exc()
        return False


def calculate_model_metrics(X, y=None):
    """Calcula métricas del modelo para monitoreo."""
    
    logger.info("\n" + "="*60)
    logger.info("CÁLCULO DE MÉTRICAS DEL MODELO")
    logger.info("="*60)
    
    try:
        ort_session = ort.InferenceSession(MODEL_PATH)
        input_name = ort_session.get_inputs()[0].name
        
        # Realizar predicciones (una por una)
        predictions = []
        for i in range(len(X)):
            sample = X[i:i+1]  # Mantener dimensión (1, num_features)
            result = ort_session.run(None, {input_name: sample})
            if isinstance(result[0], np.ndarray):
                pred = result[0].flatten()[0]
            else:
                pred = result[0]
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Métricas básicas
        metrics = {
            'mean_prediction': float(predictions.mean()),
            'std_prediction': float(predictions.std()),
            'min_prediction': float(predictions.min()),
            'max_prediction': float(predictions.max()),
            'num_samples': len(predictions),
            'num_features': X.shape[1]
        }
        
        logger.info("\n[*] Métricas calculadas:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculando métricas: {e}")
        return None


def pretrain_workflow():
    """Ejecuta el workflow completo de pre-entrenamiento."""
    
    logger.info("="*60)
    logger.info("INICIO DE PRE-ENTRENAMIENTO")
    logger.info(f"Entorno: {ENV}")
    logger.info("="*60)
    
    # Paso 1: Cargar datos
    logger.info("\n[PASO 1] Cargando datos de entrenamiento...")
    logger.info("-"*60)
    df_train = load_training_data()
    
    if df_train is None:
        logger.error("No se pudieron cargar los datos de entrenamiento")
        return False
    
    # Paso 2: Extraer features y labels
    logger.info("\n[PASO 2] Extrayendo features y labels...")
    logger.info("-"*60)
    X, y = extract_features_and_labels(df_train)
    
    if X is None:
        logger.error("No se pudieron extraer features")
        return False
    
    # Paso 3: Validar modelo con datos
    logger.info("\n[PASO 3] Validando modelo con datos...")
    logger.info("-"*60)
    if not validate_model_with_data(X, y):
        logger.error("Validación del modelo falló")
        return False
    
    # Paso 4: Calcular métricas
    logger.info("\n[PASO 4] Calculando métricas...")
    logger.info("-"*60)
    metrics = calculate_model_metrics(X, y)
    
    if metrics is None:
        logger.warning("No se pudieron calcular métricas")
    
    logger.info("\n" + "="*60)
    logger.info("✓ PRE-ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    logger.info("="*60)
    
    return True


if __name__ == "__main__":
    success = pretrain_workflow()
    sys.exit(0 if success else 1)
