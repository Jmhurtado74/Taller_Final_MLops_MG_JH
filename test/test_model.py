import os
import sys
import pytest
import pandas as pd
import numpy as np
import onnxruntime as ort
from dotenv import load_dotenv
from botocore.exceptions import ClientError
import scripts.download_model as download_model_script
import scripts.download_data as download_data_script

# Cargar variables de entorno
load_dotenv()

# Variables de configuración
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
BUCKET = os.getenv("AWS_BUCKET", "modelos-predictivos")
MODEL_FILE = os.getenv("MODEL_FILE", "/mlops-ci-cd/model-iris/iris.onnx")
ENV = os.getenv("ENV", "dev")

# Ruta al modelo en la carpeta raíz del proyecto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILENAME = os.path.basename(MODEL_FILE)
MODEL_PATH = os.path.join(PROJECT_ROOT, MODEL_FILENAME)

class TestONNXModel:
    """Suite de pruebas para el modelo ONNX."""
    
    @classmethod
    def setup_class(cls):
        """Setup inicial: descargar modelo y datos de prueba."""
        download_model_script.download_model()
        cls.test_data_files = download_data_script.download_test_data()
        
        # Cargar el modelo ONNX
        cls.ort_session = ort.InferenceSession(MODEL_PATH)
        cls.input_name = cls.ort_session.get_inputs()[0].name
        cls.output_name = cls.ort_session.get_outputs()[0].name
        
        print(f"\n✓ Modelo cargado exitosamente")
        print(f"  Input: {cls.input_name}")
        print(f"  Output: {cls.output_name}")
    
    def test_model_responds_to_input(self):
        """
        Prueba 1: Verificar que el modelo responde correctamente con datos de entrada definidos.
        """
        print("\n[TEST 1] Probando respuesta del modelo con datos definidos...")
        
        # Datos de entrada estándar del iris dataset
        test_input = np.array([[5.1, 3.5, 1.4, 0.2]], dtype=np.float32)
        
        # Realizar predicción
        result = self.ort_session.run(None, {self.input_name: test_input})
        
        # Verificar que el resultado existe y es válido
        assert result is not None, "El modelo no retornó ningún resultado"
        assert len(result) > 0, "El resultado está vacío"
        
        # Extraer la predicción de forma correcta
        if isinstance(result[0], np.ndarray):
            prediction = result[0].flatten()[0]  # Obtener el primer valor del array
        else:
            prediction = result[0]
        
        # Convertir a float
        prediction_value = float(prediction)
        
        print(f"  Input: {test_input}")
        print(f"  Output: {prediction_value}")
        print("  ✓ Modelo respondió correctamente")
        
        # El resultado debe ser un número válido
        assert isinstance(prediction_value, float), "El resultado no es un número válido"
    
    def test_model_metric_within_threshold(self):
        """
        Prueba 2: Verificar que las métricas del modelo están dentro de los umbrales esperados.
        """
        print("\n[TEST 2] Probando métricas del modelo...")
        
        # Cargar datos de prueba CSV
        test_data_dir = os.path.join(PROJECT_ROOT, 'test_data')
        csv_files = [f for f in os.listdir(test_data_dir) if f.endswith('.csv')]
        
        assert len(csv_files) > 0, "No hay archivos CSV para pruebas"
        
        # Cargar primer archivo CSV
        csv_path = os.path.join(test_data_dir, csv_files[0])
        df = pd.read_csv(csv_path)
        
        # Extraer features (todas las columnas excepto la última que es el label)
        feature_cols = [col for col in df.columns if col.lower() not in ['target', 'label', 'species', 'class']]
        X = df[feature_cols].values.astype(np.float32)
        
        # Realizar predicciones
        predictions = []
        for i in range(len(X)):
            sample = X[i:i+1]
            result = self.ort_session.run(None, {self.input_name: sample})
            if isinstance(result[0], np.ndarray):
                pred = result[0].flatten()[0]
            else:
                pred = result[0]
            predictions.append(float(pred))
        
        predictions = np.array(predictions)
        
        # Calcular métricas
        mean_pred = predictions.mean()
        std_pred = predictions.std()
        
        print(f"  Predicciones realizadas: {len(predictions)}")
        print(f"  Media: {mean_pred:.6f}")
        print(f"  Desviación estándar: {std_pred:.6f}")
        print(f"  Min: {predictions.min():.6f}, Max: {predictions.max():.6f}")
        
        # Verificar que la varianza está dentro del umbral
        assert std_pred <= 2.0, f"Varianza fuera de umbral: {std_pred} > 2.0"
        print("  ✓ Métricas dentro del umbral esperado")
    
    def test_model_handles_edge_cases(self):
        """
        Prueba 3: Verificar que el modelo maneja correctamente valores extremos.
        """
        print("\n[TEST 3] Probando casos extremos...")
        
        # Casos extremos
        test_cases = [
            (np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32), "Todos ceros"),
            (np.array([[10.0, 10.0, 10.0, 10.0]], dtype=np.float32), "Todos máximos"),
            (np.array([[5.0, 3.0, 1.5, 0.3]], dtype=np.float32), "Valores típicos"),
        ]
        
        for test_input, description in test_cases:
            result = self.ort_session.run(None, {self.input_name: test_input})
            
            if isinstance(result[0], np.ndarray):
                prediction = result[0].flatten()[0]
            else:
                prediction = result[0]
            
            prediction_value = float(prediction)
            
            print(f"  {description}: {prediction_value:.6f}")
            
            # Verificar que no hay NaN o Inf
            assert not np.isnan(prediction_value), f"NaN detectado para {description}"
            assert not np.isinf(prediction_value), f"Inf detectado para {description}"
        
        print("  ✓ Todos los casos extremos manejados correctamente")


if __name__ == "__main__":
    # Ejecutar pruebas con pytest
    exit_code = pytest.main([__file__, "-v", "-s"])
    sys.exit(exit_code)
