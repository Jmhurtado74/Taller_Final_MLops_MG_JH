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
        
        prediction = result[0][0] if isinstance(result[0], np.ndarray) else result[0]
        
        print(f"  Input: {test_input}")
        print(f"  Output: {prediction}")
        print("  ✓ Modelo respondió correctamente")
        
        # El resultado debe ser un número válido
        assert isinstance(float(prediction), float), "El resultado no es un número válido"


    def test_download_data_files(self):
        """
        Prueba 2: Verificar que los archivos de datos de prueba se descargaron correctamente.
        """
        print("\n[TEST 2] Verificando descarga de archivos de datos de prueba...")
        
        assert len(self.test_data_files) > 0, "No se descargaron archivos de datos de prueba"
        
        for file_path in self.test_data_files:
            assert os.path.exists(file_path), f"El archivo {file_path} no existe"
            df = pd.read_csv(file_path)
            assert not df.empty, f"El archivo {file_path} está vacío"
            print(f"  ✓ Archivo {file_path} descargado y contiene datos")
        
        print("  ✓ Todos los archivos de datos de prueba están disponibles y son válidos")

    def test_download_model_file(self):
        """
        Prueba 3: Verificar que el archivo del modelo ONNX se descargó correctamente.
        """
        print("\n[TEST 3] Verificando descarga del archivo del modelo ONNX...")
        
        assert os.path.exists(MODEL_PATH), f"El archivo del modelo {MODEL_PATH} no existe"
        
        # Intentar cargar el modelo para verificar su validez
        try:
            ort.InferenceSession(MODEL_PATH)
            print(f"  ✓ Archivo del modelo {MODEL_PATH} descargado y es válido")
        except Exception as e:
            pytest.fail(f"El archivo del modelo no es válido: {e}")
    

if __name__ == "__main__":
    # Ejecutar pruebas con pytest
    exit_code = pytest.main([__file__, "-v", "-s"])
    sys.exit(exit_code)
