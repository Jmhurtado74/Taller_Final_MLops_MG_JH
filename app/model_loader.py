import boto3
import os
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

def download_model():
    s3 = boto3.client('s3', 
                      aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
                      aws_secret_access_key=os.getenv('AWS_SECRET_KEY'))
    
    # El nombre del bucket y archivo vienen de variables de entorno
    bucket = os.getenv('AWS_BUCKET')
    file_key = os.getenv('MODEL_FILE')
    
    # Obtener el nombre del archivo desde la ruta (ej: "iris.onnx" de "/mlops-ci-cd/model-iris/iris.onnx")
    model_filename = os.path.basename(file_key)
    
    # Descargar en la carpeta raíz del proyecto
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_path = os.path.join(project_root, model_filename)
    
    print(f"Descargando modelo desde S3: {file_key}")
    print(f"Guardando en: {local_path}")
    s3.download_file(bucket, file_key, local_path)
    print("Modelo descargado exitosamente.")
