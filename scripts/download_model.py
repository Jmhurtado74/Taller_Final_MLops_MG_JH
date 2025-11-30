#!/usr/bin/env python
"""
Script para descargar el modelo ONNX desde S3.
Usado durante la construcción del contenedor Docker.
"""

import os
import boto3
from dotenv import load_dotenv

def download_model():
    """Descarga el modelo ONNX desde S3."""
    
    # Cargar variables de entorno
    load_dotenv()
    
    AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
    AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
    AWS_BUCKET = os.getenv('AWS_BUCKET')
    MODEL_FILE = os.getenv('MODEL_FILE')

    print("[*] Iniciando descarga del modelo ONNX desde S3...")
    print(f"    - Bucket: {AWS_BUCKET}")
    print(f"    - Archivo: {MODEL_FILE}")
    
    if not all([AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_BUCKET, MODEL_FILE]):
        print("ERROR: Variables de entorno no están configuradas correctamente")
        return False
    
    try:
        # Crear cliente S3
        s3 = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY
        )
        
        # Obtener nombre del archivo desde la ruta
        model_filename = os.path.basename(MODEL_FILE)
        
        # Ruta absoluta en la raíz del proyecto
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_path = os.path.join(PROJECT_ROOT, model_filename)
        
        print(f"[*] Descargando modelo: {MODEL_FILE}")
        print(f"[*] Bucket: {AWS_BUCKET}")
        print(f"[*] Guardando en: {local_path}")
        
        # Descargar el archivo
        s3.download_file(AWS_BUCKET, MODEL_FILE, local_path)
        
        print(f"[✓] Modelo descargado exitosamente: {model_filename}")
        return True
        
    except Exception as e:
        print(f"[✗] Error descargando el modelo: {e}")
        return False
