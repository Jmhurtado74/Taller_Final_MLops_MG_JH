import os
import boto3
from dotenv import load_dotenv

def download_test_data():
    # Cargar variables de entorno
    load_dotenv()
    
    AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
    AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
    AWS_BUCKET = os.getenv('AWS_BUCKET')
    
    if not all([AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_BUCKET]):
        print("ERROR: Variables de entorno no están configuradas correctamente")
        return False
    
    try:
        # Ruta de la carpeta de datos en la raíz del proyecto
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        TEST_DATA_DIR = os.path.join(PROJECT_ROOT, 'test_data')
        
        # Crear directorio local para datos si no existe
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        
        # Ruta base de los datos en S3
        s3_data_prefix = "mlops-ci-cd/model-iris/dataset"
        
        print("[*] Validando archivos de prueba...")
        print(f"[*] Carpeta destino: {TEST_DATA_DIR}")
        
        # Verificar qué archivos CSV ya existen localmente
        existing_files = set()
        if os.path.exists(TEST_DATA_DIR):
            existing_files = {f for f in os.listdir(TEST_DATA_DIR) if f.endswith('.csv')}
        
        if existing_files:
            print(f"[✓] Se encontraron {len(existing_files)} archivo(s) CSV existente(s):")
            for csv_file in existing_files:
                print(f"    - {TEST_DATA_DIR}/{csv_file}")
        
        # Crear cliente S3
        s3 = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY
        )
        
        print(f"[*] Buscando archivos CSV en s3://{AWS_BUCKET}/{s3_data_prefix}/...")
        
        response = s3.list_objects_v2(
            Bucket=AWS_BUCKET,
            Prefix=s3_data_prefix
        )
        
        if 'Contents' not in response:
            print(f"[✗] No se encontraron archivos en s3://{AWS_BUCKET}/{s3_data_prefix}/")
            return False
        
        # Descargar archivos CSV que no existan localmente
        csv_files = []
        files_downloaded = []
        
        for obj in response['Contents']:
            if obj['Key'].endswith('.csv'):
                file_name = os.path.basename(obj['Key'])
                local_path = os.path.join(TEST_DATA_DIR, file_name)
                
                # Verificar si el archivo ya existe
                if os.path.exists(local_path):
                    print(f"[✓] Archivo ya existe: {local_path}")
                    csv_files.append(local_path)
                else:
                    print(f"[*] Descargando: {obj['Key']}...")
                    s3.download_file(AWS_BUCKET, obj['Key'], local_path)
                    csv_files.append(local_path)
                    files_downloaded.append(file_name)
                    print(f"[✓] Guardado en: {local_path}")
        
        if csv_files:
            print(f"\n[✓] Datos disponibles ({len(csv_files)} archivo(s))")
            print(f"    Ubicación: s3://{AWS_BUCKET}/{s3_data_prefix}/")
            if files_downloaded:
                print(f"    {len(files_downloaded)} archivo(s) descargados:")
                for csv_file in files_downloaded:
                    print(f"      - {csv_file}")
            if existing_files:
                print(f"    {len(existing_files)} archivo(s) ya existían")
            return True
        else:
            print("[✗] No se encontraron archivos CSV")
            return False
        
    except Exception as e:
        print(f"[✗] Error: {e}")
        return False
