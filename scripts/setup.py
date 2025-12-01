import sys
from dotenv import load_dotenv

# Importar funciones de descarga
import download_model
import download_data

def setup_environment():
    """Valida y descarga modelo y datos de prueba si es necesario."""
    
    load_dotenv()
    
    print("=" * 60)
    print("SETUP: Validación y Descarga de Recursos")
    print("=" * 60)
    
    # Step 1: Descargar modelo
    print("\n[PASO 1] Descargando modelo ONNX...")
    print("-" * 60)
    try:
        download_model.download_model()
        print("[✓] Modelo ONNX disponible")
    except Exception as e:
        print(f"[✗] Error al descargar modelo: {e}")
        return False
    
    # Step 2: Descargar datos de prueba
    print("\n[PASO 2] Validando datos de prueba...")
    print("-" * 60)
    try:
        success = download_data.download_test_data()
        if not success:
            print("[✗] Error al obtener datos de prueba")
            return False
        print("[✓] Datos de prueba disponibles")
    except Exception as e:
        print(f"[✗] Error al validar datos: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ SETUP COMPLETADO EXITOSAMENTE")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = setup_environment()
    sys.exit(0 if success else 1)
