import os
import zipfile
import shutil

# Ruta base donde están los .zip y la carpeta images
base_dir = r"C:\Users\Bryant\OneDrive\Documentos\GitHub\Detector-Senales-Trafico\data\raw"

# Carpeta donde están las subcarpetas train/test/val
images_dir = os.path.join(base_dir, "images")

# Definir las carpetas de destino
destination_folders = {
    'test': os.path.join(images_dir, 'test'),
    'train': os.path.join(images_dir, 'train'),
    'val': os.path.join(images_dir, 'val')
}

# Crear las carpetas destino si no existen
for folder_name, folder_path in destination_folders.items():
    if not os.path.exists(folder_path):
        print(f"Creando carpeta: {folder_path}")
        os.makedirs(folder_path, exist_ok=True)

# Listar todos los archivos .zip en base_dir
zip_files = [f for f in os.listdir(base_dir) if f.endswith('.zip')]

if not zip_files:
    print("No se encontraron archivos .zip en la ruta especificada.")
else:
    print(f"Archivos .zip encontrados: {zip_files}")

# Procesar cada archivo zip
for zip_filename in zip_files:
    zip_path = os.path.join(base_dir, zip_filename)

    # Decidir a qué carpeta de images irá
    if 'test' in zip_filename:
        dest_folder = destination_folders['test']
    elif 'train' in zip_filename:
        dest_folder = destination_folders['train']
    elif 'val' in zip_filename:
        dest_folder = destination_folders['val']
    else:
        print(f"No se reconoce la categoría del archivo {zip_filename}, se omite.")
        continue

    print(f"Procesando {zip_filename}...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extraer solo los archivos que están dentro de la carpeta images/
        for member in zip_ref.infolist():
            if member.filename.startswith('images/') and not member.is_dir():
                # Obtener el nombre del archivo sin la carpeta images/
                filename = os.path.basename(member.filename)
                if filename:  # Ignorar entradas vacías
                    # Ruta de salida
                    extracted_path = os.path.join(dest_folder, filename)
                    # Extraer y guardar el archivo
                    with zip_ref.open(member) as source, open(extracted_path, "wb") as target:
                        shutil.copyfileobj(source, target)

    print(f"Descompresión de {zip_filename} completada.")

print("¡Todos los archivos fueron descomprimidos correctamente!")