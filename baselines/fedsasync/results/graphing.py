import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 1. Configuración de parámetros ---
datasets = ['cifar10']        # Agrega aquí más: ['cifar10', 'mnist']
fs_list = [0.25]              # Agrega aquí más: [0.25, 0.5]
m_list = [8, 10]          # Los valores que quieres comparar en una misma gráfica

base_path = '/home/usuario/Escritorio/SAFlwr/baselines/fedsasync/results'

# --- 2. Bucles anidados ---
for ds in datasets:
    for fs in fs_list:
        
        # Creamos una figura nueva para cada combinación de Dataset y FS
        plt.figure(figsize=(10, 6))
        encontrado_al_menos_uno = False
        
        for m in m_list:
            # Construcción dinámica de la ruta del archivo
            # Ejemplo: .../cifar10/FedSaSync_fs0.25_m8.csv
            file_name = f'FedSaSync_fs{fs}_m{m}.csv'
            full_path = os.path.join(base_path, ds, file_name)
            
            if os.path.exists(full_path):
                try:
                    df = pd.read_csv(full_path)
                    
                    # Graficar la línea de este 'm'
                    plt.plot(df['time'], df['loss'], 
                             marker='.',       # Marcador más pequeño para que no se sature
                             linestyle='-', 
                             label=f'm = {m}') # La etiqueta indica el valor de m
                    
                    encontrado_al_menos_uno = True
                except Exception as e:
                    print(f"Error al leer {file_name}: {e}")
            else:
                print(f"Archivo no encontrado: {full_path}")

        # --- 3. Personalización y guardado de la gráfica comparativa ---
        if encontrado_al_menos_uno:
            plt.title(f'Comparativa de m | Dataset: {ds} | fs: {fs}')
            plt.xlabel('Tiempo (segundos)')
            plt.ylabel('Pérdida (Loss)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(title="Valor de m")
            
            # Guardamos el PDF con un nombre descriptivo
            nombre_pdf = f"Comparativa_{ds}_fs{fs}.pdf"
            plt.savefig(nombre_pdf, format='pdf', bbox_inches='tight')
            print(f"Generado: {nombre_pdf}")
        
        # Cerramos la figura para liberar memoria y empezar la siguiente limpia
        plt.close()

print("Proceso finalizado.")