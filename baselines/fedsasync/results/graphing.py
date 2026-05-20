import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 1. Global Configuration ---
# Define the parameters for the grid search/comparison
DATASETS = ['cifar10']
FS_VALUES = [0, 1, 2]
M_VALUES = [0, 6, 7, 8, 9, 10]
EXECUTIONS = 10

# Root directory where the results are stored
BASE_PATH = '/home/usuario/Escritorio/SAFlwr/baselines/fedsasync/results'

def generate_comparison_plots():
    """
    Iterates through datasets and fs values to create comparison plots 
    of 'm' values (Time vs Loss).
    """
    print("Starting plot generation...")

    results = []

    for ds in DATASETS:
        for fs in FS_VALUES:
            
            # Initialize a new figure for each Dataset/FS combination
            plt.figure(figsize=(10, 6))
            file_found_in_group = False
            
            for m in M_VALUES:
                # Construct file path dynamically
                # Pattern: {BASE_PATH}/{dataset}/FedSaSync_fs{fs}_m{m}.csv
                if m == 0:
                    file_name = f'FedAvg_fs{fs}.csv'
                    label = 'FedAvg'
                else:
                    file_name = f'FedSaSync_fs{fs}_m{m}.csv'
                    label = f'm = {m}'

                dfs = []

                for exe in range(EXECUTIONS):
                    full_path = os.path.join(BASE_PATH, ds, str(exe+1), file_name)
                
                    if os.path.exists(full_path):
                        try:
                            # Load data
                            df = pd.read_csv(full_path)
                        
                            # Validate if required columns exist
                            if {'time', 'loss', 'train_time'}.issubset(df.columns):
                                dfs.append(df[['time', 'loss', 'train_time']].copy())
                            else:
                                print(f"Warning: Missing columns in {file_name}")
                        except Exception as e:
                            print(f"Error processing {file_name}: {e}")
                    else:
                        print(f"File not found: {full_path}")
                    
                if dfs:
                    merged = None

                    for i, d in enumerate(dfs):
                        d = d.sort_values('time').rename(columns={
                            'loss': f'loss_{i}',
                            'train_time': f'train_time_{i}'
                        })
                        if merged is None:
                            merged = d
                        else:
                            merged = pd.merge_asof(
                                merged.sort_values('time'),
                                d.sort_values('time'),
                                on='time',
                                direction='nearest',
                                tolerance=0.5
                            )

                    merged = merged.sort_values('time')

                    # Calculate loss_mean per configuration
                    loss_cols = [c for c in merged.columns if c.startswith('loss_')]
                    merged['loss_mean'] = merged[loss_cols].mean(axis=1, skipna=True)

                    # Calculate train_time_mean per configuration
                    train_time_cols = [c for c in merged.columns if c.startswith('train_time_')]
                    merged['train_time_mean'] = merged[train_time_cols].mean(axis=1, skipna=True)

                    # Calculate execution time percentage
                    train_time_total = merged['train_time_mean'].sum()
                    time_total = merged['time'].iloc[-1]
                    train_pct = train_time_total / time_total * 100

                    results.append({
                        'dataset': ds,
                        'fs': fs,
                        'm': m,
                        'train_pct': train_pct
                    })

                    # Suavizado
                    merged['loss_smooth'] = merged['loss_mean'].ewm(span=5, adjust=False).mean()

                    plt.plot(
                        merged['time'],
                        merged['loss_smooth'],
                        linestyle='-',
                        linewidth=2,
                        alpha=0.9,
                        label=label
                    )
                    file_found_in_group = True                    

            # --- Finalize and Save the Plot ---
            if file_found_in_group:
                plt.xlabel('Time')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{ds}_fs{fs}.pdf", format="pdf")
            else:
                plt.close()

    results_df = pd.DataFrame(results)

    print("Table plot.")

    # Obtener datasets únicos
    datasets = results_df["dataset"].unique()
    for dataset in datasets:
        df_ds = results_df[results_df["dataset"] == dataset]

        table_data = df_ds.pivot(
            index="fs",
            columns="m",
            values="train_pct"
        )
        
        # Separar columnas
        cols = list(table_data.columns)
        # Extraer FedAvg (m=0) si existe
        cols_no_fedavg = [c for c in cols if c != 0]
        # Reordenar: resto + FedAvg al final
        new_order = cols_no_fedavg + [0]
        table_data = table_data[new_order]

        col_labels = [
            "FedAvg" if m == 0 else f"m = {m}"
            for m in table_data.columns
        ]

        row_labels = [
            f"Slow = {slow}"
            for slow in table_data.index
        ]

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis("off")

        cell_text = table_data.round(2).astype(str) + "%"

        table = ax.table(
            cellText=cell_text.values,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellLoc='center',
            loc='center'
        )

        # Añadir esquina superior izquierda
        w = table[(1, -1)].get_width()      # ancho columna etiquetas fila
        h = table[(0, 0)].get_height()      # alto cabecera

        corner = table.add_cell(
            0, -1,
            width=w,
            height=h,
            text="Strategy →\nSlow clients ↓",
            loc='center'
        )

        corner.set_fontsize(9)

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)

        plt.tight_layout()
        plt.savefig(f"{dataset}_train_pct.pdf", format="pdf")
        plt.close()

    print("Process finished.")

if __name__ == "__main__":
    generate_comparison_plots()