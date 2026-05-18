import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 1. Global Configuration ---
# Define the parameters for the grid search/comparison
DATASETS = ['cifar10']
FS_VALUES = [0.3]
M_VALUES = [7]

# Root directory where the results are stored
BASE_PATH = '/home/usuario/Escritorio/SAFlwr/baselines/fedsasync/results'
OUTPUT_DIR = '/home/usuario/Escritorio/SAFlwr/baselines/fedsasync/results/plots' # Directory to save the generated PDFs

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def generate_comparison_plots():
    """
    Iterates through datasets and fs values to create comparison plots 
    of 'm' values (Time vs Loss).
    """
    print("Starting plot generation...")

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

                full_path = os.path.join(BASE_PATH, ds, file_name)
                
                if os.path.exists(full_path):
                    try:
                        # Load data
                        df = pd.read_csv(full_path)
                        
                        # Validate if required columns exist
                        if 'time' in df.columns and 'loss' in df.columns:
                            df = df.sort_values('time')
                            df['loss_smooth'] = df['loss'].ewm(span=9, adjust=False).mean()
                            plt.plot(
                                df['time'], 
                                df['loss'], 
                                marker='.', 
                                linestyle='-', 
                                alpha=0.8,
                                label=label
                            )
                            file_found_in_group = True
                        else:
                            print(f"Warning: Missing columns in {file_name}")
                            
                    except Exception as e:
                        print(f"Error processing {file_name}: {e}")
                else:
                    print(f"File not found: {full_path}")

            # --- Finalize and Save the Plot ---
            if file_found_in_group:
                # Add metadata and styling
                plt.title(f'Federated Learning Analysis: {ds} (fs={fs})')
                plt.xlabel('Time (seconds)')
                plt.ylabel('Validation Loss')
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.legend(title="Client Count (m)")
                
                # Export to PDF
                output_filename = f"{ds}_fs{fs}.pdf"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                
                plt.savefig(output_path, format='pdf', bbox_inches='tight')
                print(f"Successfully generated: {output_path}")
            
            # Close the figure to free up memory
            plt.close()

    print("Process finished.")

if __name__ == "__main__":
    generate_comparison_plots()