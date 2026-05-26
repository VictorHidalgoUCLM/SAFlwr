import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 1. Global Configuration ---
# Define the parameters for the grid search/comparison
DATASETS = ['cifar10', 'mnist']
FS_VALUES = [0, 1, 2]
M_VALUES = [0, 7, 8, 9, 10]

# Root directory where the results are stored
BASE_PATH = '/home/usuario/Escritorio/SAFlwr/baselines/fedsasync/_static'

def generate_comparison_plots():
    """
    Iterates through datasets and fs values to create comparison plots 
    of 'm' values (Time vs Loss).
    """
    print("Starting plot generation...")

    results = []

    for ds in DATASETS:
        fig, axes = plt.subplots(
            1,
            len(FS_VALUES),
            figsize=(8 * len(FS_VALUES), 5.5),
            sharey=True
        )

        # Fix case with a single subplot
        if len(FS_VALUES) == 1:
            axes = [axes]

        for ax, fs in zip(axes, FS_VALUES):
            for m in M_VALUES:

                # Build file name
                if m == 0:
                    file_name = f'FedAvg_fs{fs}.csv'
                    label = 'FedAvg'
                else:
                    file_name = f'FedSaSync_fs{fs}_m{m}.csv'
                    label = f'FedSaSync (m = {m})'

                full_path = os.path.join(
                    BASE_PATH,
                    ds,
                    file_name
                )

                # File existence check
                if not os.path.exists(full_path):
                    print(f"File not found: {full_path}")
                    continue

                try:
                    df = pd.read_csv(full_path)

                except Exception as e:
                    print(f"Error loading {file_name}: {e}")
                    continue

                required_cols = {
                    'time',
                    'loss',
                    'train_time'
                }

                if not required_cols.issubset(df.columns):
                    print(
                        f"Missing columns in {file_name}"
                    )
                    continue

                df = df[
                    ['time', 'loss', 'train_time']
                ].copy()

                df = df.sort_values('time')

                # Training percentage
                train_time_total = df['train_time'].sum()
                total_time = df['time'].iloc[-1]
                train_pct = (
                    train_time_total /
                    total_time * 100
                )

                results.append({
                    'dataset': ds,
                    'fs': fs,
                    'm': m,
                    'train_pct': train_pct
                })

                # Plot on current subplot
                ax.plot(
                    df['time'],
                    df['loss'],
                    linewidth=1.5,
                    alpha=0.9,
                    label=label
                )

            # Configure subplot
            ax.set_title(
                f"Slow clients = {fs}",
                fontsize=16
            )

            ax.set_xlabel("Time")
            ax.grid(alpha=0.3)

        # Shared Y label only once
        axes[0].set_ylabel("Loss")

        handles, labels = (
            axes[0].get_legend_handles_labels()
        )

        order = list(range(len(labels)))
        fedavg_idx = labels.index('FedAvg')
        order = [i for i in order if i != fedavg_idx] + [fedavg_idx]

        fig.legend(
            [handles[i] for i in order],
            [labels[i] for i in order],
            loc='lower center',
            bbox_to_anchor=(0.5, -0.08),
            ncol=len(M_VALUES),
            frameon=False,
            fontsize=16
        )

        plt.tight_layout(
            rect=[0, 0, 1, 1]
        )
        plt.savefig(
            os.path.join(
                BASE_PATH,
                f"{ds}_comparison.png"
            ),
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()

    results_df = pd.DataFrame(results)
    
    print("Table plot.")

    # Get all datasets
    datasets = results_df["dataset"].unique()
    for dataset in datasets:
        df_ds = results_df[results_df["dataset"] == dataset]

        table_data = df_ds.pivot(
            index="fs",
            columns="m",
            values="train_pct"
        )
        
        # Column separation
        cols = list(table_data.columns)
        # Extract FedAvg (m=0) if exists
        cols_no_fedavg = [c for c in cols if c != 0]
        # Reorder: all + FedAvg at the end
        new_order = cols_no_fedavg + [0]
        table_data = table_data[new_order]

        col_labels = [
            "FedAvg" if m == 0 else f"FedSaSync (m = {m})"
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

        # Add top left corner
        w = table[(1, -1)].get_width()      # width
        h = table[(0, 0)].get_height()      # height

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

        fig.suptitle(f"{dataset} - Train percentage", fontsize=14, y=0.8)

        plt.tight_layout(pad=0)
        plt.savefig(
            f"{BASE_PATH}/{dataset}_train_pct.png",
            dpi=300,
            bbox_inches='tight',
            pad_inches=0
        )
        plt.close()

    print("Process finished.")

if __name__ == "__main__":
    generate_comparison_plots()