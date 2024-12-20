import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_tools import math_tools as mt
import os


def concat_excel_files(f_path, sheet_name):
    # Get a list of all .xlsx files in the folder
    excel_files = [f for f in os.listdir(f_path) if f.endswith('.xlsx')]
    all_data = []

    # Read and concatenate data from the specified sheet
    for file in excel_files:
        file_path = os.path.join(f_path, file)
        try:
            # Read the specific sheet from the Excel file
            df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
            df = df.dropna(subset=['DateTime']).reset_index(drop=True)
            df = df.drop(columns='Unnamed: 0', errors='ignore')  # Avoid errors if the column doesn't exist

            for col in [7] + list(range(11, len(df.columns))):
                df.iloc[:, col] = np.array(df.iloc[:, col], dtype=np.float32)
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {sheet_name} from {file}: {e}")

    # Concatenate all DataFrames
    if all_data:
        all_data = pd.concat(all_data, ignore_index=True)
        # Convert to numeric, handling errors if any column types are incorrect
        all_data.iloc[:, 1:] = all_data.iloc[:, 1:]
        return all_data
    else:
        print(f"No valid data found for sheet {sheet_name}.")
        return None


def create_lever_ratio(array_strat, array_algo, lookback, max_lever):
    """
    For each element in array_a, divide it by the minimum of all elements
    before it in array_b.

    Parameters:
    - array_a (np.ndarray): Array to perform division.
    - array_b (np.ndarray): Array to compute minimums of previous elements.

    Returns:
    - result (np.ndarray): Resulting array after division.
    """
    result = np.ones_like(array_strat, dtype=float)

    for i in range(len(array_strat)):
        if i == 0:
            result[i] = 1
        else:
            lb = lookback if i > lookback else 0

            min_strat = np.min(array_strat[lb:i])
            min_algo = np.min(array_algo[lb:i])

            if array_algo[i] == 0 or min_strat == 0 or min_algo == 0:
                result[i] = 1
            else:
                result[i] = min(max(min_algo / min_strat, 1), max_lever)

    return result


def plot_rolling_sum(df, sheet_name, param_id, f_path):
    if df is not None and not df.empty:
        # Calculate rolling sum
        rolling_df = df.copy()
        rolling_df['DateTime'] = pd.to_datetime(rolling_df['DateTime'], errors='coerce', unit='ns')
        rolling_df = rolling_df.sort_values(by='DateTime').reset_index(drop=True)
        rolling_df['PnL_algo_tot'] = rolling_df['PnL'].cumsum()
        rolling_df['PnL_one_dir_tot'] = rolling_df['PnL_one_dir'].cumsum()
        rolling_df['PnL_two_dir_tot'] = rolling_df['PnL_two_dir'].cumsum()

        rolling_df['Maxdraw_algo'] = mt.calculate_max_drawdown(rolling_df['PnL_algo_tot'])
        rolling_df['Maxdraw_one_dir'] = mt.calculate_max_drawdown(rolling_df['PnL_one_dir_tot'])
        rolling_df['Maxdraw_two_dir'] = mt.calculate_max_drawdown(rolling_df['PnL_two_dir_tot'])

        adj_one_dir_pnl = (rolling_df['PnL_one_dir_tot'].values *
                           create_lever_ratio(rolling_df['Maxdraw_one_dir'].values,
                                              rolling_df['Maxdraw_algo'].values,
                                              lookback=10,
                                              max_lever=5))
        adj_two_dir_pnl = (rolling_df['PnL_two_dir_tot'].values *
                           create_lever_ratio(rolling_df['Maxdraw_two_dir'].values,
                                              rolling_df['Maxdraw_algo'].values,
                                              lookback=50,
                                              max_lever=5))

        # Create subplots
        fig, axes = plt.subplots(3, 1, sharex=False, figsize=(12, 20))

        # PnL plot
        axes[0].plot(rolling_df['DateTime'], rolling_df["PnL_algo_tot"], label="Algo PnL Tot", color='darkred')
        axes[0].plot(rolling_df['DateTime'], rolling_df["PnL_one_dir_tot"], label="One Dir Tot", color='darkblue')
        axes[0].plot(rolling_df['DateTime'], rolling_df["PnL_two_dir_tot"], label="Two Dir PnL Tot", color='green')
        axes[0].set_title('PnL')
        axes[0].legend()
        axes[0].grid(True)

        # Max Drawdown plot
        axes[1].plot(rolling_df['DateTime'], rolling_df["Maxdraw_algo"], label="Algo", color='darkred')
        axes[1].plot(rolling_df['DateTime'], rolling_df["Maxdraw_one_dir"], label="One Dir", color='darkblue')
        axes[1].plot(rolling_df['DateTime'], rolling_df["Maxdraw_two_dir"], label="Two Dir", color='green')
        axes[1].set_title('Max Drawdown')
        axes[1].legend()
        axes[1].grid(True)

        # Adjusted PnL plot
        axes[2].plot(rolling_df['DateTime'], rolling_df["PnL_algo_tot"], label="Algo", color='darkred')
        axes[2].plot(rolling_df['DateTime'], adj_one_dir_pnl, label="One Dir", color='darkblue')
        axes[2].plot(rolling_df['DateTime'], adj_two_dir_pnl, label="Two Dir", color='green')
        axes[2].set_title('PnL - Adjusted for Algo Maxdraw')
        axes[2].legend()
        axes[2].grid(True)

        # Apply x-axis formatting to the entire figure
        for ax in axes:
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            ax.tick_params(axis='x', rotation=45)

        # Save Excel file
        excel_path = os.path.join(f_path, f"{sheet_name}_{param_id}_total.xlsx")
        rolling_df.to_excel(excel_path, index=False)
        print(f"Saved Excel file to: {excel_path}")

        # Save the plot as an image
        img_loc = os.path.join(f_path, f"{param_id}_{sheet_name}.png")
        plt.savefig(img_loc)
        print(f"Saved plot to: {img_loc}")

        plt.close(fig)

    else:
        print(f"No data available to plot for {sheet_name}.")


side = 'Bull'
folder_main = r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\Double_Candles\ATR\NQ\15min\15min_test_20years\classification_lstm'
folder_path = os.path.join(folder_main, side, "Data")

# Filter only directories
result_files = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
for param in result_files:
    print(f'param: {param}')
    rf = os.path.join(folder_path, param)

    agg_pnl_data = concat_excel_files(rf, f"{side}_PnL")

    # Plot the graphs
    plot_rolling_sum(agg_pnl_data, f"{side}_PnL", param, folder_path)

