# --- Plotting Dependencies ---
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

import os

from CD_analysis import parse_jasco_cd_file

# --- Constants (Define these based on your experiment) ---
DEFAULT_PROTEIN_CONCENTRATION_UM = 18
DEFAULT_PATH_LENGTH_MM = 1.0 # Match the filenames (1mm)
DEFAULT_NUM_RESIDUES = 33   # Replace with your protein's residue count

# --- Calculation Functions ---
def get_mean_residue_ellipticity(protein_concentration_uM=DEFAULT_PROTEIN_CONCENTRATION_UM,
                                 path_length_mm=DEFAULT_PATH_LENGTH_MM,
                                 num_residues=DEFAULT_NUM_RESIDUES):
    """Calculates the conversion factor from mdeg to MRE (deg·cm²·dmol⁻¹)."""
    if not all([protein_concentration_uM, path_length_mm, num_residues]):
         raise ValueError("Missing one or more parameters for MRE calculation: concentration, path length, num residues")
    path_length_cm = path_length_mm / 10.0  # Convert mm to cm
    concentration_M = protein_concentration_uM * 1e-6  # Convert µM to M
    # Formula for MRE = (mdeg * MW) / (10 * pathlength_cm * concentration_g_L)
    # Or MRE = mdeg / (10 * pathlength_cm * concentration_M * num_residues)
    # We return the factor to multiply mdeg by.
    denominator = (10.0 * path_length_cm * concentration_M * num_residues)
    if denominator == 0:
        raise ValueError("MRE calculation denominator is zero. Check concentration, path length, and num residues.")
    return 1.0 / denominator

def interpolate_and_smooth_data(x_values, y_values, new_x_grid=None, window_length=7, polyorder=2):
    """
    Interpolate and smooth Y data based on X data.

    Parameters:
    - x_values: NumPy array of original x values.
    - y_values: NumPy array of original y values.
    - new_x_grid: NumPy array of new x values for interpolation. If None, creates a default grid.
    - window_length: The length of the filter window for smoothing.
    - polyorder: The order of the polynomial used to fit the samples.

    Returns:
    - new_x_grid: The x grid used for interpolation.
    - smoothed_y_values: The interpolated and smoothed Y values.
    """
    # Ensure inputs are numpy arrays
    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)

    # Handle decreasing X values (like wavelength scans) for interpolation
    sort_indices = np.argsort(x_values)
    x_sorted = x_values[sort_indices]
    y_sorted = y_values[sort_indices]

    # If no new x grid is provided, create one based on sorted data
    if new_x_grid is None:
        new_x_grid = np.linspace(x_sorted.min(), x_sorted.max(), num=500)
    else:
        new_x_grid = np.asarray(new_x_grid)
        new_x_grid.sort() # Ensure the new grid is sorted

    # Interpolate the data onto the new grid
    # Use bounds_error=False and fill_value="extrapolate" cautiously if needed,
    # or ensure new_x_grid is within the bounds of x_sorted.
    try:
        interpolation_function = interp1d(x_sorted, y_sorted, kind='cubic', bounds_error=False, fill_value=np.nan)
        interpolated_y_values = interpolation_function(new_x_grid)
    except ValueError as e:
         print(f"Interpolation failed: {e}. Check data range and new_x_grid.")
         # Fallback to linear or handle differently?
         interpolation_function = interp1d(x_sorted, y_sorted, kind='linear', bounds_error=False, fill_value=np.nan)
         interpolated_y_values = interpolation_function(new_x_grid)


    # Apply Savitzky-Golay filter to the interpolated data
    # Handle potential NaNs from extrapolation before smoothing
    valid_indices = ~np.isnan(interpolated_y_values)
    if not np.any(valid_indices):
        print("Warning: All interpolated values are NaN. Smoothing cannot be applied.")
        return new_x_grid, interpolated_y_values # Return NaNs

    interp_y_valid = interpolated_y_values[valid_indices]
    smoothed_y_values_valid = interp_y_valid # Default if smoothing fails

    if len(interp_y_valid) > polyorder and len(interp_y_valid) > window_length:
        # Ensure window_length is odd and less than data length
        if window_length % 2 == 0:
            window_length += 1
        window_length = min(window_length, len(interp_y_valid) - (1 if len(interp_y_valid) % 2 == 0 else 0)) # Ensure less than length and odd
        if window_length > polyorder:
             try:
                smoothed_y_values_valid = savgol_filter(interp_y_valid, window_length, polyorder)
             except ValueError as e:
                 print(f"Warning: Savgol filter failed: {e}. Returning interpolated data.")
        else:
             print(f"Warning: window_length ({window_length}) must be greater than polyorder ({polyorder}). Skipping smoothing.")
    else:
        print(f"Warning: Not enough data points ({len(interp_y_valid)}) for smoothing with window {window_length} and polyorder {polyorder}. Skipping smoothing.")

    # Place smoothed values back into the original array structure with NaNs
    smoothed_y_values = np.full_like(interpolated_y_values, np.nan)
    smoothed_y_values[valid_indices] = smoothed_y_values_valid

    return new_x_grid, smoothed_y_values

# --- Plotting Functions (Adapted) ---

def plot_cd_spectra(cd_data_df, sample_name,
                    protein_concentration_uM=DEFAULT_PROTEIN_CONCENTRATION_UM,
                    path_length_mm=DEFAULT_PATH_LENGTH_MM,
                    num_residues=DEFAULT_NUM_RESIDUES,
                    overlay_df=None, overlay_label="Overlay",
                    title="Circular Dichroism Spectrum", save_dir=".",
                    smooth_window=7, smooth_polyorder=2,
                    x_limits=(190, 260), y_limits_mre=None): # y_limits in MRE units
    """
    Plots CD spectra (Wavelength vs MRE), optionally with an overlay.

    Args:
        cd_data_df (pd.DataFrame): DataFrame with 'Wavelength_nm' and 'CD_mdeg' columns.
        sample_name (str): Name of the primary sample.
        protein_concentration_uM (float): Protein concentration in µM.
        path_length_mm (float): Cuvette path length in mm.
        num_residues (int): Number of residues in the protein.
        overlay_df (pd.DataFrame, optional): DataFrame for overlay spectrum. Defaults to None.
        overlay_label (str, optional): Label for the overlay spectrum. Defaults to "Overlay".
        title (str, optional): Plot title base. Defaults to "Circular Dichroism Spectrum".
        save_dir (str, optional): Directory to save plots. Defaults to ".".
        smooth_window (int): Savgol filter window length.
        smooth_polyorder (int): Savgol filter polynomial order.
        x_limits (tuple): Wavelength range for plotting.
        y_limits_mre (tuple, optional): Y-axis limits in MRE units. Auto-scaled if None.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Check required columns
    required_cols = ['Wavelength_nm', 'CD_mdeg']
    if not all(col in cd_data_df.columns for col in required_cols):
         raise ValueError(f"Main DataFrame missing required columns: {required_cols}")
    if overlay_df is not None and not all(col in overlay_df.columns for col in required_cols):
         raise ValueError(f"Overlay DataFrame missing required columns: {required_cols}")

    # --- Process Primary Data ---
    mre_factor = get_mean_residue_ellipticity(protein_concentration_uM, path_length_mm, num_residues)

    # Interpolate and smooth the CD data
    wavelengths = cd_data_df['Wavelength_nm'].values
    cd_values = cd_data_df['CD_mdeg'].values
    new_wavelength_grid, smoothed_cd_values = interpolate_and_smooth_data(
        wavelengths, cd_values, window_length=smooth_window, polyorder=smooth_polyorder
    )

    # Calculate MRE for smoothed data
    smoothed_mre = smoothed_cd_values * mre_factor

    # Create a DataFrame for smoothed data
    smoothed_cd_df = pd.DataFrame({
        'Wavelength_nm': new_wavelength_grid,
        'Smoothed_MRE_deg_cm2_dmol': smoothed_mre
    })

    # --- Process Overlay Data (if provided) ---
    smoothed_overlay_df = None
    if overlay_df is not None:
        overlay_wavelengths = overlay_df['Wavelength_nm'].values
        overlay_cd_values = overlay_df['CD_mdeg'].values
        # Use the same new grid for consistency
        overlay_new_grid, overlay_smoothed_cd = interpolate_and_smooth_data(
            overlay_wavelengths, overlay_cd_values, new_x_grid=new_wavelength_grid,
            window_length=smooth_window, polyorder=smooth_polyorder
        )
        overlay_smoothed_mre = overlay_smoothed_cd * mre_factor
        smoothed_overlay_df = pd.DataFrame({
            'Wavelength_nm': overlay_new_grid, # Should be same as new_wavelength_grid
            'Smoothed_MRE_deg_cm2_dmol': overlay_smoothed_mre
        })

    # --- Plotting ---
    plt.figure(figsize=(5,4))
    sns.set_style("ticks") # or "whitegrid"

    # Plot the smoothed primary data
    primary_plot_data = smoothed_cd_df.dropna() # Drop NaNs that might result from interpolation/smoothing edges
    sns.lineplot(
        data=primary_plot_data,
        x='Wavelength_nm',
        y='Smoothed_MRE_deg_cm2_dmol',
        linewidth=1.5, # Slightly thicker line
        label=f'{sample_name}'
    )

    # Plot the smoothed overlay data
    if smoothed_overlay_df is not None:
        overlay_plot_data = smoothed_overlay_df.dropna()
        sns.lineplot(
            data=overlay_plot_data,
            x='Wavelength_nm',
            y='Smoothed_MRE_deg_cm2_dmol',
            color='black',
            linestyle='--', # Dashed line for overlay
            label=f'{overlay_label}',
            linewidth=1.5,
        )

    # Customize the plot
    full_title = f"{title}\n{sample_name}" + (f" vs {overlay_label}" if overlay_df is not None else "")
    plt.title(full_title, fontsize=14)
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel("Mean Residue Ellipticity (deg·cm²·dmol⁻¹)", fontsize=12)
    plt.xlim(x_limits)
    if y_limits_mre:
        plt.ylim(y_limits_mre)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(title='Sample', loc='best', fontsize=10)
    # Add horizontal line at 0
    plt.axhline(0, color='grey', linestyle='-', linewidth=0.5)

    # Save the plot
    filename_base = f"cd_spectrum_{sample_name.replace(' ', '_')}"
    if overlay_df is not None:
        filename_base += "_vs_" + overlay_label.replace(' ', '_')
    for ext in ["png", "svg"]:
        plt.savefig(os.path.join(save_dir, f"{filename_base}.{ext}"), dpi=300, bbox_inches="tight")

    plt.tight_layout()
    # plt.show()


def plot_melting_curve(melt_data_df, sample_name, wavelength_nm=222, # Common wavelength for melts
                      protein_concentration_uM=DEFAULT_PROTEIN_CONCENTRATION_UM,
                      path_length_mm=DEFAULT_PATH_LENGTH_MM,
                      num_residues=DEFAULT_NUM_RESIDUES,
                      reverse_melt_df=None, reverse_label="Cooling", # For hysteresis
                      title="Thermal Melting Curve", save_dir=".",
                      y_limits_mre=None,x_limits=None): # y_limits in MRE units
    """
    Plots a thermal melting curve (Temperature vs MRE).

    Args:
        melt_data_df (pd.DataFrame): DataFrame with 'Temperature_C' and 'CD_mdeg' columns.
        sample_name (str): Name of the sample.
        wavelength_nm (int/float): Wavelength at which the melt was monitored (for title/info).
        protein_concentration_uM (float): Protein concentration in µM.
        path_length_mm (float): Cuvette path length in mm.
        num_residues (int): Number of residues in the protein.
        reverse_melt_df (pd.DataFrame, optional): DataFrame for the reverse (cooling) melt.
        reverse_label (str): Label for the reverse melt curve.
        title (str, optional): Plot title base. Defaults to "Thermal Melting Curve".
        save_dir (str, optional): Directory to save plots. Defaults to ".".
        y_limits_mre (tuple, optional): Y-axis limits in MRE units. Auto-scaled if None.
    """
    os.makedirs(save_dir, exist_ok=True)

     # Check required columns
    required_cols = ['Temperature_C', 'CD_mdeg']
    if not all(col in melt_data_df.columns for col in required_cols):
         raise ValueError(f"Main melt DataFrame missing required columns: {required_cols}")
    if reverse_melt_df is not None and not all(col in reverse_melt_df.columns for col in required_cols):
         raise ValueError(f"Reverse melt DataFrame missing required columns: {required_cols}")

    mre_factor = get_mean_residue_ellipticity(protein_concentration_uM, path_length_mm, num_residues)

    # Prepare primary melt data
    melt_data = melt_data_df.copy() # Avoid modifying original df
    melt_data['MRE_deg_cm2_dmol'] = melt_data['CD_mdeg'] * mre_factor
    # Sort by temperature for plotting
    melt_data = melt_data.sort_values(by='Temperature_C').dropna(subset=['MRE_deg_cm2_dmol'])

    # Prepare reverse melt data (if provided)
    reverse_data = None
    if reverse_melt_df is not None:
        reverse_data = reverse_melt_df.copy()
        reverse_data['MRE_deg_cm2_dmol'] = reverse_data['CD_mdeg'] * mre_factor
        reverse_data = reverse_data.sort_values(by='Temperature_C').dropna(subset=['MRE_deg_cm2_dmol'])


    # Set up the plot
    plt.figure(figsize=(5,4)) # Slightly different aspect ratio might be good
    sns.set_style("ticks")

    # Plot the primary melting curve (Heating)
    sns.lineplot(data=melt_data, x='Temperature_C', y='MRE_deg_cm2_dmol',
                 marker='o', linewidth=1.5, markersize=5, label=f'{sample_name} (Heating)')

    # Plot the reverse melting curve (Cooling)
    if reverse_data is not None:
         sns.lineplot(data=reverse_data, x='Temperature_C', y='MRE_deg_cm2_dmol',
                      marker='s', linewidth=1.5, markersize=5, linestyle='--', color='grey',
                      label=f'{sample_name} ({reverse_label})')


    # Customize the plot
    full_title = f"{title} at {wavelength_nm} nm\n{sample_name}"
    plt.title(full_title, fontsize=14)
    plt.xlabel('Temperature (°C)', fontsize=12)
    plt.ylabel("Mean Residue Ellipticity (deg·cm²·dmol⁻¹)", fontsize=12)
    if y_limits_mre:
        plt.ylim(y_limits_mre)
    else:
        # Auto-adjust y-limits slightly if not specified
        all_mre_values = melt_data['MRE_deg_cm2_dmol']
        if reverse_data is not None:
            all_mre_values = pd.concat([all_mre_values, reverse_data['MRE_deg_cm2_dmol']])
        if not all_mre_values.empty:
            min_y, max_y = all_mre_values.min(), all_mre_values.max()
            padding = (max_y - min_y) * 0.05 # 5% padding
            plt.ylim(min_y - padding, max_y + padding)

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()

    # Save the plot
    filename_base = f"melting_curve_{sample_name.replace(' ', '_')}_{wavelength_nm}nm"
    if reverse_data is not None:
        filename_base += "_hysteresis"
    for ext in ["png", "svg"]:
        plt.savefig(os.path.join(save_dir, f"{filename_base}.{ext}"), dpi=300, bbox_inches="tight")

    # plt.show()




# --- Main Execution ---
if __name__ == "__main__":

    # Define the directory containing the data files
    # IMPORTANT: Replace with the actual path to your files
    data_dir = "/home/tadas/code/CD_analysis/inputs/bristol_cd" 
    output_dir = "/home/tadas/code/CD_analysis/outputs/bristol_cd"
    # os.makedirs(output_dir, exist_ok=True)

    # --- Define Experimental Parameters ---
    # These might vary per sample, adjust as needed
    params_df = pd.read_csv("/home/tadas/code/CD_analysis/inputs/bristol_cd/deltaprot_designs_data_with_results.csv")[["Well Position","cd_path_length_mm",
 "cd_sample_molar_conc","sequence_length"]]
    # set Well Position as index
    params_df.set_index("Well Position", inplace=True)

    # show cd_path_length_mm of "B2" well position 
    params = {
        "y_limits_mre":(-40000, 0),
        "x_limits":(200, 260),
    }

    # --- Define Files ---
    files = {
        "Blank": "20250404_1mm_BP-Buffer_blank.txt",
        "A9": "20250404_1mm_BP-Buffer_A9.txt",
        "A12": "20250404_1mm_BP-Buffer_A12.txt",
        "B8": "20250404_1mm_BP-Buffer_B8.txt",
        "B8_melt": "20250404_1mm_BP-Buffer_B8_melt.txt",
        "B8_remelt": "20250404_1mm_BP-Buffer_B8_melt@.txt", # Cooling curve
        "B8_post_melt": "20250404_1mm_BP-Buffer_B8_post_melt.txt"
    }

    # --- Parse Files ---
    parsed_data = {}
    for name, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        try:
            print(f"Parsing {name}: {filename}...")
            metadata, df = parse_jasco_cd_file(filepath)
            parsed_data[name] = {'meta': metadata, 'data': df}
            print(f"  -> Found {len(df)} data points. X-axis: {df.columns[0]}")
            # print(df.head(2)) # Optional: print head for verification
        except FileNotFoundError:
            print(f"  -> File not found. Skipping.")
        except ValueError as e:
            print(f"  -> Error parsing file: {e}. Skipping.")
        print("-" * 20)

    # --- Basic Data Subtraction (Example: Subtract Blank from Samples) ---
    if "Blank" in parsed_data and 'data' in parsed_data["Blank"]:
        blank_df = parsed_data["Blank"]['data']
        # Ensure blank has same wavelengths/temperatures for simple subtraction
        # More robust subtraction might require interpolation

        for name in ["A9", "A12", "B8", "B8_post_melt"]:
             if name in parsed_data and 'data' in parsed_data[name]:
                sample_df = parsed_data[name]['data']
                # Check if x-axes match (simple check)
                if sample_df.columns[0] == blank_df.columns[0] and \
                   len(sample_df) == len(blank_df) and \
                   np.allclose(sample_df.iloc[:, 0], blank_df.iloc[:, 0]):

                    y_col_name = sample_df.columns[1] # e.g., 'CD_mdeg'
                    print(f"Subtracting Blank from {name}...")
                    # Create a new DataFrame for subtracted data
                    subtracted_df = sample_df.copy()
                    subtracted_df[y_col_name] = sample_df[y_col_name] - blank_df[y_col_name]
                    # Store it back, maybe with a new key or overwrite
                    parsed_data[f"{name}_Subtracted"] = {'meta': parsed_data[name]['meta'], 'data': subtracted_df}
                    print("   -> Subtraction complete.")
                else:
                    print(f"Warning: Cannot directly subtract Blank from {name}. X-axes differ or lengths mismatch. Skipping subtraction.")
                    # Add interpolation logic here if needed for robust subtraction
                    parsed_data[f"{name}_Subtracted"] = parsed_data[name] # Keep original if subtraction fails


    # --- Generate Plots ---

    # Plot individual (subtracted) spectra


    if "A9_Subtracted" in parsed_data:
        plot_cd_spectra(parsed_data["A9_Subtracted"]['data'], sample_name="Sample A9",
                protein_concentration_uM=params_df.loc["A9", "cd_sample_molar_conc"]*10**6, num_residues=params_df.loc["A9", "sequence_length"], path_length_mm=params_df.loc["A9", "cd_path_length_mm"], save_dir=output_dir,**params)

    if "A12_Subtracted" in parsed_data:
         plot_cd_spectra(parsed_data["A12_Subtracted"]['data'], sample_name="Sample A12",
                        protein_concentration_uM=params_df.loc["A12", "cd_sample_molar_conc"]*10**6, num_residues=params_df.loc["A12", "sequence_length"], path_length_mm=params_df.loc["A12", "cd_path_length_mm"], save_dir=output_dir,**params)

    # Plot B8 Pre- vs Post-Melt (Subtracted)
    if "B8_Subtracted" in parsed_data and "B8_post_melt_Subtracted" in parsed_data:
         plot_cd_spectra(parsed_data["B8_Subtracted"]['data'], sample_name="Sample B8 Pre-Melt",
                        overlay_df=parsed_data["B8_post_melt_Subtracted"]['data'], overlay_label="B8 Post-Melt",
                        protein_concentration_uM=params_df.loc["B8", "cd_sample_molar_conc"]*10**6, num_residues=params_df.loc["B8", "sequence_length"], path_length_mm=params_df.loc["B8", "cd_path_length_mm"], save_dir=output_dir,**params,
                        title="CD Spectra Comparison")

    # Plot B8 Melting Curve (Heating vs Cooling) - Using original (non-subtracted) data
    # Melting curve baseline drift is often less critical than spectral baseline
     # Example limits for melt

    if "B8_melt" in parsed_data and "B8_remelt" in parsed_data:
        # You might want specific MRE limits for melts
        
        plot_melting_curve(parsed_data["B8_melt"]['data'], sample_name="Sample B8",
                           reverse_melt_df=parsed_data["B8_remelt"]['data'], reverse_label="Cooling",
                           wavelength_nm=222, # Assume 222 nm unless specified elsewhere
                           protein_concentration_uM=params_df.loc["B8", "cd_sample_molar_conc"]*10**6, path_length_mm=params_df.loc["B8", "cd_path_length_mm"], num_residues=params_df.loc["B8", "sequence_length"], save_dir=output_dir,**params)
    elif "B8_melt" in parsed_data:
         # Plot just the heating curve if cooling curve is missing
         plot_melting_curve(parsed_data["B8_melt"]['data'], sample_name="Sample B8",
                            wavelength_nm=222,
                           protein_concentration_uM=params_df.loc["B8", "cd_sample_molar_conc"]*10**6, path_length_mm=params_df.loc["B8", "cd_path_length_mm"], num_residues=params_df.loc["B8", "sequence_length"], save_dir=output_dir,**params)

    print(f"\nProcessing complete. Plots saved to '{output_dir}'")