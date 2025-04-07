import pandas as pd
import numpy as np
import os
import re # For robust splitting

def parse_jasco_cd_file(filepath):
    """
    Parses a Jasco CD data file (.txt format).

    Args:
        filepath (str): The path to the Jasco CD data file.

    Returns:
        tuple: A tuple containing:
            - metadata (dict): A dictionary containing the header information.
            - data (pd.DataFrame): A pandas DataFrame with the spectral or melt data.
                                    Columns are named based on XUNITS/YUNITS/Y2UNITS.
    Raises:
        FileNotFoundError: If the filepath does not exist.
        ValueError: If the file format is unexpected or data conversion fails.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    metadata = {}
    data_lines = []
    reading_data = False

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.upper() == 'XYDATA':
                    reading_data = True
                    continue

                if not reading_data:
                    # Use tab as the primary separator for metadata
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        key, value = parts[0].strip(), parts[1].strip()
                        # Attempt to convert numerical metadata
                        try:
                            if key in ['FIRSTX', 'LASTX', 'MAXY', 'MINY', 'FIRSTY', 'DELTAX']:
                                value = float(value)
                            elif key in ['NPOINTS']:
                                value = int(value)
                        except ValueError:
                            pass # Keep as string if conversion fails
                        metadata[key] = value
                    # Handle lines without explicit tabs if needed (though format seems consistent)
                    # else:
                    #     print(f"Skipping metadata line without tab: {line}")

                else: # Reading data section
                    # Use regex to split by whitespace, handling potential multiple spaces/tabs
                    parts = re.split(r'\s+', line)
                    if len(parts) == 3:
                        try:
                            x = float(parts[0])
                            y = float(parts[1])
                            y2 = float(parts[2])
                            data_lines.append([x, y, y2])
                        except ValueError:
                            print(f"Warning: Could not parse data line: {line} in {filepath}")
                    # Handle lines with only 2 columns if Y2 is sometimes missing (unlikely based on examples)
                    elif len(parts) == 2:
                         try:
                            x = float(parts[0])
                            y = float(parts[1])
                            data_lines.append([x, y, np.nan]) # Add NaN for missing Y2
                         except ValueError:
                            print(f"Warning: Could not parse data line: {line} in {filepath}")
                    else:
                         print(f"Warning: Unexpected number of columns in data line: {line} in {filepath}")


    except Exception as e:
        raise ValueError(f"Error reading file {filepath}: {e}")

    if not data_lines:
        # Handle case where XYDATA was found but no data followed, or file was empty/metadata only
        print(f"Warning: No data points found in {filepath}")
        # Decide on return type: empty DataFrame or raise error? Let's return empty for now.
        df = pd.DataFrame(columns=['X', 'Y', 'Y2']) # Default column names
    else:
        df = pd.DataFrame(data_lines) # Create DataFrame first

    # --- Determine Column Names ---
    x_units = metadata.get('XUNITS', 'X')
    y_units = metadata.get('YUNITS', 'Y')
    y2_units = metadata.get('Y2UNITS', 'Y2')

    # Basic cleaning of units for column names
    def clean_col_name(name):
        name = name.replace('[', '').replace(']', '').replace(' ', '_')
        # Handle special characters if needed, e.g., degree symbol
        name = name.replace('°', 'deg') # Example
        name = name.replace('⁻¹', '_inv') # Example for inverse units
        return name

    col_x = 'Wavelength_nm' if 'NANO' in x_units.upper() else 'Temperature_C' if 'TEMP' in x_units.upper() else clean_col_name(x_units)
    col_y = 'CD_mdeg' if 'CD' in y_units.upper() else clean_col_name(y_units)
    col_y2 = 'HT_V' if 'HT' in y2_units.upper() else clean_col_name(y2_units)

    df.columns = [col_x, col_y, col_y2]

    # --- Data Validation/Consistency Check ---
    if 'NPOINTS' in metadata and metadata['NPOINTS'] != len(df):
        print(f"Warning: NPOINTS metadata ({metadata['NPOINTS']}) does not match number of data rows ({len(df)}) in {filepath}")
    # Add more checks if needed (e.g., FIRSTX/LASTX vs actual data range)

    return metadata, df