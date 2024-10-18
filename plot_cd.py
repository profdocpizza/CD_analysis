# load csv

# Read the file to see its structure and check how the tables are separated
import re
import pandas as pd


def parse_cd_csv(file_content):
    # Split the content to find the "Data:" section
    data_section = file_content.split("Data:\n\n")[1]

    # Split the data section into individual table sections (split by one empty line, accounting for blank lines)
    tables_raw = re.split(r"\n\n+", data_section.strip())

    # Initialize an empty dictionary to hold the tables
    tables = {}

    # Loop through the raw tables and parse them
    for section in tables_raw:
        # Adjusting the regex to capture all possible table titles, including CircularDichroism
        match = re.search(
            r"(CircularDichroism|HV|Absorbance|Voltage|Count|SE|Temperature)", section
        )
        if match:
            title = match.group(1)

            # Split the section into lines
            lines = section.strip().split("\n")

            # The second line contains the index and column headers (e.g., Wavelength,Temperature)
            index_and_columns = lines[1].split(",")

            # The third line contains the actual column headers (e.g., ,5,10,15,20,...)
            column_headers = lines[2].split(",")

            # The rest of the lines are data rows
            data = [line.split(",") for line in lines[3:] if line.strip()]

            # Ensure the number of columns matches, trimming or padding the headers if necessary
            max_columns = max(len(row) for row in data)
            adjusted_column_headers = column_headers[:max_columns]
            if len(adjusted_column_headers) < max_columns:
                adjusted_column_headers += [""] * (
                    max_columns - len(adjusted_column_headers)
                )

            # Create a DataFrame
            df = pd.DataFrame(
                data, columns=[index_and_columns[0]] + adjusted_column_headers[1:]
            )
            df.set_index(index_and_columns[0], inplace=True)

            # Convert the data to numeric where possible
            df = df.apply(pd.to_numeric, errors="coerce")

            # Add the DataFrame to the dictionary
            tables[title] = df

    return tables


# Parse the tables again using the updated approach
cd_dataframes = parse_cd_csv(
    "/home/tadas/code/CD_analysis/deltaprot1_50uM_5C_80Cmelt00002.csv"
)

# Display the found table titles and a sample of one table to verify
print(cd_dataframes["CircularDichroism"].Wavelength)
