import pandas as pd

file_path = 'your_file.xlsx'  # Replace with your actual file path
sheet_name = 'Sheet1'  # Specify the sheet name if necessary

# Read the Excel file
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Replace the text in the specified column
column_name = 'Grade'  # Replace with the actual column name where the text appears
df[column_name] = df[column_name].replace('anaplastic; Grade IV', 4)

# Save the modified DataFrame back to an Excel file
output_file_path = 'modified_file.xlsx'
df.to_excel(output_file_path, index=False)

print(f"File saved to {output_file_path}")
