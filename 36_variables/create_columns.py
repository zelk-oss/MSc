import csv 

file_path = '../myqgs/data_1e5points_1000ws/evol_fields_1_1e-7.dat'
output_file1 = 'output_first_20_columns.txt'
output_file2 = 'output_last_16_columns.txt'
accelerate = 1  # Change this to set the acceleration factor

# Lists to store the split data
data_first_20 = []
data_last_16 = []

# Read and process the file
with open(file_path, 'r') as file:
    for index, line in enumerate(file):
        # Skip lines based on the acceleration factor
        if index % accelerate == 0:
            try:
                row = [float(value) for value in line.split()]
                if len(row) < 37:
                    print(f"Line has insufficient data: {line.strip()}")
                    continue
                # Remove the first column
                row = row[1:]
                # Split into two parts: first 20 and last 16 columns
                data_first_20.append(row[:20])
                data_last_16.append(row[20:36])
            except ValueError:
                print(f"Could not convert line to floats: {line}")

# Write the first 20 columns to the first output file
with open(output_file1, 'w') as file1:
    for row in data_first_20:
        file1.write(' '.join(map(str, row)) + '\n')

# Write the last 16 columns to the second output file
with open(output_file2, 'w') as file2:
    for row in data_last_16:
        file2.write(' '.join(map(str, row)) + '\n')

print(f"Data processing complete. Output files generated:\n{output_file1}\n{output_file2}")


with open('maooam_columns.txt') as f:
    reader = csv.reader(f, delimiter=' ', skipinitialspace=True)
    first_row = next(reader)
    num_cols = len(first_row)
    print("number of columns: ", num_cols)
    lines = len(f.readlines())
    print('Total Number of lines:', lines)