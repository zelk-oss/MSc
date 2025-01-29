import csv

# Prompt user for file names
csv_file = input("diagnostics_timeseries.csv").strip()
if not csv_file:
    csv_file = 'diagnostics_timeseries.csv'

txt_file = input("diagnostics_timeseries.txt").strip()
if not txt_file:
    txt_file = 'diagnostics_timeseries.txt'

try:
    # Open and process files
    with open(csv_file, "r") as my_input_file:
        with open(txt_file, "w") as my_output_file:
            csv_reader = csv.reader(my_input_file)
            for row in csv_reader:
                my_output_file.write(" ".join(row) + '\n')

    print(f"Successfully written the data from '{csv_file}' to '{txt_file}'.")
except FileNotFoundError:
    print(f"Error: File '{csv_file}' not found. Please check the file name and try again.")
except Exception as e:
    print(f"An error occurred: {e}")
