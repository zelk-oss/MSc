import csv

# Prompt user for input and output file names
dat_file = 'evol_fields_1_1e-7.dat'

txt_file = 'evol_fields_1_1e-7.txt'

try:
    # Open and process files
    with open(dat_file, "r") as my_input_file:
        with open(txt_file, "w") as my_output_file:
            # Use a CSV reader with space delimiter to read the DAT file
            csv_reader = csv.reader(my_input_file, delimiter=' ')
            for row in csv_reader:
                if len(row) > 1:  # Ensure there's more than one column
                    # Ignore the first column by slicing the list
                    my_output_file.write(" ".join(row[1:]) + '\n')

    print(f"Successfully written the data from '{dat_file}' to '{txt_file}'.")
except FileNotFoundError:
    print(f"Error: File '{dat_file}' not found. Please check the file name and try again.")
except Exception as e:
    print(f"An error occurred: {e}")
