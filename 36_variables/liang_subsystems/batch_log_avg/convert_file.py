import csv
import re

def extract_p_value(text):
    """
    Given a string like "(False, 0.707503791029669, 0.47925345763253335)",
    extract and return the third value (the p-value) as a string.
    """
    # Use regex to capture three comma-separated fields inside parentheses.
    m = re.search(r'\(\s*[^,]+,\s*[^,]+,\s*([^)]+)\s*\)', text)
    return m.group(1).strip() if m else ""

# Open the composite input file.
with open('combined_log_results_strong.csv', newline='') as infile:
    reader = csv.DictReader(infile)
    
    tab_rows = []
    tba_rows = []
    
    # Process each row of the input file.
    for row in reader:
        # Extract the time index "i" from the "File" column by removing trailing s or w.
        i = row['File'].rstrip('sw')
        
        # For the causal link from atmosphere to ocean (TAB)
        tab_row = {
            'i': i,
            'TE': row['TAB'],
            'err TE': row['Error TAB'],
            'p-value': extract_p_value(row['Significant TAB'])
        }
        tab_rows.append(tab_row)
        
        # For the causal link from ocean to atmosphere (TBA)
        tba_row = {
            'i': i,
            'TE': row['TBA'],
            'err TE': row['Error TBA'],
            'p-value': extract_p_value(row['Significant TBA'])
        }
        tba_rows.append(tba_row)

# Write the TAB output file.
with open('output_LKIF_strong_atmosphere_ocean.csv', 'w', newline='') as tabfile:
    fieldnames = ['i', 'TE', 'err TE', 'p-value']
    writer = csv.DictWriter(tabfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in tab_rows:
        writer.writerow(row)

# Write the TBA output file.
with open('output_LKIF_strong_ocean_atmosphere.csv', 'w', newline='') as tbafile:
    fieldnames = ['i', 'TE', 'err TE', 'p-value']
    writer = csv.DictWriter(tbafile, fieldnames=fieldnames)
    writer.writeheader()
    for row in tba_rows:
        writer.writerow(row)

print("Files have been successfully created: output_LKIF_atmosphere_ocean.csv and output_LKIF_ocean_atmosphere.csv")
