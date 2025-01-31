import pandas as pd
import os

# Define the function to process the file
def correct_signs(filename):
    # Read the CSV file
    df = pd.read_csv(filename)
    
    # Ensure the required columns exist
    if "InfoFlow" not in df.columns or "Tau" not in df.columns:
        print("Error: The file does not contain the required columns 'InfoFlow' and 'Tau'.")
        return
    
    # Correct the signs
    df.loc[df["InfoFlow"] < 0, "Tau"] *= -1
    
    # Construct the new filename
    base, ext = os.path.splitext(filename)
    new_filename = f"{base}_corrected{ext}"
    
    # Save the corrected file
    df.to_csv(new_filename, index=False)
    print(f"Corrected file saved as {new_filename}")

# Example usage (replace 'data.csv' with your actual filename)
if __name__ == "__main__":
    input_filename = "liang_res_11days_100yr_weak_avg.csv"  # Change this to your actual file name
    correct_signs(input_filename)
