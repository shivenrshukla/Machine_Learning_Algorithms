import pandas as pd

# Parameters for the dataset
principal = 1000  # Fixed principal amount
rate = 0.05       # Fixed rate (5%)
time_periods = range(1, 21)  # Time from 1 to 20 years

# Generate the dataset
data = {
    "Time (Years)": time_periods,
    "Principal ($)": [principal] * len(time_periods),
    "Rate (%)": [rate * 100] * len(time_periods),  # Convert rate to percentage
    "Interest ($)": [principal * rate * t for t in time_periods]
}

# Create DataFrame
interest_dataset = pd.DataFrame(data)

# Save to a CSV file
file_path = "/Users/shivenshukla/Desktop/Data Science/Kaggle/input/interest_dataset.csv"
interest_dataset.to_csv(file_path, index=False)

file_path