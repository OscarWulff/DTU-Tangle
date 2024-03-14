import pandas as pd
import numpy as np

# Create a 3x3 matrix with random 1s and 0s
data = np.random.randint(2, size=(30, 20))

# Create a DataFrame from the matrix
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv("/Users/MortenHelsoe/Desktop/DTU/6. Semester/Bachelor Projekt/Tangle-lib-ORM/DTU-Tangle/csv_test/test.csv", index=False)

print("CSV file 'example.csv' created successfully:")
print(df)