import pandas as pd
import numpy as np

# Create a 3x3 matrix with random 1s and 0s
data = np.random.randint(2, size=(50, 10))

# Create a DataFrame from the matrix
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv("/Users/MortenHelsoe/Desktop/DTU/6. Semester/Bachelor Projekt/Tangle-lib-ORM/DTU-Tangle/csv_test/test.csv", index=False)

print("CSV file 'example.csv' created successfully:")
print(df)
# import numpy as np

# # Define the dimensions of the dataset
# num_people = 100
# num_questions = 50

# # Create an empty numpy array to store the data
# data = np.zeros((num_people, num_questions), dtype=int)

# # Define the bias for the two groups
# bias_group1 = 0.8  # Group 1 tends to answer '1' for the first half and '0' for the second half
# bias_group2 = 0.2  # Group 2 tends to answer '0' for the first half and '1' for the second half

# # Generate data for each person
# for person in range(num_people):
#     # Assign responses for the first half of questions
#     for question in range(num_questions // 2):
#         # Determine response based on bias
#         if np.random.random() < bias_group1:
#             data[person, question] = 1
#         else:
#             data[person, question] = 0
            
#     # Assign responses for the second half of questions
#     for question in range(num_questions // 2, num_questions):
#         # Determine response based on bias
#         if np.random.random() < bias_group2:
#             data[person, question] = 1
#         else:
#             data[person, question] = 0

# # Save the data to a CSV file with row and column indices
# np.savetxt("biased_dataset.csv", data, delimiter=",", fmt='%d', header=",".join(map(str, range(1, num_questions + 1))), comments='')

# print("Dataset generated and saved as 'biased_dataset.csv'.")
