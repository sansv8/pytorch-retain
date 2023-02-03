"""
Author: Amith Panuganti
Description: Creates a histogram from the dataset to understand more about the data itself
             Is not in heartFailure.py due a risk of the data coming in wrong. 
Data: 1/20/22
"""

# Improt pyplot and pickle
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Firstly, unpickle the original datasets 
with open('Data/train.seqs', 'rb') as f:
    train_seqs = pickle.load(f)
with open('Data/train.labels', 'rb') as f:
    train_labels = pickle.load(f)
with open('Data/valid.seqs', 'rb') as f:
    valid_seqs = pickle.load(f)
with open('Data/valid.labels', 'rb') as f:
    valid_labels = pickle.load(f)
with open('Data/test.seqs', 'rb') as f:
    test_seqs = pickle.load(f)
with open('Data/test.labels', 'rb') as f:
    test_labels = pickle.load(f)

# Firslty, we find the perecentage of data that are heart failure and non heartfailure 
# Combined all the labels into one list
labels = train_labels + valid_labels + test_labels

# Get the length of all labels, which is the number of pateints in the dataset
total_number_patients = len(labels)

# Create two counter, one for patients that have heart failure and the other without it
nonHF = 0
HF = 0

# Go thought each patient in labels
for patient in labels:
    # If patient is 0
    if patient == 0:
        # Add patient to nonHF
        nonHF = nonHF + 1
    # Otherwise, if not
    else:
        # Add patient to HF
        HF = HF + 1

# Next, combine all the seqs into one list
seqs = train_seqs + valid_seqs + test_seqs

# Go through each seqs and get its len
len_of_seqs = [len(seq) for seq in seqs]

# Make copy of len of seqs
len_of_seqs = len_of_seqs

# Print the number of patients
print("Number of patients: ", total_number_patients)

# Number of patients without heart failure
print("Number of patients without heart failure: ", nonHF)

# The number of patients with heart failure
print("Number of patients with heart failure: ", HF)

# The percentage of patients without heart failure
print("Percentage of patients without heart failure: ", (nonHF/total_number_patients * 100), "%")

# The percentage of patients with heart failure
print("Percentage of patients with heart failure: ", ((HF/total_number_patients) * 100), "%")

# The input with the most number of visits
print("Highest number of visits: ", max(len_of_seqs))

# Mean number of total visits
print("Mean number of total visits: ", sum(len_of_seqs)/total_number_patients)

# Create text file to store previous printed information
with open("hfDataInfo.txt", 'w') as f:
    f.write("Number of patients: {}\n".format(total_number_patients))
    f.write("Number of patients without heart failure: {}\n".format(nonHF))
    f.write("Number of patients with heart failure: {}\n".format(HF))
    f.write("Percentage of patients without heart failure: {}%\n".format((nonHF/total_number_patients * 100)))
    f.write("Percentage of patients with heart failure: {}%\n".format(((HF/total_number_patients) * 100)))
    f.write("Highest number of visits: {}\n".format(max(len_of_seqs)))
    f.write("Mean number of total visits: {}\n".format(sum(len_of_seqs)/total_number_patients))


# Go through each length in len_of_seqs
for i in range(0, len(len_of_seqs)):
    # If seq is greater than 5
    if len_of_seqs[i] > 5:
        # Set len_of_seqs[i] to be 6
        len_of_seqs[i] = 6

# Plot as histogram
counts, bins = np.histogram(len_of_seqs)
plt.hist(len_of_seqs, [1,2,3,4,5,6,7])

# Add title
plt.title("Frequencey of Visits of Patients")

# Add labels for the x and y axis
plt.xlabel("Number of visits")
plt.ylabel("Frequency of vists")

# Relabel xticks
plt.xticks([1,2,3,4,5,6,7], [1,2,3,4,5,">5", ""])

# Add numbers above each bar 
labels = [1,2,3,4,5,">5"]

# Add value labels
# i represent the xPos of the text
i = 1
for count in counts:
    # If count is 0
    if count != 0:
        # Plot text on theg raph
        plt.text(i+0.5, count+10, count, ha="center")

        # Increase i by 1
        i = i + 1
    
# Save figure
plt.savefig("Frequency of Visits")


