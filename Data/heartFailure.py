# Imports
import pickle
from sklearn.model_selection import train_test_split
from datetime import datetime
import numpy as np
import statistics as s
from tabulate import tabulate

# Convert codes from DIAGNOSES_ICD.csv to ICD-9 codes
# input: dxStr - The diagnoses string
# output: Its icd9 version
def convert_to_icd9(dxStr):
    # If code starts with E
	if dxStr.startswith('E'):
        # If length of code is greater than 4, add .
		if len(dxStr) > 4: return dxStr[:4] + '.' + dxStr[4:]
        # Otherwise, return string itself
		else: return dxStr
    # Otherwise, if codes does not start with E
	else:
        # If elngth of codes is greater than 3, add . to dxStr
		if len(dxStr) > 3: return dxStr[:3] + '.' + dxStr[3:]
        # Otherwise, return string
		else: return dxStr

# Create two maps that will map patient id to admissions id 
# and admissions id to date
# input: admissionFile
# output: pidAdmMap - Map of each pateint's id to their admissions ids
# output: admDataMap - Map of each admission id to its admissions datae
def map_PID_AMID_DATE(admissionsFile):
    # Dict to map each Patient's ID to their Admissions's IDs
    pidAdmMap = {}

    # Dict to map Each Admission ID to Admissions Date 
    admDateMap = {}

    # Open admissions file and read each line
    infd = open(admissionsFile, 'r')
    infd.readline()

    # For each line in file 
    for line in infd:
        # Split line into tokens
        tokens = line.strip().split(',')

        # Get Patient id
        pid = int(tokens[1])

        # Get Admission ID
        admId = int(tokens[2])

        # Get Admission Time
        admTime = datetime.strptime(tokens[3], '%Y-%m-%d %H:%M:%S')

        # Map admID to admTime
        admDateMap[admId] = admTime

        # If pid is in pidaDMap, add admId to current list. Otherwise, create new list and add admId
        if pid in pidAdmMap: pidAdmMap[pid].append(admId)
        else: pidAdmMap[pid] = [admId]
    # Close file
    infd.close()

    # Return both maps
    return pidAdmMap, admDateMap

# Map each admission id to whether a diagnoses contains a heart disease or not
# input: diagnosesFile - File containg daignoses for each admissions
# output: mapAdmHF - Map of each admission id to whether the patient was diagnose with heart failure
def map_AdmID_HF(diagnosesFile):
    # First, create our dictionary that will map Admission ID to Heart Failure
    mapAdmHF = {}

    # A patient is diagnosed with HF if the diagnoses matches the follwoing codes
    qualified_codes = ["398.91", "402.01", "402.11", "402.91", "404.01",
                        "404.03", "404.11", "404.13", "404.91", "404.93",
                        "428.0", "428.1", "428.20", "428.21", "428.22", "428.23",
                        "428.30", "428.31", "428.32", "428.33", "428.40",
                        "428.41", "428.42", "428.43", "428.9"]

    # Next, open diagnoses file and read each line
    infd = open(diagnosesFile, 'r')
    infd.readline()

    # For each file in the diagnoses file
    for line in infd:
        # Split line into tokens
        tokens = line.strip().split(',')

        # Get the admission id
        admID = int(tokens[2])

        # Next, convert the code into ICD_9 code
        code = convert_to_icd9(tokens[4][1:-1])

        # Next, represent whether the patient has Heart Failure
        # 0 for no HF, 1 for HF
        dxHF = 0 

        # Next, check if code is in qualified_codes. If true, set dxHF to 1
        # to indicate that the patient has heart disease for that visit
        if code in qualified_codes:
            dxHF = 1
            
        # Next, map admID to dxHF
        # If mapAdmHF does not have admID mapped or the mapped value is 0
        # map admID to dxHF
        if admID not in mapAdmHF:
            mapAdmHF[admID] = dxHF
        elif mapAdmHF[admID] == 0:
            mapAdmHF[admID] = dxHF

    # Close file
    infd.close()

    # Return mapAdmHF
    return mapAdmHF

# Create two maps, patient to list of HF Status for each visit and patient to list to dates
# Input: pidAdmMap - Map for each Patient ID to its Admission Ids
# Input: admDateMap - Map each Admission ID to its Data
# Input: mapAdmHF - Map each Admission ID to whether the diagnoses was heart failure or not
# Ouput: pidHFsMAP - Map each Patient ID to a list of whether the patient has HF or not for each admission
# Output: pidAdmDateMap - Map each Patient ID to a list of Admission Dates
def map_PID_HF_Date(pidAdmMap, admDateMap, mapAdmHf):
    # Firstly, create two maps
    # This maps the PID to a list of visits of whether the patients has heart failure or not
    pidHFsMap = {}

    # This maps the PID to a list of their admissions dates
    # This is used to make sure the sequence of visits are in the correct order
    pidAdmDateMap = {}

    # Go through each pid in adm map
    for pid in pidAdmMap:
        # Go through each adm id in the list
        for admID in pidAdmMap[pid]:
            # Check if pid is in pidHFsMap
            # If it is not
            if pid not in pidHFsMap:
                # Start new sequence of visit for paitent
                pidHFsMap[pid] = [mapAdmHf[admID]]
                pidAdmDateMap[pid] = [admDateMap[admID]]
            else:
                # Add new element to current sequence of visit for the patient
                pidHFsMap[pid].append(mapAdmHf[admID])
                pidAdmDateMap[pid].append(admDateMap[admID])

        # Sort the sequence of visits for each patient 
        pidHFsMap[pid] = [i for _, i in sorted(zip(pidAdmDateMap[pid], pidHFsMap[pid]))]
        pidAdmDateMap[pid].sort()
        
    # Return both maps
    return pidHFsMap, pidAdmDateMap

# This function will choose all pateints with heart failure from the dataset
# Input: Map of each patient ID to a list of whether the patient has HF or not for each admission
# Output: Map of each patient that has Heart Failure to the index of the last admission id where they are diagnose with HF
def find_HF_Patients(pidHFsMap):
    # Create a new map that will map each patient to the index of when the patient first recieve HF
    hfPatients = {}
    # Go thorugh each patient in pidHFsMap
    for patient in pidHFsMap:
        # Find the index of when the patient last receive Heart Failure
        # We are doing this so that it will increase the number of patients with
        # heart failure in the dataset
        try:
            index = len(pidHFsMap[patient]) - 1 - pidHFsMap[patient][::-1].index(1)

            # Next, check if the index is greater than 0
            # If index is greater than 0 add patient to hfPatients
            if index > 0:
                hfPatients[patient] = index
        # If a index is not found catch the exception
        except:
            continue
    
    # Finally return hfPatients
    return hfPatients

# Find all the patients that never have HF from the dataset
# Input: Map of each patient ID to a list of whether the patient has HF or not for each admission
# Output: List of patients that do not have heart failure
def find_non_HF_Patients(pidHFsMap):
    # Create a list that will store non HF patients
    nonHFPatients = []

    # Go thorugh each patient in pidHFsMap
    for patient in pidHFsMap:
        # Check if 1 is in pidHFsMap
        # If 1 in not in pidHFsMap[patient] and len of 
        # Add pateint to list
        if 1 not in pidHFsMap[patient]:
            nonHFPatients.append(patient)
    
    # Return nonHFPatients
    return nonHFPatients

# Map each admission id to diagnoses
# Input: diagnosesFile - File containig diagnoses code fore each patient
# Input: diagnosesGroupsFile - File contains diagnoses code groups for each diagnoses code
# Output: admDxMap - Contains a map of each admission id to a list of diagnoses codes
def map_diagnoses(diagnosesFile, diagnosesGroupsFile):
    # Unpickle the diagnosesGroups
    # Which contains the the code groups for each diagnoses code
    with open(diagnosesGroupsFile, 'rb') as f:
        diagnosesGroups = pickle.load(f)
    
    # Map each admission id to diagnoses
    admDxMap = {}

    # Read through each line of map_diagnoses
    infd = open(diagnosesFile, 'r')
    infd.readline()
    for line in infd:
        # Split up the lines into tokens
        tokens = line.split(",")

        # Get admID and code for the line
        admID = int(tokens[2])

        # Get the code and memove any suffix and prefixes
        code = tokens[4]
        code = code.removesuffix("\n")
        code = code.removesuffix("\"")
        code = code.removeprefix("\"")

        # Next, look into admDxMap and find the group the code is associated with
        if code in diagnosesGroups:
            # If code is in diagnosesGroups, then set code to be its code groups 
            # from diagnosesGroups
            code = diagnosesGroups[code]
        else:
            # Otherwise, the patient was not diagnosed
            code = "DNo"
        
        # Add code to map
        # If Admission ID is in admDxMap
        if admID in admDxMap:
            # If code is not in admDxMap[admID] and code is not DNo
            # Append it to the list
            # Otherwise, do not
            if code not in admDxMap[admID] and code != "DNo":
                admDxMap[admID].append(code)
        # Otherwise
        else:
            # If the code exists, add code to list 
            if code != "DNo":
                admDxMap[admID] = [code]
            #Otherwise, map admId to empty list
            else:
                admDxMap[admID] = []
    
    # Close the file
    infd.close()

    # Return admDxMap
    return admDxMap

# Map each admission to diagnose code without groupings 
# Input: diagnosesFile - File containig diagnoses code fore each patient
# Output: admDxMap - Contains a map of each admission id to a list of diagnoses codes
def map_diagnoses_without_group(diagnosesFile):
    # Map each admission id to diagnoses
    admDxMap = {}

    # Read through each line of map_diagnoses
    infd = open(diagnosesFile, 'r')
    infd.readline()
    for line in infd:
        # Split up the lines into tokens
        tokens = line.split(",")

        # Get admID and code for the line
        admID = int(tokens[2])

        # Get the code and memove any suffix and prefixes
        code = tokens[4]
        code = code.removesuffix("\n")
        code = code.removesuffix("\"")
        code = code.removeprefix("\"")

        # Add M to code
        code = "D" + code

        # If codes is empty
        if code == "D":
            # Set code to be DNo
            code = "DNo"

        # Add code to map
        # If Admission ID is in admDxMap
        if admID in admDxMap:
            # If code is not in admDxMap[admID] and code is not DNo
            # Append it to the list
            # Otherwise, do not
            if code not in admDxMap[admID] and code != "DNo":
                admDxMap[admID].append(code)
        # Otherwise
        else:
            # If the code exists, add code to list 
            if code != "DNo":
                admDxMap[admID] = [code]
            #Otherwise, map admId to empty list
            else:
                admDxMap[admID] = []

    # Close the file
    infd.close()

    # Return admDxMap
    return admDxMap

# Map each admission to procedure code
# Input: procedureFile - File containing procedure codes for each admission id
# Input: procedureGroupsFile - File containing procedure code groups for each procedure code
# Input: Dataset - Contains list of medical codes for each admission id
# Output: Dataset - Updated version of input dataset with the addtion of procedure codes for each admission id
def map_procedures(procedureFile, procedureGroupsFile, dataset):
    # Unpicke the procedureGroupsFile
    # This file contians the grouping for the procedures
    with open(procedureGroupsFile, 'rb') as f:
        procedureGroups = pickle.load(f)

    # Next, read the procedureFile
    # Read through each line of map_diagnoses
    infd = open(procedureFile, 'r')
    infd.readline()
    for line in infd:
        # Split up the lines into tokens
        tokens = line.split(",")

        # Get the admission ID
        admID = int(tokens[2])

        # Get CPT_CD
        code = tokens[5]
        code = code.removeprefix("\"")
        code = code.removesuffix("\"")
       
        # If code is in procedureGroups
        if code in procedureGroups:
            # Set code to be its group from procedureGroups
            code = procedureGroups[code]
        else:
            # If not, add P to code
            code = "P" + code

        # If admID is in procedureMap
        if admID in dataset:
            # Codes is not in dataset[admID], append it
            if code not in dataset[admID]:
                dataset[admID].append(code)
        else:
            #Else, map amdID to code
            dataset[admID] = [code]

    # Close file
    infd.close()

    # Return dataset
    return dataset

# Map each amdID to its procedure code without grouping
# Input: Dataset - Contains list of medical codes for each admission id
# Input: procedureFile - File containing procedure codes for each admission id
# Output: Dataset - Updated version of input dataset with the addtion of procedure codes for each admission id
def map_procedures_without_grouping(procedureFile, dataset):
    # Next, read the procedureFile
    # Read through each line of map_diagnoses
    infd = open(procedureFile, 'r')
    infd.readline()
    for line in infd:
        # Split up the lines into tokens
        tokens = line.split(",")

        # Get the admission ID
        admID = int(tokens[2])

        # Get CPT_CD
        code = tokens[5]
        code = code.removeprefix("\"")
        code = code.removesuffix("\"")
       
        # Add P to code
        code = "P" + code

        # If code is just P
        if(code == "P"):
            #CONTINUE
            continue
        
         # If admID is in procedureMap
        if admID in dataset:
            # Codes is not in dataset[admID], append it
            if code not in dataset[admID]:
                dataset[admID].append(code)
        else:
            #Else, map amdID to code
            dataset[admID] = [code]

    # Close file
    infd.close()

    # Return dataset
    return dataset

# Map each admID to its medication codes
# Input: medicationFile - File containing medication codes for each admission id
# Input: dataset - Contains a map of each admission id to a list of medical codes
# Ouput: dataset - Updated verision of input dataset with the addition of medication codes for each admission id
def map_medication(medicationFile, dataset):
    # Read through medicationfiLE
    infd = open(medicationFile, 'r')
    infd.readline()
    for line in infd:
        # Split line into tokens
        tokens = line.split(",")

        # Next, get the medication code and admID
        admID = int(tokens[2])

        # Get medication code
        code = tokens[11]
        code = code.removeprefix("\"")
        code = code.removesuffix("\"")

        # Add M to code to desinguish this code from other types of code
        code = "M" + code

        # If the codes does not exist continue
        if code == "M":
            continue

        # If admID is not in dataset, add it to dataset
        if admID not in dataset:
            dataset[admID] = [code]
        else:
            # If not append code to admID
            if code not in dataset[admID]:
                dataset[admID].append(code)

    # Close file
    infd.close()

    # Return dataset
    return dataset

# Create the sequences of visits using our dataset, pidAdmMap, and pidAdmDate
# Input: dataset - Contains list of medical codes for each patient
# Input: admDateMap - Map of each admission id to its date
# Input: pidAdmMap - Map of each patient id to a list of admission ids
# Input: hfPatinets - Map of each patient with heart failure to the index of their last visit that has heart failure
# Ouput: sequences - Map each Patient ID to a list of visits with each visits containing a list of medical codes
# Ouput: hfPatients - Updated verision of input hfPatients containing update version of the last visit whether the patient had HF
def create_sequences(dataset, admDateMap, pidAdmMap, hfPatients):
    # Firstly, we go create our list of sequences for each pid
    sequences = {}

    # Next, go through each pid and admIdList in pidAdmMap
    for pid in pidAdmMap:
        # Get the list of admission ids from the map
        admIdList = pidAdmMap[pid]

        # Sort the list using admDateMap to sort the pidAdmMap
        # Convert admID for dataset to string
        sortedList = sorted([(admDateMap[admId], dataset[admId]) for admId in admIdList])
        sortedList = [i[1] for i in sortedList]

        # Next, remove all of the empty lists from sorted list
        if [] in sortedList:
            # Create a count of all [] in list
            count = sortedList.count([])
            sortedList = [value for value in sortedList if len(value) != 0]

            # If pid is in hfPatients
            if pid in hfPatients:
                # Subtract index by count
                hfPatients[pid] = hfPatients[pid] - count
                
                # If hfPatients[pid] is less than 1, skip
                if hfPatients[pid] < 1:
                    continue
            
            # If sortedList is empty skip
            if len(sortedList) == 0:
                # Skip list
                continue

        # Next, map sortedList to pid
        sequences[pid] = sortedList
    
    # Finally, return sequences
    return sequences, hfPatients

# Get the final list of sequencs by choosing certain patients based on if they meet the criteria or not
# Input: sequences - Map of each patient id to a list of visits containig a list of medical codes for each visit
# Input: hfPatient - Map of each patient with heart failure to the index of the last visit where they did had heart failure
# Input: nonHFPatients - List of patients without heart failure
# Output: hf_sequences - Map of each patient with heart failure to list of visits contains codes for each visit
# Output: nonHF_sequences - Map of each patient without heart failure to list of visits contains codes for each visit 
def create_final_sequences(sequences, hfPatients, nonHFPatients):
    # Create two maps, one for Heart Failure Sequences and one with no Heart Failure Sequences
    hf_sequences = {}
    nonHF_sequences = {}

    # Firstly, go thorugh each patient in nonHFPatients
    for pid in nonHFPatients:
        # If pid is in sequences
        if pid in sequences:
            # If the length of sequecnes[pid] is greater than 1
            if len(sequences[pid]) > 1:
                    # Map pid to its sequences except its last visit
                    nonHF_sequences[pid] = sequences[pid][:-1]

    # Next, go through each pid in hfPatients
    for pid in hfPatients:
        # If pid is in sequecnes
        if pid in sequences:
            # Map each pid to its sequence up to a certain index
            index = hfPatients[pid]
            hf_sequences[pid] = sequences[pid][:index]
    
    # Return final sequences
    return hf_sequences, nonHF_sequences

# Analyze the dataset for each visits
# Input: sequences, the sequence of visits
# Input: name to store information about codes
def analyze_sequences(sequences, name):
    # Firstly, create four lists
    # One list contains the total number of codes for each visit
    # One list contains the total number of diagnoses codes for each visit
    # One list contains the total number of procedure codes for each visit
    # One list contains the total number of medication codes for each visit
    total_counts = []
    diagnoses_counts = []
    procedure_counts = []
    medication_counts = []

    # Go through each patient in sequences
    for patient in sequences:
        # Go through each visit in sequecnes
        for visit in sequences[patient]:
            # Get size of visit and add it to toal_counts
            total_counts.append(len(visit))

            # Next, keep track of three counts
            # The count of diagnoses codes
            # The count of procedure codes
            # The count of medication codes
            diagnosesCount = 0
            procedureCount = 0
            medicationCount = 0

            # Next, go through each code in sequences
            for code in visit:
                # Get first letter of code
                letter = code[0]

                # If letter is D
                if letter == 'D':
                    # Increase diagnosesCount by 1
                    diagnosesCount = diagnosesCount + 1
                # Otherwise if letter is P
                elif letter == 'P':
                    # Increase procedureCount
                    procedureCount = procedureCount + 1
                # Otherwise, if letter is M
                else:
                    # Incrase medicationCount
                    medicationCount = medicationCount + 1
            
            # Push each count of code to their respective lists
            diagnoses_counts.append(diagnosesCount)
            procedure_counts.append(procedureCount)
            medication_counts.append(medicationCount)
    
    # Create dictionary
    table = [['Code', 'Mean', 'Standard Deviation', 'Minimum', '1st Quartile', 'Median', '3rd Quartile', 'Maximum']]
    # Gather several metrics for each list
    # This includes: mean, min, max, 1st quartile, 3rd quartile, median, std, and mode
    # Total_counts
    total_mean = np.mean(total_counts)
    total_min = np.min(total_counts)
    total_max = np.max(total_counts)
    total_median = np.median(total_counts)
    total_stdev = np.std(total_counts)
    total_mode = s.mode(total_counts)
    total_1st_quartile = np.quantile(total_counts, 0.25)
    total_3rd_quartile = np.quantile(total_counts, 0.75)

    # Add them to table
    table.append(['All', total_mean, total_stdev, total_min, total_1st_quartile, total_median, total_3rd_quartile, total_max])

    # Diagnoses_counts
    diagnoses_mean = np.mean(diagnoses_counts)
    diagnoses_min = np.min(diagnoses_counts)
    diagnoses_max = np.max(diagnoses_counts)
    diagnoses_median = np.median(diagnoses_counts)
    diagnoses_stdev = np.std(diagnoses_counts)
    diagnoses_mode = s.mode(diagnoses_counts)
    diagnoses_1st_quartile = np.quantile(diagnoses_counts, 0.25)
    diagnoses_3rd_quartile = np.quantile(diagnoses_counts, 0.75)

    # Add them to table
    table.append(['Diagnoses', diagnoses_mean, diagnoses_stdev, diagnoses_min, diagnoses_1st_quartile, diagnoses_median, diagnoses_3rd_quartile, diagnoses_max])

    # Procedure_counts
    procedure_mean = np.mean(procedure_counts)
    procedure_min = np.min(procedure_counts)
    procedure_max = np.max(procedure_counts)
    procedure_median = np.median(procedure_counts)
    procedure_stdev = np.std(procedure_counts)
    procedure_mode = s.mode(procedure_counts)
    procedure_1st_quartile = np.quantile(procedure_counts, 0.25)
    procedure_3rd_quartile = np.quantile(procedure_counts, 0.75)

    # Add them to table
    table.append(['Procedure', procedure_mean, procedure_stdev, procedure_min, procedure_1st_quartile, procedure_median, procedure_3rd_quartile, procedure_max])

    # Medication_counts
    medication_mean = np.mean(medication_counts)
    medication_min = np.min(medication_counts)
    medication_max = np.max(medication_counts)
    medication_median = np.median(medication_counts)
    medication_stdev = np.std(medication_counts)
    medication_mode = s.mode(medication_counts)
    medication_1st_quartile = np.quantile(medication_counts, 0.25)
    medication_3rd_quartile = np.quantile(medication_counts, 0.75)

    # Add them to table
    table.append(['Medication', medication_mean, medication_stdev, medication_min, medication_1st_quartile, medication_median, medication_3rd_quartile, medication_max])

    # Save all statistices in a file based on name
    with open(name, 'w') as f:
        # Write the table
        f.write("Code Statistics\n")
        f.write("{}\n".format(tabulate(table, headers='firstrow', tablefmt='fancy_grid')))

        # Total Codes
        f.write("\nAll Codes\n")
        f.write("Mean: {}\n".format(total_mean))
        f.write("Standard Deviation: {}\n".format(total_stdev))
        f.write("Mode: {}\n".format(total_mode))
        f.write("Minimum: {}\n".format(total_min))
        f.write("1st Quartile: {}\n".format(total_1st_quartile))
        f.write("Median: {}\n".format(total_median))
        f.write("3rd Quartile: {}\n".format(total_3rd_quartile))
        f.write("Max: {}\n".format(total_max))

        # Diagnoses Codes
        f.write("\nDiagnoses Codes\n")
        f.write("Mean: {}\n".format(diagnoses_mean))
        f.write("Standard Deviation: {}\n".format(diagnoses_stdev))
        f.write("Mode: {}\n".format(diagnoses_mode))
        f.write("Minimum: {}\n".format(diagnoses_min))
        f.write("1st Quartile: {}\n".format(diagnoses_1st_quartile))
        f.write("Median: {}\n".format(diagnoses_median))
        f.write("3rd Quartile: {}\n".format(diagnoses_3rd_quartile))
        f.write("Max: {}\n".format(diagnoses_max))

        # Procedure Codes
        f.write("\nProcedure Codes\n")
        f.write("Mean: {}\n".format(procedure_mean))
        f.write("Standard Deviation: {}\n".format(procedure_stdev))
        f.write("Mode: {}\n".format(procedure_mode))
        f.write("Minimum: {}\n".format(procedure_min))
        f.write("1st Quartile: {}\n".format(procedure_1st_quartile))
        f.write("Median: {}\n".format(procedure_median))
        f.write("3rd Quartile: {}\n".format(procedure_3rd_quartile))
        f.write("Max: {}\n".format(procedure_max))

        # Medication codes
        f.write("\nMedication Codes\n")
        f.write("Mean: {}\n".format(medication_mean))
        f.write("Standard Deviation: {}\n".format(medication_stdev))
        f.write("Mode: {}\n".format(medication_mode))
        f.write("Minimum: {}\n".format(medication_min))
        f.write("1st Quartile: {}\n".format(medication_1st_quartile))
        f.write("Median: {}\n".format(medication_median))
        f.write("3rd Quartile: {}\n".format(medication_3rd_quartile))
        f.write("Max: {}\n".format(medication_max))
    
# Convert our string sequences to our int sequences
# Inputs: sequences - List of visits containing a list of medical codes for each visit for each pateint
# Inputs: types: - A dictionary containing that maps medical codes to its number value
# Output: seq - A updated versionn of sequences which replace each medical code with number values
# Output: types - Updated version of types that maps new medical codes to number values
def convert_to_int_sequences(sequences, types):
    # Firstly, create a new list of sequences
    seqs = []
    
    # Next, go through each pid in sequences
    for pid in sequences:
        # Next, create a new list for that patient
        patient = []

        # Go through each visit of the patient
        for visit in sequences[pid]:
            # Next, create a new list for visit 
            nextVisit = []

            # Go through each code in the visit 
            for code in visit:
                # Check if the code is in types
                # If code is not in types
                if code not in types:
                    # Set the value of code to be len of types
                    types[code] = len(types)+1
                
                # Convert the code to an integer and add it to the visit
                nextVisit.append(types[code])
            
            # Next, add the visit to the patient
            patient.append(nextVisit)
        
        # Next, add each patient to seqs
        seqs.append(patient)

    # Finally, return both seqs, and types
    return seqs, types

# Create the dataset
# Input: hfSet - Sequence of visits for each patient that has heart failure
# Input: nonHFSet - Sequence of visits for each pateint that does not have heart failure
# Input: validation_split - The precentage of how much the dataset should be used for validation
# Input: test_split - The percentage of how much the dataset should be used for testing
# Output: trainSeqs - List of sequence of visits for the training set
# Output: trainLabels - List of labels for the training set
# Output: validationSeqs - List of sequence of visits for the validation set
# Output: validationLabels - list of labels for the validation set
# Output: testSeqs - List of sequence of visits for the test set
# Output: testLabels - List of labels for the test set
def createDataset(hfSet, nonHFSet, validation_split, test_split):
    # First, create two sets, one with labels and one with outputs
    inputs = []
    labels =[]

    # Next, go through each patient in hfSet
    for patient in hfSet:
        # Add patient to inputs
        inputs.append(patient)

        # Add 1 to labels
        labels.append(1)

    # Next, go through each patient in nonHFSet
    for patient in nonHFSet:
        # Add patient to inputs
        inputs.append(patient)

        # Add labels to inputs
        labels.append(0)

    # Split up the inputs and labels
    trainSeqs, testSeqs, trainLabels, testLabels = train_test_split(inputs, labels, test_size=test_split, random_state=1, stratify=labels)

    # Get ratio of validation size
    validationSize = validation_split / (1-test_split)

    # Split up trainSeqs and testSeqs into train and validation sets
    trainSeqs, validationSeqs, trainLabels, validationLabels = train_test_split(trainSeqs, trainLabels, test_size=validationSize, random_state=1, stratify=trainLabels)

    # Return all datasets
    return trainSeqs, trainLabels, validationSeqs, validationLabels, testSeqs, testLabels
    
# Main function
if __name__ == "__main__":
    # First, we need the following files
    # ADMISSIONS.csv, DIAGNOSES_ICD.csv, CPTEVENT.csv, procedures.groups, diagnoses.gorups, and PRESCRIPTIONS.csv
    admissionsFile = "ADMISSIONS.csv"
    diagnosesFile = "DIAGNOSES_ICD.csv"
    procedureFile = "CPTEVENTS.csv"
    procedureGroupsFile = "procedures.groups"
    diagnosesGroupsFile = "diagnoses.groups"
    medicationFile = "PRESCRIPTIONS.csv"

    # Make two maps, PID TO ADMID and ADMID TO ADMTIME
    pidAdmMap, admDateMap = map_PID_AMID_DATE(admissionsFile)

    # Map a map for Admission ID to HF
    admHFMap = map_AdmID_HF(diagnosesFile)

    # List all HFs for each patient
    # pidAdmDateMap is not used. It is only there to make sure that pidHFsMap
    # is have the correct sequence of visits for each patient
    pidHFsMap, pidAdmDateMap = map_PID_HF_Date(pidAdmMap, admDateMap, admHFMap)
    
    # Get the patients with heart failure and the index of when they first receive heart failure
    hfPatients = find_HF_Patients(pidHFsMap)

    # Get the patients without heart failure
    nonHFPatients = find_non_HF_Patients(pidHFsMap)

    # Map each admID to its diagnoses code
    dataset = map_diagnoses(diagnosesFile, diagnosesGroupsFile)

    # Map each admID to its procedure code
    dataset = map_procedures(procedureFile, procedureGroupsFile, dataset)

    # Map each admID to its medication code
    dataset = map_medication(medicationFile, dataset)

    # Create our sequences from the dataset
    sequences, hfPatients = create_sequences(dataset, admDateMap, pidAdmMap, hfPatients)
    
    # Next, we focus on getting our final list of sequences
    hf_sequences, nonhf_sequences = create_final_sequences(sequences, hfPatients, nonHFPatients)

    # Next, we convert the string sequences into into sequences
    # We first create a new map called types which keep track of the number of value of a code
    types = {}

    # Next, we covert the hf_sequences into int sequences by passing in types
    hf_seqs, types = convert_to_int_sequences(hf_sequences, types)
    
    # Next, we convert the nonhf_sequences into int sequences
    nonHF_seqs, types = convert_to_int_sequences(nonhf_sequences, types)

    # Next, we divide up both hf_seqs and nonHF_seqs to their train, validation, and test sets
    # Using a 0.75:0.10:015 ratio
    trainSeqs, trainLabels, validationSeqs, validationLabels, testSeqs, testLabels = createDataset(hf_seqs, nonHF_seqs, 0.10, 0.15)

    """
    # Finally, save the all sequecnes for the training, validation, and test sets using pickle
    pickle.dump(trainSeqs, open('train.seqs', 'wb'), -1)
    pickle.dump(validationSeqs, open('valid.seqs', 'wb'), -1)
    pickle.dump(testSeqs, open('test.seqs', 'wb'), -1)
    pickle.dump(trainLabels, open('train.labels', 'wb'), -1)
    pickle.dump(validationLabels, open('valid.labels', 'wb'), -1)
    pickle.dump(testLabels, open('test.labels', 'wb'), -1) 
    """

    # Focus on anaylysize the dataset
    # Analyze sequence for heart failure
    analyze_sequences(hf_sequences, "Codes_With_Group_HF_Seqs_Statistics.txt")

    # Analyze sequence for non_hf sequences
    analyze_sequences(nonhf_sequences, "Codes_With_Group_NonHF_Seqs_Statistics.txt")

    # Add both sequences together
    seqs = {}
    for patient in hf_sequences:
        seqs[patient] = hf_sequences[patient]
    for patient in nonhf_sequences:
        seqs[patient] = nonhf_sequences[patient]

    # Analyze all sequences
    analyze_sequences(seqs, "Codes_With_Goup_All_Seqs_Statistics.txt")

    # Create dataset without any code grouping
    # Map each admID to its diagnoses code without group
    dataset_without_group = map_diagnoses_without_group(diagnosesFile)

    # Map each admID to its procedure code without grouping
    dataset_without_group = map_procedures_without_grouping(procedureFile, dataset_without_group)

    # Map each admID to its medication code
    dataset_without_group = map_medication(medicationFile, dataset_without_group)

    # Create sequences from dataset
    sequences_without_group, hfPatients = create_sequences(dataset_without_group, admDateMap, pidAdmMap, hfPatients)
    
    # Next, we focus on getting our final list of sequences
    hf_sequences_wihout_group, nonhf_sequences_without_group = create_final_sequences(sequences_without_group, hfPatients, nonHFPatients)

    # Anlyze hf_sequecnes_wihtout_group
    analyze_sequences(hf_sequences_wihout_group, "Codes_Without_Group_HF_Seqs_Statisitcs.txt")

    # Analyze  nonHF_sequences_wihtout_group
    analyze_sequences(nonhf_sequences_without_group, "Codes_Without_Group_NonHF_Seqs_Statistics.txt")

    # Analyze all sequences without group
    # Add both sequences together
    seqs_without_grouping = {}
    for patient in hf_sequences_wihout_group:
        seqs_without_grouping[patient] = hf_sequences_wihout_group[patient]
    for patient in nonhf_sequences_without_group:
        seqs_without_grouping[patient] = nonhf_sequences_without_group[patient]

    # Analyze seqs_without_grouping
    analyze_sequences(seqs_without_grouping, "Codes_Without_Group_All_Seqs_Statistics.txt")
    
    # Next, create types for without group
    without_group_types = {}
    # Next, we covert the hf_sequences_wihout_group into int sequences by passing in without_group_types
    _, without_group_types = convert_to_int_sequences(hf_sequences_wihout_group, without_group_types)
    
    # Next, we convert the nonhf_sequences into int sequences
    _, without_group_types = convert_to_int_sequences(nonhf_sequences_without_group, without_group_types)

    # Print the number of codes being used in each dataset
    print("Number of codes used in dataset with grouping: ", len(types))
    print("Number of codes used in dataset without grouping: ", len(without_group_types))

    # Save print information 
    with open("Additional_Codes_Statistics.txt", "w") as f:
        f.write("Number of codes used in dataset with grouping: {}\n".format(len(types)))
        f.write("Number of codes used in dataset without grouping: {}\n".format(len(without_group_types)))