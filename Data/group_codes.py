# Create a map that will map each group to a list of diagnoses
import pickle

# Create single level diagnoses groups 
# Input: diagnosesFile - The file used to group diagnososes codes together
# Output: diagnosesGroups - Contains code groups for diagnoses codes
def single_level_group_diagnoses(diagnosesFile):
    # Map each group to diagnoses groups
    diagnosesGroups = {}

    # Keep track if we should add a group or not
    readGroup = False
    
    # Open and read diagnosesFile
    infd = open(diagnosesFile, 'r')
    infd.readline()

    # Keep track of the currentGroup
    currentGroup = ""

    # Read each line in diagnosesFile
    for line in infd:
        # If readGroup is False
        if readGroup == False:
            # If the first letter of the line is a digit
            if line[0].isdigit() == True:
                # Get the space of group
                space = line.index(" ")

                # Get the group name 
                group = "D" + line[0:space]

                # Map group to empty list
                diagnosesGroups[group] = []

                # Set currentGroup to be group
                currentGroup = group

                # Set readGroup to be true
                readGroup = True
        # Otherwise, 
        else:
            # Get length of line
            length = len(line)

            # If length is greater than 1
            if length > 1:
                # Split line into codes
                codes = line.split()

                # Go through each code in codes
                for code in codes:
                    diagnosesGroups[code] = currentGroup
            else:
                # All codes have been read for a group
                # Set readGroup to false
                readGroup = False
    
    # Close file
    infd.close()

    # Return diagnosesCodes
    return diagnosesGroups

# Group the cpt codes 
# Input: procdureFile - File that contains procedure codes
# Output: procedureGroups - Contains grouping for procedure codes
def group_cpt_codes(procedureFile):
    # Map each group to a list of cpt codes
    procedureGroups = {}

    # Open and read procedureFile
    infd = open(procedureFile, 'r')
    infd.readline()

    # Check if we are reading the firstLine
    firstLine = False
    
    # Next, go through each line in infd
    for line in infd:
        # If we are reading the firstLine, skip loop
        if firstLine == False:
            firstLine = True
            continue

        # Split the lines into its individual columns
        columns = line.split(',')

        # Next, get the group from the line
        group = "P" + columns[1]

        # Next, remove ' from the first column and split -
        columns[0] = columns[0].removeprefix("'")
        columns[0] = columns[0].removesuffix("'")
        ends = columns[0].split("-")

        suffix = ""
        prefix = ""
        lowest = ends[0]
        highest = ends[1]
        
        # Next, check if ends contains a suffix or prefix
        # If ends contain a prefix
        if(ends[0][0].isdigit() == False):
            # Set prefix to be the first digit
            prefix = ends[0][0]

            # Remove prefix from both ends
            lowest = lowest.removeprefix(prefix)
            highest = highest.removeprefix(prefix)

        # Otherwise if the ends contains a suffix
        elif(ends[0][-1].isdigit() == False):
            # Set suffix to be the last digit
            suffix = ends[0][-1]

            # Remove suffix from both ends
            lowest = lowest.removesuffix(suffix)
            highest = highest.removesuffix(suffix)

        # Convert lowest and highest to digits
        lowest = int(lowest)
        highest = int(highest)

        # Go through each string in the range from lowest to highest
        for i in range(lowest, highest+1):
            # Create a new string by padding prefix + i + suffix
            code = prefix + str(i) + suffix

            # If code is not less than 5
            if(len(code) < 5):
                # Create stirng of 0s
                zeros = ""
                for i in range(5-len(code)):
                    zeros = zeros + "0"

                # If there isn't a prefix, add code[0] + zeroes + codes[1:]
                if prefix != "":
                    code = code[0] + zeros + code[1:] 
                # Otherwise, if not, add zeros to code
                else:
                    code = zeros + code

            # Add code to map
            procedureGroups[code] =  group
    
    # Close file
    infd.close()

    # Return map
    return procedureGroups

# Main function
if __name__ == "__main__":
    # Get the files for grouping
    diagnosesFile = "AppendixASingleDX.txt"
    procedureFile = "CCS_services_procedures_v2022-1_052422.csv"

    # Group the diagnoses and procedure codes
    diagnosesGroups = single_level_group_diagnoses(diagnosesFile)
    procedureGroups = group_cpt_codes(procedureFile)

    # Finally, use pickle to store the diagnoses and procedureGroups
    pickle.dump(diagnosesGroups, open("diagnoses.groups", "wb"), -1)
    pickle.dump(procedureGroups, open("procedures.groups", "wb"), -1)
    