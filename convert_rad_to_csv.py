import pandas as pd
import numpy as np
from matplotlib import pyplot as pp
import os

pd.set_option('display.max_rows', 100000)
pd.set_option('display.max_columns', 100000)

rad_dir = 'C:/Users/...' #Location of .RAD files
output_dir = 'C:/Users/...' #Destination for .csv files

rad_files = [] #list to hold .RAD file names

#Scrape all .RAD file names in the directory above
for file in os.listdir(rad_dir):
    if file.endswith('.rad'):
        rad_files.append(file) 

#Loop through all .RAD files in the specified directory, create a dataframe, save dataframe as .csv
for file in rad_files:
    
    #Change current directory back to location where source .RAD files live
    os.chdir(rad_dir)
    
    #Read in .RAD file
    data = pd.read_csv(file, sep = "\t", skiprows = [0,1,2,3,4,5,6,7,8,9,10,11,12,13])
    
    #Copy .RAD file name except for '.RAD' and replace it with '.csv
    new_file_name = file[0:-4] + '.csv'
    
    #Initializing lists to hold all associated data in the .RAD file
    samples = []
    time = []
    speed = []
    accel = []
    dist = []

    #This basically just extracts all the rows of data and appends to the above lists
    # it's very clunky because the .RAD file are so whack/inconsistently delimeted
    for row in data.index[0:-7]:

        x = data.loc[row, :].str.split(" ")[0]

        filtered = []
        for element in x:

            if element == '':
                continue
            else:
                filtered.append(element)

        samples.append(filtered[0])
        time.append(filtered[1])
        speed.append(filtered[2])
        accel.append(filtered[3])
        dist.append(filtered[4])

    #Create dataframe with all the completed lists with the data
    csv_file = pd.DataFrame({'Samples': samples,
                            'Time': time,
                            'Speed': speed,
                            'Accel': accel,
                            'Dist': dist})
    
    #Change current directory to output directory to save csv
    os.chdir(output_dir)
    
    #Save csv
    csv_file.to_csv(new_file_name)
