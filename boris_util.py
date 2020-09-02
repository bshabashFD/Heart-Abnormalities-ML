import numpy as np
import pandas as pd
import librosa


# will need this to read wav file
from scipy.io.wavfile import read


def _read_in_files(file_name_dataframe):
    '''
        This function accepts a dataframe which contains
        file names as well as their labels. It reads in
        the audio files' content and creates a new
        dataframe with the signal in it and the labels
        from the original file
        
        If the signal contains 10,000 measurements, the
        resulting dataframe will have 10,001 columns
        (10,000 measurements + 1 label)
    '''
    
    # Step 1: Find the longest file and its size
    max_size = 0
    for i, row in file_name_dataframe.iterrows():
        file_name = "453_923_bundle_archive/"+row['fname']

        a = read(file_name)
        file_as_array = np.array(a[1],dtype=float)
        if max_size < file_as_array.shape[0]:
            max_size = file_as_array.shape[0]

    
    print(f"Longest file has {max_size} measurements")

    # Step two, create an empty placeholder for 
    # each file, fill in its data, and append it
    # to a list
    list_of_files = []

    for i, row in file_name_dataframe.iterrows():
        file_name = "453_923_bundle_archive/"+row['fname']

        a = read(file_name)
        file_as_array = np.array(a[1],dtype=float)

        # The placeholder is the same size as the largest file
        # so all resulting rows end up with as many column
        # as the longest file
        placeholder_array = np.zeros((max_size,))
        placeholder_array[-file_as_array.shape[0]:] = file_as_array[:]

        list_of_files.append(placeholder_array)

    # Now we just convert the list of file data into
    # a pandas dataframe

    file_name_as_numbers_dataframe = pd.DataFrame(data=np.array(list_of_files))
    file_name_as_numbers_dataframe['label'] = file_name_dataframe['label']
    return file_name_as_numbers_dataframe

############################################################

def read_original_data(folder_location):

    '''
        This function reads the files present in 
        [folder_location] and returns the files
        as a single dataframe containing
        the audio measurements as features (usually many
        columns since each audio file has many measurements)
        and the label as the target.

    '''

    # Read files and drop unlabeled test set used on Kaggle
    set_a_df = pd.read_csv(folder_location+'/set_a.csv')
    set_b_df = pd.read_csv(folder_location+'/set_b.csv')

    # remove unlabeled files within each dataframe, all other files are named in the CSV
    # as they are on the hard drive
    set_a_df = set_a_df[~set_a_df['label'].isna()]
    set_b_df = set_b_df[~set_b_df['label'].isna()]

    # Perform string operations to match names to their filename on the hard drive

    set_b_df['fname'] = set_b_df['fname'].str.replace('Btraining_', '')
    set_b_df['fname'] = set_b_df['fname'].str.replace('normal_', 'normal__')
    set_b_df['fname'] = set_b_df['fname'].str.replace('murmur_', 'murmur__')
    set_b_df['fname'] = set_b_df['fname'].str.replace('extrastole_', 'extrastole__')
    set_b_df['fname'] = set_b_df['fname'].str.replace('normal__noisynormal__', 'normal_noisynormal_')
    set_b_df['fname'] = set_b_df['fname'].str.replace('murmur__noisymurmur__', 'murmur_noisymurmur_')

    # Combine the dataframes
    combined_df = pd.concat([set_a_df, set_b_df], axis=0)

    # Reset the index and then drop the old index
    combined_df = combined_df.reset_index(drop=True)

    combined_as_number_df = _read_in_files(combined_df)

    return combined_as_number_df



############################################################
def read_and_combine_data(folder_location, read_noisy_data=True):

    '''
        This function reads the files present in 
        [folder_location] and returns the files
        as a single dataframe containing
        the file names
        and the label as the target.

    '''

    # Read files and drop unlabeled test set used on Kaggle
    set_a_df = pd.read_csv(folder_location+'/set_a.csv')
    set_b_df = pd.read_csv(folder_location+'/set_b.csv')

    # remove unlabeled files within each dataframe, all other files are named in the CSV
    # as they are on the hard drive
    set_a_df = set_a_df[~set_a_df['label'].isna()]
    set_b_df = set_b_df[~set_b_df['label'].isna()]

    # Perform string operations to match names to their filename on the hard drive

    set_b_df['fname'] = set_b_df['fname'].str.replace('Btraining_', '')
    set_b_df['fname'] = set_b_df['fname'].str.replace('normal_', 'normal__')
    set_b_df['fname'] = set_b_df['fname'].str.replace('murmur_', 'murmur__')
    set_b_df['fname'] = set_b_df['fname'].str.replace('extrastole_', 'extrastole__')
    set_b_df['fname'] = set_b_df['fname'].str.replace('normal__noisynormal__', 'normal_noisynormal_')
    set_b_df['fname'] = set_b_df['fname'].str.replace('murmur__noisymurmur__', 'murmur_noisymurmur_')

    # Combine the dataframes
    combined_df = pd.concat([set_a_df, set_b_df], axis=0)

    # Reset the index and then drop the old index
    combined_df = combined_df.reset_index(drop=True).drop(['sublabel', 'dataset'], axis=1)

    if not (read_noisy_data):
        combined_df = combined_df[~combined_df['fname'].str.contains('noisy')]

    # read the signals and sampling rates into two lists
    signals = []
    sampling_rates = []

    

    for index, row in combined_df.iterrows():
        y, sr = librosa.load(folder_location+"/"+row['fname'])
        signals.append(y)
        sampling_rates.append(sr)

        print(f"processed {round(100*index/combined_df.shape[0], 2)}%", end="\r")
    
    print(f"processed 100.00%")

    # then conver them into a dataframe
    signal_df = pd.DataFrame({'signal': signals, 
                              'sampling_rate': sampling_rates,
                              'label': combined_df['label']})


    return signal_df
############################################################
############################################################
############################################################
############################################################
############################################################


