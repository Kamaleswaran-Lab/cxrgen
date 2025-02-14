import pandas as pd
from pathlib import Path
import os
import sys
import numpy as np



def main():
    root = Path('/hpc/group/kamaleswaranlab/EmoryDataset/Images/chest_xrays')
    metadf = pd.read_csv(root / 'metadata_with_supertables_filtered_notes_filtered_matching_times_min_series_img_paths_datetime_hashed.csv')
    print("Metadata size before filtering:", len(metadf))

    metadf = metadf.loc[(metadf.SeriesSelector != 0) & (metadf.SeriesSelector != 1001)]
    print("Metadata size after filtering out CXRs of wrong windows:", len(metadf))

    unique_encounters = metadf['ENCOUNTER_NBR'].unique()
    print("Number of unique encounters:", len(unique_encounters))

    unsuccesful_encounters = []
    timing_mismatch_encounters = []
    for encounter in unique_encounters:
        encounter_df = metadf[metadf['ENCOUNTER_NBR'] == encounter]
        if len(encounter_df) == 0:
            print("No images for encounter:", encounter)
            unsuccesful_encounters.append(encounter)
            continue 

        supertable_path = encounter_df['supertable_path_hashed'].values[0]
        if supertable_path:
            try:
                supertable = pd.read_pickle(supertable_path)
            except:
                print("Error reading supertable:", supertable_path)
                unsuccesful_encounters.append(encounter)
                continue
            supertable['cxr_timing'] = [None]*len(supertable)
            supertable['cxr_timing_approx_flag'] = [None]*len(supertable)
            for idx, row in encounter_df.iterrows():
                approx_flag = False 
                img_time = row['StudyDateTimeProcessed']
                
                #Approximate time if study time is missing, impute to the first hour of the study date
                if pd.isnull(img_time):
                    img_time = pd.to_datetime(row['StudyDate'])
                    approx_flag = True  
                else:
                    img_time = pd.to_datetime(img_time)  
                
                #If the image time is before the supertable time by 7 days, skip this image
                if (pd.to_datetime(supertable.index[0]) - img_time) > pd.Timedelta('7 days'):
                    print("Image time is before supertable time for encounter:", encounter)
                    timing_mismatch_encounters.append(encounter)
                    continue    
                
                supertable_idx = supertable.loc[supertable.index >= img_time].index
                
                #If the image time is more than 1 day after the supertable time, skip this image
                if len(supertable_idx) == 0:
                    if (img_time - pd.to_datetime(supertable.index[-1])) > pd.Timedelta('1 days'):
                        print("Image time doesn't match supertable time for encounter:", encounter)
                        timing_mismatch_encounters.append(encounter)
                        continue
                    else:
                        supertable_idx = supertable.index[-1]
                else:  
                    supertable_idx = supertable_idx[0]
                
                supertable.at[supertable_idx, 'cxr_timing'] = row['AccessionNumber'] + '_' + str(row['SeriesNumber'])
                supertable.at[supertable_idx, 'cxr_timing_approx_flag'] = approx_flag

            new_pickle_path = row['supertable_path_hashed'].replace('.pickle', '_timing_corrected.pickle')
            supertable.to_pickle(new_pickle_path)      
            print("Saved new supertable with CXR timing:", new_pickle_path)   
        else:
            unsuccesful_encounters.append(encounter)
            continue

    import numpy as np
    np.save('unsuccesful_encounters.npy', np.array(unsuccesful_encounters))
    np.save('timing_mismatch_encounters.npy', np.array(timing_mismatch_encounters))
    

if __name__ == "__main__":
    main()