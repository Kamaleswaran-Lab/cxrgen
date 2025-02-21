import pandas as pd
import numpy as np
import os
from pathlib import Path
import multiprocessing as mp
from collections import defaultdict, Counter
from functools import partial
import pickle

columns = {
    'age', 'gender', 'race', 'ethnicity', 'cci9', 'cci10', 
    'bed_type', 'icu_type', 'covid', 'procedure', 
    'vent_mode', 'vent_rate_set', 'vent_tidal_rate_set',
    'vent_tidal_rate_exhaled', 'peep', 'Oxygen_Flow_Rate',
}

def get_empty_stats():
    """Create a fresh stats dictionary for each process"""
    return {
        'age_min': float('inf'),  
        'age_max': -float('inf'),  
        'age_nan': 0,
        'cci9_min': float('inf'),
        'cci9_max': -float('inf'),
        'cci9_nan': 0,
        'cci10_min': float('inf'),
        'cci10_max': -float('inf'),
        'cci10_nan': 0,
        'vent_rate_set_min': float('inf'),
        'vent_rate_set_max': -float('inf'),
        'vent_rate_set_nan': 0,
        'vent_tidal_rate_set_min': float('inf'),
        'vent_tidal_rate_set_max': -float('inf'),
        'vent_tidal_rate_set_nan': 0,
        'vent_tidal_rate_exhaled_min': float('inf'),
        'vent_tidal_rate_exhaled_max': -float('inf'),
        'vent_tidal_rate_exhaled_nan': 0,
        'peep_min': float('inf'),
        'peep_max': -float('inf'),
        'peep_nan': 0,
        'Oxygen_Flow_Rate_min': float('inf'),
        'Oxygen_Flow_Rate_max': -float('inf'),
        'Oxygen_Flow_Rate_nan': 0,
        'covid': 0,
        'procedure': 0,
        'gender': Counter(),
        'race': Counter(),
        'ethnicity': Counter(),
        'icu_type': Counter(),
        'bed_type': Counter(),
        'vent_mode': Counter()
    }

def process_supertable(supertable_path):
    """Process a single supertable and return its stats"""
    local_stats = get_empty_stats()
    supertable = pd.read_pickle(supertable_path)
    save = False

    for column in columns:
        if column in supertable.columns:
            if column in {'age', 'cci9', 'cci10', 'vent_rate_set', 'vent_tidal_rate_set',
                         'vent_tidal_rate_exhaled', 'peep', 'Oxygen_Flow_Rate'}:
                if supertable[column].dtype == 'object':
                    supertable[column] = pd.to_numeric(supertable[column], errors='coerce')
                min_val, max_val = supertable[column].min(), supertable[column].max()
                if np.isnan(min_val):
                    local_stats[column + '_nan'] += 1
                else:
                    local_stats[column + '_min'] = min_val
                    local_stats[column + '_max'] = max_val
            elif column in {'covid', 'procedure'}:
                local_stats[column] += supertable[column].max() if not np.isnan(supertable[column].max()) else 0
            else:
                unique_vals = supertable[column].unique()
                for val in unique_vals:
                    if not pd.isna(val):
                        local_stats[column][val] += 1
        else:
            local_stats[column] = None
            supertable[column] = [np.nan]*len(supertable)
            save = True
    
    if save:
        supertable.to_pickle(supertable_path)
        print("Added missing columns to", supertable_path)
    
    return local_stats

def merge_stats(stats_list):
    """Merge stats from multiple processes"""
    final_stats = get_empty_stats()
    
    for local_stats in stats_list:
        for key, value in local_stats.items():
            if isinstance(value, Counter):
                final_stats[key].update(value)
            elif key.endswith('_min'):
                final_stats[key] = min(final_stats[key], value)
            elif key.endswith('_max'):
                final_stats[key] = max(final_stats[key], value)
            elif isinstance(value, (int, float)):
                final_stats[key] += value
            elif value is None:
                final_stats[key] = None
    
    return final_stats

def main():
    root = Path('/hpc/group/kamaleswaranlab/EmoryDataset/Images/chest_xrays')
    supertable_path = root / 'matched_supertables_with_images'
    supertable_template = "_timing_corrected.pickle"

    supertables = list(supertable_path.glob("*" + supertable_template))

    cpu_count = mp.cpu_count()
    
    with mp.Pool(cpu_count) as pool:
        results = pool.map(process_supertable, supertables)
    # Merge results from all processes
    final_stats = merge_stats(results)
    
    with open(root / 'supertable_stats.pickle', 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    main()
    

    