import os
from pathlib import Path


root = Path('/hpc/group/kamaleswaranlab/EmoryDataset/Images/chest_xrays')

supertable_path = root / 'matched_supertables_with_images'
supertable_template = "_timing_corrected.pickle"

supertables = list(supertable_path.glob("*" + supertable_template))

processed_supertable_template = "_processed.pickle"
processed_supertables = list(supertable_path.glob("*" + processed_supertable_template))

# Find supertables that are not processed
processed_supertables = [str(supertable).replace(processed_supertable_template, ".pickle") for supertable in processed_supertables]
unprocessed_supertables = [supertable for supertable in supertables if supertable not in processed_supertables]

print(f"Total supertables: {len(supertables)}")
print(f"Total processed supertables: {len(processed_supertables)}")
print(f"Total unprocessed supertables: {len(unprocessed_supertables)}")
print("Unprocessed supertables:")

for supertable in unprocessed_supertables:
    print(supertable)