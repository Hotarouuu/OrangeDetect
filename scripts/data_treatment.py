import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--PATH', type=str, help='Data path')


args = parser.parse_args()
path_raw = args.PATH

field = os.listdir(path_raw)
path = os.path.join(path_raw, field[1])

if path == None:
    raise TypeError('Add the path to the script with --PATH "path/to/data"')
else:

    def get_class(path):
        name = os.path.basename(os.path.dirname(path)).lower()
        if 'canker' in name:
            return 'citrus_canker'
        elif 'healthy' in name:
            return 'healthy'
        elif 'melanose' in name:
            return 'melanose'
        return 'unknown'

    def copy_files(file_paths, base_dest):
        for path in file_paths:
            class_name = get_class(path)
            class_dest = os.path.join(base_dest, class_name)
            os.makedirs(class_dest, exist_ok=True)
            shutil.copy(path, os.path.join(class_dest, os.path.basename(path)))

    # --- Data Processing Start ---

    folders = os.listdir(path)

    citrus_canker = [os.path.abspath(os.path.join(path, folders[0], f)) for f in os.listdir(os.path.join(path, folders[0]))]
    healthy       = [os.path.abspath(os.path.join(path, folders[1], f)) for f in os.listdir(os.path.join(path, folders[1]))]
    melanose      = [os.path.abspath(os.path.join(path, folders[2], f)) for f in os.listdir(os.path.join(path, folders[2]))]

    print('Starting data processing and splitting...\n')

    # Check and balance the classes
    if len(set(map(len, [citrus_canker, healthy, melanose]))) != 1:
        print('Classes are imbalanced:')
        print(f'Citrus Canker: {len(citrus_canker)} | Healthy: {len(healthy)} | Melanose: {len(melanose)}')

        min_len = min(len(citrus_canker), len(healthy), len(melanose))
        citrus_canker, healthy, melanose = citrus_canker[:min_len], healthy[:min_len], melanose[:min_len]

        print('\nBalanced classes:')
        print(f'Citrus Canker: {len(citrus_canker)} | Healthy: {len(healthy)} | Melanose: {len(melanose)}')

    # Split the data
    train = citrus_canker[:2000] + healthy[:2000] + melanose[:2000]
    test  = citrus_canker[2000:2300] + healthy[2000:2300] + melanose[2000:2300]
    eval  = citrus_canker[2300:2600] + healthy[2300:2600] + melanose[2300:2600]

    base_dir = os.path.join(path_raw, 'processed')
    os.makedirs(base_dir, exist_ok=True)  # Ensure the base directory exists
    dirs = {split: os.path.join(base_dir, split) for split in ['train', 'test', 'eval']}

    for folder in dirs.values():
        os.makedirs(folder, exist_ok=True)

    # Copy the files to their respective directories
    print('\nCopying training files...')
    copy_files(train, dirs['train'])

    print('Copying testing files...')
    copy_files(test, dirs['test'])

    print('Copying evaluation files...')
    copy_files(eval, dirs['eval'])

    print('\nProcess completed successfully!')

    # Removing raw folder
    shutil.rmtree(path=path)
