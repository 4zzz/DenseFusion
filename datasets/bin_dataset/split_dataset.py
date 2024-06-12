import glob
import os
from pathlib import Path
import json
import numpy as np

train_classes = {
    'dataset0': 0,
    'dataset1': 1,
    'dataset2': 0,
    'dataset3': 2,
    'dataset4': 2,
    'ElavatedGrayBox': 0,
    'ElevatedGreyBox': 0,
    'ElevatedGreyFullBeer': 0,
    'FirstRealSet': 3,
    'GoldBinAdditional': 3,
    'GrayBoxPad': 0,
    'LargeWoodenBoxDynamic': 2,
    'LargeWoodenBoxStatic': 2,
    'ShallowGreyBox': 0,
    'SmalGreyBasket': 4,
    'SmallGoldenBox': 3,
    'SmallWhiteBasket': 1,
    'synth_dataset5_random_origin': 5,
    'synth_dataset6_random_origin': 6,
    'synth_dataset7_random_origin': 5,
    'synth_dataset8_random_origin': 5,
}

test_classes = {
    'TestBin': 3,
    'TestCarton': 7,  # samples of this class are not in train dir
    'TestGold': 3,
    'TestSynth': 5,
}


def save_json(obj, file):
    json_object = json.dumps(obj, indent=4)
    with open(file, "w") as outfile:
        outfile.write(json_object)


def find_suitable_samples(search_path, subdir2class):
    entries = []
    for file in os.listdir(search_path):
        path = Path(search_path, file)
        if os.path.isdir(path):
            subdir = file
            has_model = True if os.path.exists(
                Path(path, 'bin.stl')) else False
            sample_type = 'synthetic' if os.path.exists(
                Path(path, 'synthetic')) else 'real'

            # print(path, f'({type})')
            positions_glob_path = str(Path(path, '**positions.exr'))
            # print('searching for samples ', positions_glob_path)
            for entry in glob.iglob(positions_glob_path):
                positions_file = entry

                prefix = entry[:-len('_positions.exr')]

                has_rgb = True
                rgb_file = prefix + '_intensitymap.png'
                if not os.path.exists(rgb_file):
                    rgb_file = prefix + '_colors.png'
                    if not os.path.exists(rgb_file):
                        has_rgb = False

                transformation_file = prefix + '.txt'

                if has_model and has_rgb:
                    entries.append({
                        'class': subdir2class[subdir],
                        'type': sample_type,
                        'positions_file': positions_file,
                        'image_file': rgb_file,
                        'transformation_file': transformation_file,
                        'model500': str(Path(path, 'bin-500.ply')),
                        'model1000': str(Path(path, 'bin-1000.ply')),
                    })
    return entries


dataset_root = 'datasets/bin_dataset/Gajdosech_etal_2021_dataset/'
entries1 = find_suitable_samples(Path(dataset_root, 'VISIGRAPP_TRAIN'), train_classes)
entries2 = find_suitable_samples(Path(dataset_root, 'VISIGRAPP_TEST'), test_classes)
all_entries = entries1 + entries2

#train_count = len(all_entries) - 100
#test_count = 100
#train_indices = np.random.choice(np.arange(len(all_entries)), train_count, replace=False)

print(f'Found {len(all_entries)} samples')

by_class = {}
for s in all_entries:
    if s['class'] in by_class:
        by_class[s['class']].append(s)
    else:
        by_class[s['class']] = [s]

train_samples = []
test_samples = []
for c in by_class:
    entries = by_class[c]
    #print(f'{len(entries)} samples of class {c}')
    test_count = np.ceil(len(entries) * 0.13)
    indices = np.arange(len(entries))
    np.random.shuffle(indices)
    tr, te = 0, 0
    for i in indices:
        if i < test_count:
            test_samples.append(entries[i])
            te += 1
        else:
            train_samples.append(entries[i])
            tr += 1
    print(f'Added {tr} train samples and {te} test samples from class {c}')

print(f'{len(train_samples)} train samples')
print(f'{len(test_samples)} test samples')


save_json(train_samples, 'datasets/bin_dataset/train.json')
save_json(test_samples, 'datasets/bin_dataset/test.json')
