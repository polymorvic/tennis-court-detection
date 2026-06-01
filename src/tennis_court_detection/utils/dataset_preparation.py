from pathlib import Path
from collections import defaultdict
import cv2
from sklearn.model_selection import train_test_split
from tennis_court_detection.utils.helpers import pipette_color
import pandas as pd


def prepare_dataset_for_shot_classification(
    images_path: Path, 
    skip_images_path: Path,
    dataset_file_output_path: Path
) -> None:

    images = [file.name for file in images_path.glob('*.png')]
    skip_images = [file.name for file in skip_images_path.glob('*.png')]

    skip_train, skip_temp = train_test_split(skip_images, test_size=0.3, random_state=123)
    skip_val, skip_test = train_test_split(skip_temp, test_size=0.5, random_state=123)

    img_color_split = defaultdict(list)
    for name in images:
        img = cv2.cvtColor(cv2.imread(str(images_path / name)), cv2.COLOR_BGR2HSV)
        hue_value = pipette_color(img)[0]
        
        if 0 <= hue_value < 15:
            img_color_split['color_group_1'].append(name)

        elif 15 <= hue_value < 50:
            img_color_split['color_group_2'].append(name)

        elif 50 <= hue_value < 115:
            img_color_split['color_group_3'].append(name)

        else:
            img_color_split['color_group_4'].append(name)

    img_color_dataset = {}
    img_color_dataset_length = {}
    for key, img_names in img_color_split.items():

        img_train, img_temp = train_test_split(img_names, test_size=0.3, random_state=123)
        img_val, img_test = train_test_split(img_temp, test_size=0.5, random_state=123)

        img_color_dataset[key] = img_train, img_val, img_test
        img_color_dataset_length[key] = len(img_train), len(img_val), len(img_test)


    img_datasets = ([], [], [])
    skip_datasets = (skip_train, skip_val, skip_test)
    for t in range(3):
        m = max([val[t] for val in img_color_dataset_length.values()])


        for i in range(m):
            should_break = False
            for val in img_color_dataset.values():

                try:
                    img = val[t][i]

                except Exception:
                    continue

                img_datasets[t].append(img)

                if len(img_datasets[t]) == len(skip_datasets[t]):
                    should_break = True
                    break

            if should_break:
                break

    dataset_names = (['train', 'val', 'test'])
    dfs = []
    for t in range(3):

        print(len(img_datasets[t]), len(skip_datasets[t]))


        df = pd.DataFrame({
            'img_name': img_datasets[t] + skip_datasets[t],
            'dataset': [dataset_names[t]] * (len(img_datasets[t]) + len(skip_datasets[t])),
            'label': [1] * len(img_datasets[t]) + [0] * len(skip_datasets[t])
        })
        dfs.append(df)

    df_dataset = pd.concat(dfs)

    df_dataset.to_csv(dataset_file_output_path, index=False)


    

