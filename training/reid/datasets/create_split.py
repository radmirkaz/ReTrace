import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def save_annotations_from_cropped(csv_path, annotation_path):
    df = pd.read_csv(csv_path)

    # Добавляем фиктивные координаты (так как изображения уже обрезаны)
    df["x_1"] = 0
    df["y_1"] = 0
    df["x_2"] = 1
    df["y_2"] = 1

    # Стратифицированное разделение
    train, val = train_test_split(
        df,
        test_size=0.1,
        random_state=42,
        stratify=df["class"]
    )

    # Создание директории, если не существует
    if not os.path.isdir(annotation_path):
        os.makedirs(annotation_path)

    # Сохранение файлов
    train[["img_path", "x_1", "y_1", "x_2", "y_2", "class"]].to_csv(
        os.path.join(annotation_path, "train.txt"), index=False, sep=','
    )
    val[["img_path", "x_1", "y_1", "x_2", "y_2", "class"]].to_csv(
        os.path.join(annotation_path, "val.txt"), index=False, sep=','
    )

    print(f"✅ Saved: train ({len(train)}) and val ({len(val)}) in {annotation_path}")


if __name__ == '__main__':
    save_annotations_from_cropped("data/annotation_fixed_addon2.csv", "data/")
    