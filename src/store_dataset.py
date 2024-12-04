import os
import numpy as np
from PIL import Image
import csv

def writeIntoCsv(dataset_path: str, csv_file_path: str, size: tuple[int, int], limit: int = 500,):
    inputs = []
    for class_index, class_name in enumerate(os.listdir(dataset_path)):
        print(class_name)
        class_path = os.path.join(dataset_path, class_name)
        i = 0
        for image in os.listdir(class_path):
            if i == limit:
                break
            i += 1
            image_path = os.path.join(class_path, image)
            img = Image.open(image_path)
            img = img.resize(size)
            img = img.convert('L')
            img = np.array(img).flatten() / 255
            img = np.insert(img, 0, class_index)
            inputs.append(img)
    inputs = np.array(inputs)
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(inputs)

def main():
    active_directory = os.path.split(__file__)[0]
    project_directory = os.path.split(active_directory)[0]

    dataset_path = input("Insert dataset path: ")
    dataset_path = os.path.join(project_directory, dataset_path)
    csv_path = os.path.join(project_directory, "src/dataset.csv")
    limit = int(input("Insert maximum number of images to store for each category"))
    size = int(input("Insert size: "))
    writeIntoCsv(dataset_path, csv_path, (size, size), limit=limit)

if __name__ == "__main__":
    main()