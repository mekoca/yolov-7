from zipfile import ZipFile, BadZipFile
import os

def extract_zip_file(zip_path, extract_path):
    try:
        with ZipFile(zip_path, 'r') as zfile:
            zfile.extractall(extract_path)
        os.remove(zip_path)
    except FileNotFoundError:
        print(f"Error: The file {zip_path} does not exist.")
    except BadZipFile:
        print(f"Error: The file {zip_path} is a bad zip file and cannot be extracted.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Adjust these paths according to where you've stored the zip files on your local machine
zip_train_path = "/yolov7/train2017.zip"
zip_val_path = "/yolov7/val2017.zip"
zip_ann_path = "/yolov7/annotations_trainval2017.zip"

extract_zip_file(zip_train_path, "./coco_train2017")
extract_zip_file(zip_val_path, "./coco_val2017")
extract_zip_file(zip_ann_path, "./coco_ann2017")




