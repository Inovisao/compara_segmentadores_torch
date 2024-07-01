import os
import shutil


def swap_folders(folder1, folder2):
    # Ensure both folders exist; create them if they don't
    os.makedirs(folder1, exist_ok=True)
    os.makedirs(folder2, exist_ok=True)

    # List files in both folders
    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2)

    # Rename files in folder1 if they conflict with files in folder2
    renamed_files = {}
    for file in files1:
        if file in files2:
            new_name = "new_" + file
            src = os.path.join(folder1, file)
            dst = os.path.join(folder1, new_name)
            os.rename(src, dst)
            renamed_files[file] = new_name
            print(f"Renamed {file} to {new_name} in {folder1}")

    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2)

    # Move all files from folder1 to folder2
    for file in files1:
        src = os.path.join(folder1, file)
        dst = os.path.join(folder2, file)
        shutil.move(src, dst)
        print(f"Moved {file} from {folder1} to {folder2}")

    # Move all files from folder2 to folder1
    for file in files2:
        src = os.path.join(folder2, file)
        dst = os.path.join(folder1, file)
        shutil.move(src, dst)
        print(f"Moved {file} from {folder2} to {folder1}")

    # Rename files back to their original names in folder1 if they were renamed
    for original, new_name in renamed_files.items():
        src = os.path.join(folder2, new_name)
        dst = os.path.join(folder2, original)
        os.rename(src, dst)
        print(f"Renamed {new_name} back to {original} in {folder2}")


def delete_files(folder, filenames):
    # Delete specific files in the given folder
    for filename in filenames:
        file_path = os.path.join(folder, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted {filename} from {folder}")
        else:
            print(f"{filename} not found in {folder}"


# Example usage:
if __name__ == "__main__":
    folder1="/home/corbusier/Downloads/train"
    folder2="/home/corbusier/development/compara_segmentadores_torch_/data/all/imagens"

    swap_folders(folder1, folder2)

    folder1="/home/corbusier/Downloads/train_cocojson"
    folder2="/home/corbusier/development/compara_segmentadores_torch_/data/annotations_coco_json"

    delete_files(folder2, ["class_data.json"])
    swap_folders(folder1, folder2)
