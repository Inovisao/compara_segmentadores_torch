import numpy as np
import os
import torch
import random
import json
from PIL import Image
from pycocotools.coco import COCO

def load_class_data():
    filename = os.path.join("..", "data", "annotations_coco_json", "class_data.json")
    
    with open(filename, 'r') as f:
        class_data = json.load(f)
    
    return class_data

def move_anns():
    anns_file = os.path.join("..", "data", "all", "imagens", "_annotations.coco.json") 
    coco_annotations_dir = os.path.join("..", "data", "annotations_coco_json") 

    if os.path.exists(anns_file):
        if (not os.path.exists(coco_annotations_dir)):
            os.mkdir(coco_annotations_dir)        
        print("dor e sofrimento") 
        os.rename(anns_file, os.path.join(coco_annotations_dir, "_annotations.coco.json"))


def coco2binary(color_map):
    # Directory to save annotations
    masks_dir = "../data/masks"

    # Directory with all images and annotations .json
    annotations_file = os.path.join("..", "data", "annotations_coco_json", "_annotations.coco.json")

    # Read annotations
    coco = COCO(annotations_file)

    # Create directory to store masks
    if (not os.path.exists(masks_dir)):
        os.mkdir(masks_dir)

    # Iterate over annotations
    for img_id, img in coco.imgs.items():
        # Get image name
        img_name = img["file_name"][:-len(img["file_name"].split('.'))]

        # Print progress
        print(f"Processing image {img_name}")

        # Create and initialize the  empty array to store the mask
        mask = np.ndarray(shape=(img["height"], img["width"], 3), dtype=np.uint8)
        mask[:,:] = 0

        # Create mask and add to array
        for ann in coco.imgToAnns[img_id]:
            img_id = ann["image_id"]
            cat_id = ann["category_id"]
            
            for i in range(3):
                mask[:, :, i] += coco.annToMask(ann)
                mask[:, :, i][mask[:, :, i] == cat_id] = color_map[cat_id][i]
            
        

        # Save mask in .png format
        mask = Image.fromarray(mask, mode="RGB")
        mask_name = str(f"{img_name}.png")
        mask.save(os.path.join(masks_dir, mask_name))


def generate_unique_colors(n):
    colors = set()
    color_list = []

    while len(color_list) < n:
        # Generate random color in RGB space
        color = (random.random(), random.random(), random.random())
        color_tensor = torch.tensor(color)

        # Verify for uniqueness
        if color_tensor not in colors:
            colors.add(color_tensor)
            color_list.append(color_tensor)

    return color_list


def update_classes_from_json(file_path):
    """
    Função para extrair classes do arquivo JSON.

    Returns:
        all_classes: lista com os nomes das classes.
        supercategory_color_map: dict contendo os ids das classes como chaves e as cores como valores.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
        
        all_classes = list()
        supercategory_color_map = dict()

        for category in data["categories"]:
            category_id = category["id"]
            category_name = category["name"]

            # For background
            if category["supercategory"] == "none":
                category_name = "fundo"
                color = (0, 0, 0)
            else:
                # Generate color and rescale it to [0, 255]
                if category_id not in supercategory_color_map:
                    color = tuple(int(x * 255) for x in generate_unique_colors(1)[0])

            # Assign color to class
            supercategory_color_map[category_id] = color
            
            all_classes.append(category_name)

    return all_classes, supercategory_color_map



if __name__ == "__main__":
    # Caminho para o arquivo JSON
    move_anns()
    json_file_path = '../data/annotations_coco_json/_annotations.coco.json'

    ALL_CLASSES, supercategory_color_map = update_classes_from_json(json_file_path)
    print("Classes:", ALL_CLASSES)
    print("Color map:", supercategory_color_map)

    # Crie uma lista de cores mapeadas para as classes
    LABEL_COLORS_LIST = [supercategory_color_map[ALL_CLASSES.index(category)] for category in ALL_CLASSES]
    VIS_LABEL_MAP = [supercategory_color_map[ALL_CLASSES.index(category)] for category in ALL_CLASSES]

    print("Gerando máscaras...")
    coco2binary(supercategory_color_map)
    print("Feito!")

    new_data = {"ALL_CLASSES": ALL_CLASSES,
                "supercategory_color_map": supercategory_color_map,
                "LABEL_COLORS_LIST": LABEL_COLORS_LIST,
                "VIS_LABEL_MAP": VIS_LABEL_MAP}

    with open(os.path.join("..", "data", "annotations_coco_json", "class_data.json"), 'w') as f:
        json.dump(new_data, f)
        
