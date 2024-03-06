import os
import torch 
from PIL import Image 
import torchvision.transforms as transforms

import pickle
from tqdm import tqdm
import xml.etree.ElementTree as ET

classes = {"bus": 0, "bike": 1, "car": 2, "motor": 3, "person": 4, "rider": 5, "truck": 6}

###   Change this!!!   ###
condition = "Night-Sunny"
num_images = 2000
##########################

if condition == "Daytime_Sunny":
    image_dir = "data/Daytime_Sunny/daytime_clear/VOC2007/JPEGImages"
    annotation_dir = "data/Daytime_Sunny/daytime_clear/VOC2007/Annotations"
elif condition == "Daytime-Foggy":
    image_dir = "data/Daytime-Foggy/daytime_foggy/VOC2007/JPEGImages"
    annotation_dir = "data/Daytime-Foggy/daytime_foggy/VOC2007/Annotations"
elif condition == "Dusk-rainy":
    image_dir = "data/Dusk-rainy/dusk_rainy/VOC2007/JPEGImages"
    annotation_dir = "data/Dusk-rainy/dusk_rainy/VOC2007/Annotations"
elif condition == "Night_rainy":
    image_dir = "data/Night_rainy/night_rainy/VOC2007/JPEGImages"
    annotation_dir = "data/Night_rainy/night_rainy/VOC2007/Annotations"
elif condition == "Night-Sunny":
    image_dir = "data/Night-Sunny/Night-Sunny/JPEGImages"
    annotation_dir = "data/Night-Sunny/Night-Sunny/Annotations"
else:
    exit(1)

data, targets = [], []

count = 0
for i in tqdm(range(num_images)):
    image_file = os.listdir(image_dir)
    img_f = os.path.join(image_dir, image_file[i])
    if not os.path.isfile(img_f) or not img_f.endswith(".jpg"):
        continue
    annotation_file = image_file[i].replace("jpg", "xml")
    ann_f = os.path.join(annotation_dir, annotation_file)
    if not os.path.isfile(ann_f):
        continue

    # read image
    image = Image.open(img_f)
    image_tensor = transforms.Compose([ transforms.PILToTensor() ])(image)
    data.append(image_tensor)
    # read annotation

    XMLtree = tree = ET.parse(ann_f)
    labels, bounding_boxes = [], []
    for object in XMLtree.findall('object'):
        name = object.find('name').text
        labels.append(classes[name])
        bbox = object.find('bndbox')
        bounding_box = [float(bbox.find('xmin').text), 
                        float(bbox.find('ymin').text), 
                        float(bbox.find('xmax').text), 
                        float(bbox.find('ymax').text)]
        bounding_boxes.append(bounding_box)
    L = torch.tensor(labels, dtype=torch.long)
    B = torch.tensor(bounding_boxes, dtype=torch.float)
    target_dic = {"boxes": B, "labels": L}
    targets.append(target_dic)
    count += 1

data_size = len(data)
split1, split2 = int(data_size * 0.8), int(data_size * 0.9)
train_data, val_data, test_data = data[:split1], data[split1:split2], data[split2:]
train_target, val_target, test_target = targets[:split1], targets[split1:split2], targets[split2:]

# Pickle cannot handle dumping huge data. Use torch.save() for full data.
with open(f"{condition}_train_data.pkl", "wb") as file:
    pickle.dump(train_data, file)

with open(f"{condition}_val_data.pkl", "wb") as file:
    pickle.dump(val_data, file)

with open(f"{condition}_test_data.pkl", "wb") as file:
    pickle.dump(test_data, file)

with open(f"{condition}_train_target.pkl", "wb") as file:
    pickle.dump(train_target, file)

with open(f"{condition}_val_target.pkl", "wb") as file:
    pickle.dump(val_target, file)

with open(f"{condition}_test_target.pkl", "wb") as file:
    pickle.dump(test_target, file)

