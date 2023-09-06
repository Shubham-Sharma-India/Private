# Private
YOLOv8x_weed_crop_detection

```python
# In this notebook, I will show how to solve the problem of weed detection for the agricultural sector using the SOTA model YOLOv8x
import pandas as pd
import numpy as np
from ultralytics import YOLO
import torch

import shutil
import os

import random

from PIL import Image
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
```
```python
#Data preparation
#As training data, I took 2 datasets WeedCrop Image Dataset 2822 images of varying quality and LincoInBeet 4402 high-quality images 1920 x 1080 pixels in size.
#In total, we get 7224 images; we divide them in the ratio 5558:676:990, respectively, train:val:test.
#But first, you need to collect and structure datasets.
source_folder = "/kaggle/input/weedcrop-image-dataset/WeedCrop.v1i.yolov5pytorch"

# Папка test
test_source_images = os.path.join(source_folder, "test/images")
test_source_labels = os.path.join(source_folder, "test/labels")

test_destination_folder = "/kaggle/working/test"
os.makedirs(test_destination_folder, exist_ok=True)

# Перемещение изображений
shutil.copytree(test_source_images, os.path.join(test_destination_folder, "images"))

# Перемещение меток
shutil.copytree(test_source_labels, os.path.join(test_destination_folder, "labels"))


# Папка train
train_source_images = os.path.join(source_folder, "train/images")
train_source_labels = os.path.join(source_folder, "train/labels")

train_destination_folder = "/kaggle/working/train"
os.makedirs(train_destination_folder, exist_ok=True)

# Перемещение изображений
shutil.copytree(train_source_images, os.path.join(train_destination_folder, "images"))

# Перемещение меток
shutil.copytree(train_source_labels, os.path.join(train_destination_folder, "labels"))


# Папка valid
valid_source_images = os.path.join(source_folder, "valid/images")
valid_source_labels = os.path.join(source_folder, "valid/labels")

valid_destination_folder = "/kaggle/working/valid"
os.makedirs(valid_destination_folder, exist_ok=True)

# Перемещение изображений
shutil.copytree(valid_source_images, os.path.join(valid_destination_folder, "images"))

# Перемещение меток
shutil.copytree(valid_source_labels, os.path.join(valid_destination_folder, "labels"))
```
```python
def move_files(file_path, source_folder, destination_folder):
    # Чтение списка названий файлов
    with open(file_path, "r") as file:
        file_names = [os.path.basename(line.strip()) for line in file.readlines()]

    # Папки для изображений и меток
    images_folder = os.path.join(destination_folder, "images")
    labels_folder = os.path.join(destination_folder, "labels")
    
    # Перемещение файлов
    for file_name in file_names:
        image_file = os.path.join(source_folder, file_name)
        label_file = os.path.join(source_folder, file_name.replace(".png", ".txt"))
        if os.path.isfile(image_file):
            shutil.copy(image_file, images_folder)
        if os.path.isfile(label_file):
            shutil.copy(label_file, labels_folder)
```
```python
train_file = "/kaggle/input/amiran/all_fields_lincolnbeet/all_fields_lincolnbeet_train_.txt"
valid_file = "/kaggle/input/amiran/all_fields_lincolnbeet/all_fields_lincolnbeet_val_.txt"
test_file = "/kaggle/input/amiran/all_fields_lincolnbeet/all_fields_lincolnbeet_test_.txt"

source_folder = "/kaggle/input/amiran/all_fields_lincolnbeet/all"
train_destination = "/kaggle/working/train"
valid_destination = "/kaggle/working/valid"
test_destination = "/kaggle/working/test"

move_files(train_file, source_folder, train_destination)
move_files(valid_file, source_folder, valid_destination)
move_files(test_file, source_folder, test_destination)
```
```python
# Создадим yaml для YOLO
import  yaml

# Data structure
dataset = {
'train': '/kaggle/working/train',
'val': '/kaggle/working/valid',
'test': '/kaggle/working/test',
'nc': 2,
'names': ['crop', 'weed']
}

# save to YAML-file
with open('/kaggle/working/dataset.yaml', 'w') as file:
    yaml.dump(dataset, file)
```
```python
ls /kaggle/working
```
```python
# Папки с изображениями и метками
images_folder = "/kaggle/working/train/images"
labels_folder = "/kaggle/working/train/labels"

# Загрузка списка файлов изображений
image_files = os.listdir(images_folder)

# Выбор случайных изображений
random.shuffle(image_files)
random_image_files = image_files[:6]

# Отображение случайных изображений с метками
num_images = len(random_image_files)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i in range(num_images):
    # Загрузка изображения
    image_file = os.path.join(images_folder, random_image_files[i])
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Загрузка меток из файла
    label_file = os.path.join(labels_folder, os.path.splitext(random_image_files[i])[0] + ".txt")
    with open(label_file, "r") as file:
        labels = file.readlines()

    # Отображение изображения с метками
    for label in labels:
        class_id, x, y, width, height = map(float, label.strip().split())
        x = int(x * image.shape[1])
        y = int(y * image.shape[0])
        width = int(width * image.shape[1])
        height = int(height * image.shape[0])
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # Отображение изображения
    axes[i].imshow(image)
    axes[i].axis("off")

plt.tight_layout()
plt.show()
```
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device
```
```python
model = YOLO('yolov8x.pt')
```
```python
model.train(data='/kaggle/working/dataset.yaml ', epochs=50, imgsz=640,
            optimizer = 'AdamW', lr0 = 1e-3, 
            project = 'TG_YOLOv8x', name='Didi',
            batch=16, device=device, seed=69)
```
```python
#Train metrics and losses
#Yolo is a very convenient framework that saves logs and also ready-made charts of metrics and losses. We will not invent a bicycle and use a ready-made solution.
df = pd.read_csv('/kaggle/working/TG_YOLOv8x/Didi/results.csv')
df.columns
```
```python
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
fig.tight_layout()

# train/box_loss
axes[0, 0].plot(df['                  epoch'], df['         train/box_loss'], label='         train/box_loss')
axes[0, 0].set_title('Train Box Loss')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()

# val/box_loss
axes[0, 1].plot(df['                  epoch'], df['           val/box_loss'], label='           val/box_loss')
axes[0, 1].set_title('Validation Box Loss')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()

# train/cls_loss
axes[1, 0].plot(df['                  epoch'], df['         train/cls_loss'], label='         train/cls_loss')
axes[1, 0].set_title('Train Class Loss')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].legend()

# val/cls_loss
axes[1, 1].plot(df['                  epoch'], df['           val/cls_loss'], label='           val/cls_loss')
axes[1, 1].set_title('Validation Class Loss')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].legend()

# train/dfl_loss
axes[2, 0].plot(df['                  epoch'], df['         train/dfl_loss'], label='         train/dfl_loss')
axes[2, 0].set_title('Train Distribution Focal loss')
axes[2, 0].set_xlabel('Epoch')
axes[2, 0].set_ylabel('Loss')
axes[2, 0].legend()

# val/dfl_loss
axes[2, 1].plot(df['                  epoch'], df['           val/dfl_loss'], label='           val/dfl_loss')
axes[2, 1].set_title('Validation Distribution Focal loss')
axes[2, 1].set_xlabel('Epoch')
axes[2, 1].set_ylabel('Loss')
axes[2, 1].legend()

plt.show()
```
```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.tight_layout()

# metrics/precision(B)
axes[0, 0].plot(df['                  epoch'], df['   metrics/precision(B)'], label='   metrics/precision(B)')
axes[0, 0].set_title('Precision')
axes[0, 0].set_ylabel('Precision')
axes[0, 0].legend()

# metrics/recall(B)
axes[0, 1].plot(df['                  epoch'], df['      metrics/recall(B)'], label='      metrics/recall(B)')
axes[0, 1].set_title('Recall')
axes[0, 1].set_ylabel('Recall')
axes[0, 1].legend()

# График для metrics/mAP50(B)
axes[1, 0].plot(df['                  epoch'], df['       metrics/mAP50(B)'], label='       metrics/mAP50(B)')
axes[1, 0].set_title('mAP@0.5')
axes[1, 0].set_ylabel('mAP@0.5')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].legend()

# metrics/mAP50-95(B)
axes[1, 1].plot(df['                  epoch'], df['    metrics/mAP50-95(B)'], label='    metrics/mAP50-95(B)')
axes[1, 1].set_title('mAP@0.5:0.95')
axes[1, 1].set_ylabel('mAP@0.5:0.95')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].legend()

plt.show()
```
```python
# F1_curve.png
f1_curve = Image.open("/kaggle/working/TG_YOLOv8x/Didi/F1_curve.png")
plt.figure(figsize=(10, 10))
plt.imshow(f1_curve)
plt.title("F1 Curve")
plt.axis("off")
plt.show()

# PR_curve.png
pr_curve = Image.open("/kaggle/working/TG_YOLOv8x/Didi/PR_curve.png")
plt.figure(figsize=(10, 10))
plt.imshow(pr_curve)
plt.title("Precision-Recall Curve")
plt.axis("off")
plt.show()

# P_curve.png
p_curve = Image.open("/kaggle/working/TG_YOLOv8x/Didi/P_curve.png")
plt.figure(figsize=(10, 10))
plt.imshow(p_curve)
plt.title("Precision Curve")
plt.axis("off")
plt.show()

# R_curve.png
r_curve = Image.open("/kaggle/working/TG_YOLOv8x/Didi/R_curve.png")
plt.figure(figsize=(10, 10))
plt.imshow(r_curve)
plt.title("Recall Curve")
plt.axis("off")
plt.show()
```
```python
# confusion matrix
confusion_matrix = Image.open("/kaggle/working/TG_YOLOv8x/Didi/confusion_matrix.png")
plt.figure(figsize=(8, 6))
plt.imshow(confusion_matrix)
plt.title("Confusion Matrix")
plt.axis("off")
plt.show()
#The largest number of errors in our model were related to misclassifying objects as "background" when they were actually weeds - 903 such errors. Conversely, there were 955 errors when the model classified weeds as "background".
#In other cases, the number of errors was insignificant, in the region of 100-200 values.
```
```python
#Let's see how the model detects on the test set
res = model('/kaggle/working/test/images/bbro_bbro_14_05_2021_v_0_18.png')
detect_img = res[0].plot()
detect_img = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Отображение первого изображения
axes[0].imshow(plt.imread('/kaggle/working/test/images/bbro_bbro_14_05_2021_v_0_18.png'))
axes[0].axis('off')

# Отображение результатов модели
axes[1].imshow(detect_img)
axes[1].axis('off')

plt.show();
```
```python
model = YOLO('TG_YOLOv8x/Didi/weights/best.pt ')
```
```python
metrics = model.val(split='test', conf=0.25, device=device) # conf - это порог достоверности объекта для обнаружения
```
```python
metrics
ax = sns.barplot(x=['mAP50-95', 'mAP50', 'mAP75'], y=[metrics.box.map, metrics.box.map50, metrics.box.map75])


ax.set_title('YOLO Evaluation Metrics')
ax.set_xlabel('Metric')
ax.set_ylabel('Value')


fig = plt.gcf()
fig.set_size_inches(8, 6)

for p in ax.patches:
    ax.annotate('{:.3f}'.format(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
    

plt.show()
#Thus, the model demonstrates higher accuracy in object detection at lower confidence thresholds, and at higher thresholds its accuracy decreases. This may be due to the fact that at high confidence thresholds the model becomes more conservative and misses some objects to reduce the likelihood of false positives.
```
```python
# Извлечение значений Precision, Recall и F1
precision = metrics.results_dict['metrics/precision(B)']
recall = metrics.results_dict['metrics/recall(B)']
f1 = (2 * precision * recall) / (precision + recall)  # Вычисление F1


metrics = ['Precision', 'Recall', 'F1']
values = [precision, recall, f1]

# Создание графика с использованием sns.barplot
ax = sns.barplot(x=metrics, y=values, palette='viridis')

ax.set_title('Precision, Recall, and F1 Scores')
ax.set_xlabel('Metric')
ax.set_ylabel('Value')

for p in ax.patches:
    ax.annotate('{:.3f}'.format(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')

plt.show()
#As can be seen from the metrics, the 'Precision' model is very good, it recognizes both classes well and finds them with approximately the same indicator.
#Let's take a look at 10 random images
```
```python
# Получение случайных изображений
image_paths = random.sample(os.listdir(images_folder), 10)

# Создание фигуры с подокнами и увеличенным размером фотографий
fig, axes = plt.subplots(2, 5, figsize=(20, 12))
fig.tight_layout()

# Итерация по каждому подокну
for i, ax in enumerate(axes.flat):
    image_path = os.path.join(images_folder, image_paths[i])
    image = Image.open(image_path)
    res = model(image, verbose=False)
    detect_img = res[0].plot()
    detect_img = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)
    
    ax.imshow(detect_img)
    ax.axis('off')
    
plt.show()
```
```python
#Conclusion
#In this notebook, we trained the YOLOv8x model on pooled data, examined metrics and training loss, and analyzed the model's output on multiple images and detection performance on test data.
#To achieve higher model performance and successfully apply it in business processes, it is recommended to conduct more thorough research. Here are some steps that can be taken:

#Collect more data: More diverse data will help the model generalize and recognize objects better. You can collect new data or find available datasets to complement the current training set.
#Tune hyperparameters: Choosing optimal hyperparameters for a model can significantly impact its performance. You can experiment with different hyperparameter values ​​such as anchor sizes, confidence thresholds, and others to achieve better results.

#Conduct additional experiments: Experimenting with different model architectures, variations of YOLOv8 or other object detection models, may lead to better results. etc.
#In general, more accurate research (specifically more labeled data with weed tags and boxes), experimentation, and work to improve the model will achieve better performance and increase its applicability in real-world business scenarios. Of course, everything depends on resources, both computational and monetary.
```
