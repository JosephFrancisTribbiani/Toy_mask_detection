import torch
import json
from torch.utils.data.dataset import Dataset
import os
from PIL import Image


class HasMaskDataset(Dataset):
  def __init__(self, 
               json_loc: str = 'labels.json',
               images_loc: str = 'images',
               transform=None):
    with open(json_loc) as f:
      self.data = json.loads(f.read())
    self.images_loc = images_loc
    self.transform = transform

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    item_data = self.data[idx]
    x_center, y_center, width, hight, has_object = 0, 0, 0, 0, 0

    if len(item_data['annotations'][0]['result']) > 0:
      has_object = 1
      value = item_data['annotations'][0]['result'][0]['value']
      x_center = (value['x'] + value['width'] / 2) / 100
      y_center = (value['y'] + value['height'] / 2) / 100
      width = value['width'] / 100
      hight = value['height'] / 100

    im_name = os.path.split(item_data['data']['image'])[-1]
    im_path = os.path.join(self.images_loc, im_name)
    image = Image.open(im_path)
    image = image.convert('RGB')

    if self.transform:
      image = self.transform(image)
    return image, torch.tensor([x_center, y_center, width, hight, has_object], dtype=torch.float32)