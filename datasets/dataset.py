import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class VOCSegmentDataset(Dataset):

    VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

    VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

    height_gen = 350
    width_gen = 500
    
    def __init__(self, dataset_dir, crop_size, is_train = True):
        self.dataset_dir = dataset_dir
        self.is_train = is_train
        self.crop_size = crop_size

        features, labels = self.read_dataset()

        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        
        # We have to normalize the image
        self.features = [self.normalize_image(img) for img in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = self.voc_colormap2label()
    
    
    def read_dataset(self):
        txt_fname = os.path.join(self.dataset_dir, 'ImageSets', 
                                 'Segmentation', 'train.txt' if self.is_train else 'val.txt')
        mode = torchvision.io.image.ImageReadMode.RGB

        with open(txt_fname, 'r') as f:
            images = f.read().split()

        features, labels = [], []

        for i, fname in enumerate(images):
            image_path = os.path.join(self.dataset_dir, 'JPEGImages', f'{fname}.jpg')
            segment_path = os.path.join(self.dataset_dir, 'SegmentationClass',
                                        f'{fname}.png')

            features.append(torchvision.io.read_image(image_path))
            labels.append(torchvision.io.read_image(segment_path, mode))
        return features, labels

    def normalize_image(self, image):
        return self.transform(image.float() / 255)

    def filter(self, imgs):
        return [img for img in imgs \
                if (img.shape[1] >= self.crop_size[0] \
                   and img.shape[2] >= self.crop_size[1])]
        
    def voc_colormap2label(self):
        colormap2label = torch.zeros(256 ** 3, dtype = torch.long)
        for i, colormap in enumerate(self.VOC_COLORMAP):
            index_color = (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]
            colormap2label[index_color] = i
        return colormap2label


    def voc_label_indices(self, colormap, colormap2label):
        colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
        idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 
                + colormap[:, :, 2])
        return colormap2label[idx]

    def voc_rand_crop(self, feature, label, height, width, resize_coef = 1.05):
    
        height_resize = int(height * resize_coef)
        width_resize = int(height_resize * self.width_gen / self.height_gen)
    
        # print(f"Resize shape: ({height_resize}, {width_resize})")
        
        center_crop = torchvision.transforms.Compose([
                                torchvision.transforms.Resize((height_resize, width_resize),
                                                             antialias=False),
                                torchvision.transforms.CenterCrop((height, width))
        ])
        
        feature = center_crop(feature)
        label = center_crop(label)
        return feature, label
        
    def draw_image_label(self, feature, label):
        # We decide to draw in 1 column, 2 row
        fig, axes = plt.subplots(1, 2, figsize = (10, 10))

        axes[0].imshow(feature.permute(1, 2,0))
        axes[0].set_title('Image')
        axes[0].axis('off')

        axes[1].imshow(label)
        axes[1].set_title('Label')
        axes[1].axis('off')
    
        
    def __getitem__(self, index):
        feature, label = self.voc_rand_crop(self.features[index], self.labels[index], 
                                       *self.crop_size, resize_coef = 1.05)
        return (feature, self.voc_label_indices(label, self.colormap2label))

        

    def __len__(self):
        return len(self.features)
                                                            

