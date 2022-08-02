import os
import pickle
from PIL import Image
import numpy as np
import torch
import pdb
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class PreCIFAR(Dataset):
    def __init__(self, args, partition='train', is_training=True, is_contrast=False):
        super(PreCIFAR, self).__init__()
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        normalize = transforms.Normalize(mean=mean, std=std)
        self.is_contrast = is_training and is_contrast

        if is_training:
            if is_contrast:
                self.transform_left = transforms.Compose([
                    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    normalize
                ])
                self.transform_right = transforms.Compose([
                    transforms.RandomRotation(5),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),normalize])
        
        filename = '{}.pickle'.format(partition)         
        self.data = {}
        with open(os.path.join(args.data_root, filename), 'rb') as f:
            pack = pickle.load(f, encoding='latin1')
        self.imgs = pack['data']
        labels = pack['labels']

        cur_class = 0
        label2label = {}
        for _, label in enumerate(labels):
            if label not in label2label:
                label2label[label] = cur_class
                cur_class += 1
        new_labels = []
        for idx, label in enumerate(labels):
            new_labels.append(label2label[label])
        self.labels = new_labels

        self.imgs = [Image.fromarray(x) for x in self.imgs]
        print('Load {} Data of {} for {} in Pretraining Stage'.format(len(self.imgs), partition, args.dataset))

    def __getitem__(self, item):
        if self.is_contrast:
            left,right = self.transform_left(self.imgs[item]),self.transform_right(self.imgs[item])
            target = self.labels[item]
            return left, right, target, item
        else:
            img = self.transform(self.imgs[item])
            target = self.labels[item]
            return img, target, item
        
    def __len__(self):
        return len(self.labels)

class MetaCIFAR(Dataset):
    def __init__(self, args, n_shots, partition='test', is_training=False, fix_seed=True):
        super(MetaCIFAR, self).__init__()
        self.fix_seed = fix_seed
        self.n_ways = args.n_ways
        self.n_shots = n_shots
        self.n_queries = args.n_queries
        self.n_episodes = args.n_episodes
        self.n_aug_support_samples = args.n_aug_support_samples

        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        normalize = transforms.Normalize(mean=mean, std=std)

        if is_training:
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        
        self.test_transform = transforms.Compose([transforms.ToTensor(),normalize])

        filename = '{}.pickle'.format(partition)         
        self.data = {}
        with open(os.path.join(args.data_root, filename), 'rb') as f:
            pack = pickle.load(f, encoding='latin1')
        self.imgs = pack['data']
        labels = pack['labels']

        cur_class = 0
        label2label = {}
        for _, label in enumerate(labels):
            if label not in label2label:
                label2label[label] = cur_class
                cur_class += 1
        new_labels = []
        for idx, label in enumerate(labels):
            new_labels.append(label2label[label])
        self.labels = new_labels

        self.imgs = [Image.fromarray(x) for x in self.imgs]
        print('Load {} Data of {} for {} in Meta-Learning Stage'.format(len(self.imgs), partition, args.dataset))

        self.data = {}
        for idx in range(len(self.imgs)):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.imgs[idx])
        self.classes = list(self.data.keys())
    
    def __getitem__(self, item):
        if self.fix_seed:
            np.random.seed(item)
        cls_sampled = np.random.choice(self.classes, self.n_ways, False)
        support_xs = []
        support_ys = []
        query_xs = []
        query_ys = []
        for idx, the_cls in enumerate(cls_sampled):
            imgs = self.data[the_cls]
            support_xs_ids_sampled = np.random.choice(range(len(imgs)), self.n_shots, False)
            support_xs.extend([imgs[the_id] for the_id in support_xs_ids_sampled])
            support_ys.extend([idx] * self.n_shots)
            query_xs_ids = np.setxor1d(np.arange(len(imgs)), support_xs_ids_sampled)
            query_xs_ids = np.random.choice(query_xs_ids, self.n_queries, False)
            query_xs.extend([imgs[the_id] for the_id in query_xs_ids])
            query_ys.extend([idx] * query_xs_ids.shape[0])

        if self.n_aug_support_samples > 1:
            support_xs = support_xs * self.n_aug_support_samples 
            support_ys = support_ys * self.n_aug_support_samples 
        
        support_xs = torch.stack(list(map(lambda x: self.train_transform(x), support_xs)))
        query_xs = torch.stack(list(map(lambda x: self.test_transform(x), query_xs)))
        support_ys,query_ys = np.array(support_ys),np.array(query_ys)
      
        return support_xs, support_ys, query_xs, query_ys      
        
    def __len__(self):
        return self.n_episodes
