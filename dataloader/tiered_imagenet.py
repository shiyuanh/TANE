import os
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pdb

def load_labels(file):
    try:
        with open(file, 'rb') as fo:
            data = pickle.load(fo)
        return data
    except:
        with open(file, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
        return data



class OpenTiered(Dataset):
    def __init__(self, args, partition='test', mode='episode', is_training=False, fix_seed=True):
        super(OpenTiered, self).__init__()
        self.mode = mode
        self.fix_seed = fix_seed
        self.n_ways = args.n_ways
        self.n_open_ways = args.n_open_ways
        self.n_shots = args.n_shots
        self.n_queries = args.n_queries
        self.n_episodes = args.n_test_runs if partition == 'test' else args.n_train_runs
        self.n_aug_support_samples = args.n_aug_support_samples
        self.partition = partition

        mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        normalize = transforms.Normalize(mean=mean, std=std)

        if is_training:
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(84, padding=8),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(84, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        
        self.test_transform = transforms.Compose([transforms.ToTensor(),normalize])

        self.vector_array = {}
        key_map = {'train':'base','test':'novel_test','val':'novel_val'}
        root_path = args.data_root
        for the_file in ['test','train', 'val']:
            file = 'few-shot-wordemb-{}.npz'.format(the_file)
            self.vector_array[key_map[the_file]] = np.load(os.path.join(root_path,file))['features']

        full_file = 'few-shot-{}.npz'.format(partition)
        self.imgs = np.load(os.path.join(root_path,full_file))['features']
        labels = np.load(os.path.join(root_path,full_file))['targets']


        self.imgs = [Image.fromarray(x) for x in self.imgs]
        min_label = min(labels)
        self.labels = [x - min_label for x in labels]
        print('Load {} Data of {} for tieredImageNet in Meta-Learning Stage'.format(len(self.imgs), partition))

        self.data = {}
        for idx in range(len(self.imgs)):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.imgs[idx])
        self.classes = list(self.data.keys())  

    
    def __getitem__(self, item):
        return self.get_episode(item)
    
    def get_episode(self, item):
        
        if self.fix_seed:
            np.random.seed(item)
        cls_sampled = np.random.choice(self.classes, self.n_ways, False)
        support_xs = []
        support_ys = []
        suppopen_xs = []
        suppopen_ys = []
        query_xs = []
        query_ys = []
        openset_xs = []
        openset_ys = []

        for idx, the_cls in enumerate(cls_sampled):
            imgs = self.data[the_cls]
            support_xs_ids_sampled = np.random.choice(range(len(imgs)), self.n_shots, False)
            support_xs.extend([imgs[the_id] for the_id in support_xs_ids_sampled])
            support_ys.extend([idx] * self.n_shots)
            query_xs_ids = np.setxor1d(np.arange(len(imgs)), support_xs_ids_sampled)
            query_xs_ids = np.random.choice(query_xs_ids, self.n_queries, False)
            query_xs.extend([imgs[the_id] for the_id in query_xs_ids])
            query_ys.extend([idx] * self.n_queries)
        
        cls_open_ids = np.setxor1d(np.arange(len(self.classes)), cls_sampled)
        cls_open_ids = np.random.choice(cls_open_ids, self.n_open_ways, False)
        for idx, the_cls in enumerate(cls_open_ids):
            imgs = self.data[the_cls]
            suppopen_xs_ids_sampled = np.random.choice(range(len(imgs)), self.n_shots, False)
            suppopen_xs.extend([imgs[the_id] for the_id in suppopen_xs_ids_sampled])
            suppopen_ys.extend([idx] * self.n_shots)
            openset_xs_ids = np.setxor1d(np.arange(len(imgs)), suppopen_xs_ids_sampled)
            openset_xs_ids_sampled = np.random.choice(range(len(imgs)), self.n_queries, False)
            openset_xs.extend([imgs[the_id] for the_id in openset_xs_ids_sampled])
            openset_ys.extend([the_cls] * self.n_queries)
            

        if self.partition == 'train':
            base_ids = np.setxor1d(np.arange(len(self.classes)), np.concatenate([cls_sampled,cls_open_ids]))
            assert len(set(base_ids).union(set(cls_open_ids)).union(set(cls_sampled))) == len(self.classes)
            base_ids = np.array(sorted(base_ids))
        
        if self.n_aug_support_samples > 1:
            support_xs_aug = [support_xs[i:i+self.n_shots]*self.n_aug_support_samples for i in range(0,len(support_xs),self.n_shots)]
            support_ys_aug = [support_ys[i:i+self.n_shots]*self.n_aug_support_samples for i in range(0,len(support_ys),self.n_shots)]
            support_xs,support_ys = support_xs_aug[0],support_ys_aug[0]
            for next_xs,next_ys in zip(support_xs_aug[1:],support_ys_aug[1:]):
                support_xs.extend(next_xs)
                support_ys.extend(next_ys)

            suppopen_xs_aug = [suppopen_xs[i:i+self.n_shots]*self.n_aug_support_samples for i in range(0,len(support_xs),self.n_shots)]
            suppopen_ys_aug = [suppopen_ys[i:i+self.n_shots]*self.n_aug_support_samples for i in range(0,len(support_ys),self.n_shots)]
            suppopen_xs,suppopen_ys = suppopen_xs_aug[0],suppopen_ys_aug[0]
            for next_xs,next_ys in zip(suppopen_xs_aug[1:],suppopen_ys_aug[1:]):
                suppopen_xs.extend(next_xs)
                suppopen_ys.extend(next_ys)

        support_xs = torch.stack(list(map(lambda x: self.train_transform(x), support_xs)))
        suppopen_xs =  torch.stack(list(map(lambda x: self.train_transform(x), suppopen_xs)))
        query_xs = torch.stack(list(map(lambda x: self.test_transform(x), query_xs)))
        openset_xs = torch.stack(list(map(lambda x: self.test_transform(x), openset_xs)))
        support_ys,query_ys,openset_ys = np.array(support_ys),np.array(query_ys),np.array(openset_ys)
        suppopen_ys = np.array(suppopen_ys)
        cls_sampled, cls_open_ids = np.array(cls_sampled), np.array(cls_open_ids)

        
        if self.partition == 'train':
            return support_xs, support_ys, query_xs, query_ys, suppopen_xs, suppopen_ys, openset_xs, openset_ys, cls_sampled, cls_open_ids, base_ids, 
        else:
            return support_xs, support_ys, query_xs, query_ys, suppopen_xs, suppopen_ys, openset_xs, openset_ys, cls_sampled, cls_open_ids
        
    def __len__(self):
        return self.n_episodes