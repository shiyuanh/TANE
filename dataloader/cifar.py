import os
import pickle
from PIL import Image
import numpy as np
import torch
import pdb
from torch.utils.data import Dataset
import torchvision.transforms as transforms

INCLUDE_BASE=False

class OpenCIFAR(Dataset):    
    def __init__(self, args, partition='test', mode='episode', is_training=False, fix_seed=True, held_out=False):
        super(OpenCIFAR, self).__init__()
        self.fix_seed = fix_seed
        self.n_ways = args.n_ways
        self.n_shots = args.n_shots
        self.n_open_ways = args.n_open_ways
        self.n_queries = args.n_queries
        self.n_episodes = args.n_train_runs if partition=='train' else args.n_test_runs
        self.n_aug_support_samples = 1 if partition == 'train' else args.n_aug_support_samples
        self.partition = partition
        self.held_out = held_out

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

        with open(os.path.join(args.data_root,'category_vector.pickle'), 'rb') as f:
            pack = pickle.load(f, encoding='latin1')
            vectors = pack['vector']

        self.vector_array = {'base':vectors['train'],'nove_val':vectors['val'],'novel_test':vectors['test']}
        
        self.test_transform = transforms.Compose([transforms.ToTensor(),normalize])
        self.init_episode(args.data_root,partition)
    
    def __getitem__(self, item):
        return self.get_episode(item)

    def init_episode(self, data_root, partition):

        filename = '{}.pickle'.format(partition)         
        self.data = {}
        with open(os.path.join(data_root, filename), 'rb') as f:
            pack = pickle.load(f, encoding='latin1')
        self.imgs = pack['data']
        labels = pack['labels']

        label2label = {}
        unique_labels = sorted(list(set(labels)))
        for cur_class, label in enumerate(unique_labels):
            label2label[label] = cur_class
        new_labels = []
        for idx, label in enumerate(labels):
            new_labels.append(label2label[label])
        self.labels = new_labels

        self.imgs = [Image.fromarray(x) for x in self.imgs]
        print('Load {} Data of {} in Meta-Learning Stage'.format(len(self.imgs), partition))

        self.data = {}
        for idx in range(len(self.imgs)):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.imgs[idx])
        self.classes = list(self.data.keys())

        if self.held_out:
            for key in self.data:
                self.data[key] = self.data[key][:-100]

        if self.partition == 'test':
            if INCLUDE_BASE:
                filename = '{}.pickle'.format('train') 
                with open(os.path.join(data_root, filename), 'rb') as f:
                    pack = pickle.load(f, encoding='latin1')
                self.base_imgs = pack['data'].astype('uint8')
                labels = pack['labels']
                self.base_imgs = [Image.fromarray(x) for x in self.base_imgs]
                min_label = min(labels)
                self.base_labels = [x - min_label for x in labels]
                self.base_data = {}
                for idx in range(len(self.base_imgs)):
                    if self.base_labels[idx] not in self.base_data:
                        self.base_data[self.base_labels[idx]] = []
                    self.base_data[self.base_labels[idx]].append(self.base_imgs[idx])
                for key in self.base_data:
                    self.base_data[key] = self.base_data[key][-100:]
                self.base_classes = list(self.base_data.keys())
                
                print('Load {} Base Data of {} for miniImagenet in Meta-Learning Stage'.format(len(self.base_imgs), partition))

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
        manyshot_xs = []
        manyshot_ys = []

        # Close set preparation
        for idx, the_cls in enumerate(cls_sampled):
            imgs = self.data[the_cls]
            support_xs_ids_sampled = np.random.choice(range(len(imgs)), self.n_shots, False)
            support_xs.extend([imgs[the_id] for the_id in support_xs_ids_sampled])
            support_ys.extend([idx] * self.n_shots)
            query_xs_ids = np.setxor1d(np.arange(len(imgs)), support_xs_ids_sampled)
            query_xs_ids = np.random.choice(query_xs_ids, self.n_queries, False)
            query_xs.extend([imgs[the_id] for the_id in query_xs_ids])
            query_ys.extend([idx] * self.n_queries)
        
        # Open set preparation
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
        else:
            if INCLUDE_BASE:
                base_ids = sorted(self.base_classes)
                assert len(base_ids) > self.n_ways
                base_cls_sampled = list(np.random.choice(base_ids,  self.n_ways, False))
                base_cls_sampled.sort()
                for idx, the_cls in enumerate(base_cls_sampled):
                    imgs = self.base_data[the_cls]
                    manyshot_xs_ids_sampled = np.random.choice(range(len(imgs)), self.n_queries, False)
                    manyshot_xs.extend([imgs[the_id] for the_id in manyshot_xs_ids_sampled])
                    manyshot_ys.extend([idx] * self.n_queries)

        
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
            return support_xs, support_ys, query_xs, query_ys, suppopen_xs, suppopen_ys, openset_xs, openset_ys, cls_sampled, cls_open_ids, base_ids
        else:
            if INCLUDE_BASE:
                manyshot_xs = torch.stack(list(map(lambda x: self.test_transform(x), manyshot_xs)))
                openset_xs = torch.cat([openset_xs, manyshot_xs])
                openset_ys = torch.ones(len(openset_xs))
            return support_xs, support_ys, query_xs, query_ys, suppopen_xs, suppopen_ys, openset_xs, openset_ys, cls_sampled, cls_open_ids
    
    def __len__(self):
        return self.n_episodes