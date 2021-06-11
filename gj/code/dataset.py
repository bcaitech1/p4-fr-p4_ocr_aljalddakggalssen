import csv
import os
import random
import math
from collections import defaultdict
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import (
    Dataset,
    DataLoader,
    Sampler,
    Subset,
)

import cv2

from tqdm.auto import trange, tqdm
from torchvision import transforms
from copy import deepcopy

START = "<SOS>"
END = "<EOS>"
PAD = "<PAD>"
SPECIAL_TOKENS = [START, END, PAD]

def claheCVT(own_img) :
    lab = cv2.cvtColor(own_img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    cla_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cla_img

# Rather ignorant way to encode the truth, but at least it works.
def encode_truth(truth, token_to_id, is_reverse=False):

    truth_tokens = truth.split()
    for token in truth_tokens:
        if token not in token_to_id:
            raise Exception("Truth contains unknown token")
    truth_tokens = [token_to_id[x] for x in truth_tokens]
    if '' in truth_tokens: truth_tokens.remove('')
    if is_reverse:
        truth_tokens.reverse()
    return truth_tokens


def load_vocab(tokens_paths):
    tokens = []
    tokens.extend(SPECIAL_TOKENS)
    for tokens_file in tokens_paths:
        with open(tokens_file, "r") as fd:
            reader = fd.read()
            for token in reader.split("\n"):
                if token not in tokens:
                    tokens.append(token)
    token_to_id = {tok: i for i, tok in enumerate(tokens)}
    id_to_token = {i: tok for i, tok in enumerate(tokens)}
    return token_to_id, id_to_token


def split_gt(groundtruth, proportion=1.0, test_percent=None):
    root = os.path.join(os.path.dirname(groundtruth), "images")
    with open(groundtruth, "r") as fd:
        data=[]
        for line in fd:
            data.append(line.strip().split("\t"))
        random.shuffle(data)
        dataset_len = round(len(data) * proportion)
        data = data[:dataset_len]
        data = [[os.path.join(root, x[0]), x[1]] for x in data]
    
    if test_percent:
        test_len = round(len(data) * test_percent)
        return data[test_len:], data[:test_len]
    else:
        return data

def load_levels(level_paths):
    levels = {}
    for level_file in level_paths:
        with open(level_file, 'r') as fd:
            tmp = [line.strip().split('\t') for line in fd.readlines()]
            tmp = [(x, int(y)) for x, y in tmp]
            for path, level in tmp:
                levels[path] = level
    return levels

def load_sources(source_paths):
    sources = {}
    for source_file in source_paths:
        with open(source_file, 'r') as fd:
            tmp = [line.strip().split('\t') for line in fd.readlines()]
            tmp = [(x, int(y)) for x, y in tmp]
            for path, source in tmp:
                sources[path] = source
    return sources

class SizeBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, is_random=True):
        super().__init__(data_source=data_source)
        self.batch_size = batch_size
        self.is_random = is_random

        self.bucket_dict = defaultdict(list)

        for i in trange(len(data_source)):
            h, w = data_source.get_shape(i)
            self.bucket_dict[w].append(i)

        for k in self.bucket_dict:
            self.bucket_dict[k] = np.array(self.bucket_dict[k])
        
        self.len = 0
        self.total_cnt = {}
        for k, v in self.bucket_dict.items():
            cur_len = (len(v) + self.batch_size - 1) // self.batch_size
            self.len += cur_len
            self.total_cnt[k] = cur_len

    def __iter__(self):
        if self.is_random:
            for k, v in self.bucket_dict.items():
                np.random.shuffle(v)

        ordered_keys = []
        self.keys2idx = {}
        self.current_cnt = {}
        idx = 0
        for k in self.bucket_dict:
            self.current_cnt[k] = 0
            ordered_keys.append(k)
            self.keys2idx[k] = idx
            idx += 1

        self.ordered_keys = np.array(ordered_keys)
        self.keys_prob = np.ones_like(self.ordered_keys)

        self.left = len(self.bucket_dict)

        return self

    def __next__(self):
        if self.left > 0:
            cur_bucket = np.random.choice(self.ordered_keys, p=self.keys_prob/self.keys_prob.sum())
            idx = self.current_cnt[cur_bucket]
            cur_idxs = self.bucket_dict[cur_bucket][self.batch_size * idx : self.batch_size * (idx + 1)]

            self.current_cnt[cur_bucket] += 1
            if self.current_cnt[cur_bucket] == self.total_cnt[cur_bucket]:
                self.keys_prob[self.keys2idx[cur_bucket]] = 0
                self.left -= 1

            return cur_idxs
        else:
            raise StopIteration

    def __len__(self):
        return self.len

def collate_batch(data):
    max_len = max([len(d["truth"]["encoded"]) for d in data])
    # Padding with -1, will later be replaced with the PAD token
    padded_encoded = [
        d["truth"]["encoded"] + (max_len - len(d["truth"]["encoded"])) * [-1]
        for d in data
    ]
    return {
        "path": [d["path"] for d in data],
        "image": torch.stack([d["image"] for d in data], dim=0),
        "truth": {
            "text": [d["truth"]["text"] for d in data],
            "encoded": torch.tensor(padded_encoded)
        },
        'level': torch.tensor([d['level'] for d in data], dtype=torch.long),
        'source': torch.tensor([d['source'] for d in data], dtype=torch.long),
    }

def collate_eval_batch(data):
    max_len = max([len(d["truth"]["encoded"]) for d in data])
    # Padding with -1, will later be replaced with the PAD token
    padded_encoded = [
        d["truth"]["encoded"] + (max_len - len(d["truth"]["encoded"])) * [-1]
        for d in data
    ]
    return {
        "path": [d["path"] for d in data],
        "file_path":[d["file_path"] for d in data],
        "image": torch.stack([d["image"] for d in data], dim=0),
        "truth": {
            "text": [d["truth"]["text"] for d in data],
            "encoded": torch.tensor(padded_encoded)
        },
    }

class FlexibleInputSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)

        self.token_to_id = self.dataset.token_to_id
        self.id_to_token = self.dataset.id_to_token

    def get_shape(self, idx):
        return self.dataset.get_shape(self.indices[idx])

class LoadDataset(Dataset):
    """Load Dataset"""

    def __init__(
        self,
        groundtruth,
        tokens_file,
        levels,
        sources,
        crop=False,
        transform=None,
        rgb=3,
        max_resolution=128*128,
        is_flexible=False,
        is_reverse=False,
        use_curr=False,
        use_flip_channel=False,
        apply_clihe=False,
    ):
        """
        Args:
            groundtruth (string): Path to ground truth TXT/TSV file
            tokens_file (string): Path to tokens TXT file
            ext (string): Extension of the input files
            crop (bool, optional): Crop images to their bounding boxes [Default: False]
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(LoadDataset, self).__init__()
        self.use_curr = use_curr
        self.apply_clihe = apply_clihe
        self.crop = crop
        self.transform = transform
        self.rgb = rgb
        self.use_flip_channel = use_flip_channel
        self.token_to_id, self.id_to_token = load_vocab(tokens_file)
        self.data = [
            {
                "path": p,
                "truth": {
                    "text": truth,
                    "encoded": [
                        self.token_to_id[START],
                        *encode_truth(truth, self.token_to_id,
                            is_reverse=is_reverse),
                        self.token_to_id[END],
                    ],
                },
            }
            for p, truth in groundtruth
        ]


        for datum in self.data:
            file_path = datum['path'].split('/')[-1]
            source = sources.get(file_path, -100) # -100 crossentory 무시 index
            level = levels.get(file_path, -99) - 1 # -100 모름
            datum['source'] = source
            datum['level'] = level

        if self.use_curr:
            self.level_idxs = defaultdict(list)
            for idx, datum in enumerate(self.data):
                self.level_idxs[datum['level']].append(idx)

        self.is_flexible = is_flexible
        if self.is_flexible:
            self.shape_cache = np.zeros((len(self), 2), dtype=np.int)
            self.max_resolution = max_resolution

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        image = Image.open(item["path"])

        if self.apply_clihe:
            image = np.array(image)
            image = claheCVT(image)
            image = transforms.ToPILImage()(image)

        if self.rgb == 3:
            image = image.convert("RGB")
        elif self.rgb == 1:
            image = image.convert("L")
        else:
            raise NotImplementedError

        if self.crop:
            # Image needs to be inverted because the bounding box cuts off black pixels,
            # not white ones.
            bounding_box = ImageOps.invert(image).getbbox()
            image = image.crop(bounding_box)

        if self.transform:
            image = self.transform(image)

        if self.is_flexible:
            image = transforms.Resize(self.get_shape(i))(image)

        if self.use_flip_channel:
            tmp = image.flip(-1)
            image = torch.cat([image, tmp], dim=0)

        return {
            "path": item["path"],
            "truth": item["truth"],
            "image": image,
            'source': item['source'],
            'level': item['level'],
        }

    def get_shape(self, i):
        h, w = self.shape_cache[i]
        if h == 0 and w == 0:
            item = self.data[i]
            image = Image.open(item["path"])
            rw, rh = image.size

            T = self.max_resolution
            div = rw * rh / T
            w = round(rw/math.sqrt(div))
            h = round(rh/math.sqrt(div))
            w = round(w / 32) * 32
            h = T // w
            # h = (T // w) // 32 * 32

            self.shape_cache[i][0] = h
            self.shape_cache[i][1] = w
        return h, w

    def get_level_dataset(self, level):
        """
            일단은 완전 분리, 그전꺼 사용하고 싶으면 그런 옵션 추가하게
        """
        if not self.use_curr:
            return

        return FlexibleInputSubset(self, self.level_idxs[level])

    def get_lower_level_dataset(self, level):
        if not self.use_curr:
            return

        idxs = []
        for i in range(level+1):
            idxs.extend(self.level_idxs[i])

        return FlexibleInputSubset(self, idxs)


class LoadEvalDataset(Dataset):
    """Load Dataset"""

    def __init__(
        self,
        groundtruth,
        token_to_id,
        id_to_token,
        crop=False,
        transform=None,
        rgb=3,
        max_resolution=128*128,
        is_flexible=False,
        is_reverse=False,
        use_flip_channel=False,
        apply_clihe=False,
    ):
        """
        Args:
            groundtruth (string): Path to ground truth TXT/TSV file
            tokens_file (string): Path to tokens TXT file
            ext (string): Extension of the input files
            crop (bool, optional): Crop images to their bounding boxes [Default: False]
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(LoadEvalDataset, self).__init__()
        self.crop = crop
        self.rgb = rgb
        self.use_flip_channel = use_flip_channel
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.transform = transform
        self.apply_clihe = apply_clihe
        self.data = [
            {
                "path": p,
                "file_path":p1,
                "truth": {
                    "text": truth,
                    "encoded": [
                        self.token_to_id[START],
                        *encode_truth(truth, self.token_to_id,
                            is_reverse=is_reverse),
                        self.token_to_id[END],
                    ],
                },
            }
            for p, p1,truth in groundtruth
        ]

        self.is_flexible = is_flexible
        if self.is_flexible:
            self.shape_cache = np.zeros((len(self), 2), dtype=np.int)
            self.max_resolution = max_resolution

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        image = Image.open(item["path"])

        if self.apply_clihe:
            image = np.arrary(image)
            image = claheCVT(image)
            image = transforms.ToPILImage()(image)

        if self.rgb == 3:
            image = image.convert("RGB")
        elif self.rgb == 1:
            image = image.convert("L")
        else:
            raise NotImplementedError

        if self.crop:
            # Image needs to be inverted because the bounding box cuts off black pixels,
            # not white ones.
            bounding_box = ImageOps.invert(image).getbbox()
            image = image.crop(bounding_box)

        if self.transform:
            image = self.transform(image)

        if self.is_flexible:
            image = transforms.Resize(self.get_shape(i))(image)

        if self.use_flip_channel:
            tmp = image.flip(-1)
            image = torch.cat([image, tmp], dim=0)

        return {"path": item["path"], "file_path":item["file_path"],"truth": item["truth"], "image": image}

    def get_shape(self, i):
        h, w = self.shape_cache[i]
        if h == 0 and w == 0:
            item = self.data[i]
            image = Image.open(item["path"])
            rw, rh = image.size

            T = self.max_resolution
            div = rw * rh / T
            w = round(rw/math.sqrt(div))
            h = round(rh/math.sqrt(div))
            w = round(w / 32) * 32
            h = T // w
            # h = (T // w) // 32 * 32

            self.shape_cache[i][0] = h
            self.shape_cache[i][1] = w
        return h, w

def dataset_loader_old(options, transformed):
    print("[+] Data Loading")

    if options.data.use_small_data and options.curriculum_learning.using:
        collect_each = True
        tmp_train = []
        tmp_valid = []
    else:
        collect_each = False

    # Read data
    train_data, valid_data = [], [] 
    if options.data.random_split:
        print('Train-Test Data Loading')
        print(f'Random Split {options.data.test_proportions}')
        for i, path in enumerate(options.data.train):
            prop = 1.0
            if len(options.data.dataset_proportions) > i:
                prop = options.data.dataset_proportions[i]
            train, valid = split_gt(path, prop, options.data.test_proportions)
            train_data += train
            valid_data += valid
            print(f'From {path}')
            print(f'Prop: {prop}\tTrain +: {len(train)}\tVal +: {len(valid)}')

            if collect_each:
                tmp_train += train[:20]
                tmp_valid += valid[:20]
    else:
        print('Train Data Loading')
        for i, path in enumerate(options.data.train):
            prop = 1.0
            if len(options.data.dataset_proportions) > i:
                prop = options.data.dataset_proportions[i]
            train = split_gt(path, prop)
            train_data += train
            print(f'From {path}')
            print(f'Prop: {prop}\tVal +: {len(train)}')
            if collect_each:
                tmp_train += train[:20]

        print()
        print('Test Data Loading')
        for i, path in enumerate(options.data.test):
            valid = split_gt(path)
            valid_data += valid
            print(f'From {path}')
            print(f'Val +:\t{len(valid)}')
            if collect_each:
                tmp_valid += valid[:20]

    # Load data

    if options.data.use_small_data:
        old_train_len = len(train_data)
        old_valid_len = len(valid_data)
        if collect_each:
            train_data = tmp_train
            valid_data = tmp_valid
        else:
            train_data = train_data[:100]
            valid_data = valid_data[:100]

        print("Using Small Data")
        print(f"Train: {old_train_len} -> {len(train_data)}")
        print(f'Valid: {old_valid_len} -> {len(valid_data)}')

    levels = load_levels(options.data.level_paths)
    sources = load_sources(options.data.source_paths)

    train_dataset = LoadDataset(
        train_data, options.data.token_paths, sources=sources,
        levels=levels, crop=options.data.crop,
        transform=transformed, rgb=options.data.rgb,
        max_resolution=options.input_size.height * options.input_size.width,
        is_flexible=options.data.flexible_image_size,
        is_reverse=options.data.is_reverse,
        use_flip_channel=options.data.use_flip_channel,
    )

    valid_dataset = LoadDataset(
        valid_data, options.data.token_paths, sources=sources,
        levels=levels, crop=options.data.crop,
        transform=transformed, rgb=options.data.rgb,
        max_resolution=options.input_size.height * options.input_size.width,
        is_flexible=options.data.flexible_image_size,
        is_reverse=options.data.is_reverse,
        use_flip_channel=options.data.use_flip_channel,
    )

    if options.data.flexible_image_size:
        train_sampler = SizeBatchSampler(train_dataset, options.batch_size, is_random=True)
        train_data_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=options.num_workers,
            collate_fn=collate_batch,
        )

        valid_sampler = SizeBatchSampler(valid_dataset, options.batch_size, is_random=False)
        valid_data_loader = DataLoader(
            valid_dataset,
            batch_sampler=valid_sampler,
            num_workers=options.num_workers,
            collate_fn=collate_batch,
        )
    else:
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=options.batch_size,
            shuffle=True,
            num_workers=options.num_workers,
            collate_fn=collate_batch,
        )

        valid_data_loader = DataLoader(
            valid_dataset,
            batch_size=options.batch_size,
            shuffle=False,
            num_workers=options.num_workers,
            collate_fn=collate_batch,
        )

    print()
    return train_data_loader, valid_data_loader, train_dataset, valid_dataset

class LevelDataLoaderMaker:
    def __init__(self, dataset, use_flexible=False, args={}):
        self.dataset = dataset
        self.use_flexible = use_flexible
        self.args = args

    def _get_data_loader(self, dataset):
        if self.use_flexible:
            args = deepcopy(self.args)
            sampler = SizeBatchSampler(dataset, args['batch_size'], args['is_random'])
            del args['batch_size']
            del args['is_random']
            return DataLoader(dataset, batch_sampler=sampler, **args)
        else:
            return DataLoader(dataset, **self.args)

    def get_level_data_loader(self, level):
        dataset = self.dataset.get_level_dataset(level)
        return self._get_data_loader(dataset)

    def get_lower_level_loader(self, level):
        dataset = self.dataset.get_lower_level_dataset(level)
        return self._get_data_loader(dataset)

def dataset_loader(options, transformed):
    print("[+] Data Loading")

    # Read data
    levels = load_levels(options.data.level_paths)
    sources = load_sources(options.data.source_paths)


    if options.data.use_small_data and options.curriculum_learning.using:
        collect_each = True
        tmp_train = []
        tmp_valid = []
    else:
        collect_each = False

    # Read data
    train_data, valid_data = [], [] 
    if options.data.random_split:
        print('Train-Test Data Loading')
        print(f'Random Split {options.data.test_proportions}')
        for i, path in enumerate(options.data.train):
            prop = 1.0
            if len(options.data.dataset_proportions) > i:
                prop = options.data.dataset_proportions[i]
            train, valid = split_gt(path, prop, options.data.test_proportions)
            train_data += train
            valid_data += valid
            print(f'From {path}')
            print(f'Prop: {prop}\tTrain +: {len(train)}\tVal +: {len(valid)}')

            if collect_each:
                tmp_train += train[:20]
                tmp_valid += valid[:20]
    else:
        print('Train Data Loading')
        for i, path in enumerate(options.data.train):
            prop = 1.0
            if len(options.data.dataset_proportions) > i:
                prop = options.data.dataset_proportions[i]
            train = split_gt(path, prop)
            train_data += train
            print(f'From {path}')
            print(f'Prop: {prop}\tVal +: {len(train)}')
            if collect_each:
                tmp_train += train[:10]

        print()
        print('Test Data Loading')
        for i, path in enumerate(options.data.test):
            valid = split_gt(path)
            valid_data += valid
            print(f'From {path}')
            print(f'Val +:\t{len(valid)}')
            if collect_each:
                tmp_valid += valid[:10]

    # Load data

    if options.data.use_small_data:
        old_train_len = len(train_data)
        old_valid_len = len(valid_data)
        if collect_each:
            train_data = tmp_train
            valid_data = tmp_valid
        else:
            train_data = train_data[:100]
            valid_data = valid_data[:100]
            
        print("Using Small Data")
        print(f"Train: {old_train_len} -> {len(train_data)}")
        print(f'Valid: {old_valid_len} -> {len(valid_data)}')

    train_dataset = LoadDataset(
        train_data, options.data.token_paths, sources=sources,
        levels=levels, crop=options.data.crop,
        transform=transformed, rgb=options.data.rgb,
        max_resolution=options.input_size.height * options.input_size.width,
        is_flexible=options.data.flexible_image_size,
        use_curr=options.curriculum_learning.using,
        use_flip_channel=options.data.use_flip_channel,
        apply_clihe=options.data.apply_clihe,
    )

    valid_dataset = LoadDataset(
        valid_data, options.data.token_paths, sources=sources,
        levels=levels, crop=options.data.crop,
        transform=transformed, rgb=options.data.rgb,
        max_resolution=options.input_size.height * options.input_size.width,
        is_flexible=options.data.flexible_image_size,
        use_curr=options.curriculum_learning.using,
        use_flip_channel=options.data.use_flip_channel,
        apply_clihe=options.data.apply_clihe,
    )

    if options.curriculum_learning.using:
        if options.data.flexible_image_size:
            train_data_loader = LevelDataLoaderMaker(
                train_dataset,
                use_flexible=options.data.flexible_image_size,
                args={
                    'batch_size': options.batch_size,
                    'is_random': True,
                    'num_workers': options.num_workers,
                    'collate_fn': collate_batch,
                }
            )

            valid_data_loader = LevelDataLoaderMaker(
                valid_dataset,
                use_flexible=options.data.flexible_image_size,
                args={
                    'batch_size': options.batch_size,
                    'is_random': False,
                    'num_workers': options.num_workers,
                    'collate_fn': collate_batch,
                }
            )
        else:
            train_data_loader = LevelDataLoaderMaker(
                train_dataset,
                use_flexible=options.data.flexible_image_size,
                args={
                    'batch_size': options.batch_size,
                    'shuffle': True,
                    'num_workers': options.num_workers,
                    'collate_fn': collate_batch,
                }
            )

            valid_data_loader = LevelDataLoaderMaker(
                valid_dataset,
                use_flexible=options.data.flexible_image_size,
                args={
                    'batch_size': options.batch_size,
                    'shuffle': False,
                    'num_workers': options.num_workers,
                    'collate_fn': collate_batch,
                }
            )
    else:
        if options.data.flexible_image_size:
            train_sampler = SizeBatchSampler(train_dataset, options.batch_size, is_random=True)
            train_data_loader = DataLoader(
                train_dataset,
                batch_sampler=train_sampler,
                num_workers=options.num_workers,
                collate_fn=collate_batch,
            )

            valid_sampler = SizeBatchSampler(valid_dataset, options.batch_size, is_random=False)
            valid_data_loader = DataLoader(
                valid_dataset,
                batch_sampler=valid_sampler,
                num_workers=options.num_workers,
                collate_fn=collate_batch,
            )
        else:
            train_data_loader = DataLoader(
                train_dataset,
                batch_size=options.batch_size,
                shuffle=True,
                num_workers=options.num_workers,
                collate_fn=collate_batch,
            )

            valid_data_loader = DataLoader(
                valid_dataset,
                batch_size=options.batch_size,
                shuffle=False,
                num_workers=options.num_workers,
                collate_fn=collate_batch,
            )

    print()
    return train_data_loader, valid_data_loader, train_dataset, valid_dataset