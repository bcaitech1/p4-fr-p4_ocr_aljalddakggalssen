import csv
import os
import random
import math
from collections import defaultdict
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Sampler

from tqdm.auto import trange, tqdm
from torchvision import transforms

START = "<SOS>"
END = "<EOS>"
PAD = "<PAD>"
SPECIAL_TOKENS = [START, END, PAD]


# Rather ignorant way to encode the truth, but at least it works.
def encode_truth(truth, token_to_id):

    truth_tokens = truth.split()
    for token in truth_tokens:
        if token not in token_to_id:
            raise Exception("Truth contains unknown token")
    truth_tokens = [token_to_id[x] for x in truth_tokens]
    if '' in truth_tokens: truth_tokens.remove('')
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

class LoadDataset(Dataset):
    """Load Dataset"""

    def __init__(
        self,
        groundtruth,
        tokens_file,
        crop=False,
        transform=None,
        rgb=3,
        max_resolution=128*128,
        is_flexible=False,
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
        self.crop = crop
        self.transform = transform
        self.rgb = rgb
        self.token_to_id, self.id_to_token = load_vocab(tokens_file)
        self.data = [
            {
                "path": p,
                "truth": {
                    "text": truth,
                    "encoded": [
                        self.token_to_id[START],
                        *encode_truth(truth, self.token_to_id),
                        self.token_to_id[END],
                    ],
                },
            }
            for p, truth in groundtruth
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

        return {"path": item["path"], "truth": item["truth"], "image": image}

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
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.transform = transform
        self.data = [
            {
                "path": p,
                "file_path":p1,
                "truth": {
                    "text": truth,
                    "encoded": [
                        self.token_to_id[START],
                        *encode_truth(truth, self.token_to_id),
                        self.token_to_id[END],
                    ],
                },
            }
            for p, p1,truth in groundtruth
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        image = Image.open(item["path"])
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

        return {"path": item["path"], "file_path":item["file_path"],"truth": item["truth"], "image": image}

def dataset_loader(options, transformed):
    print("[+] Data Loading")

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

        print()
        print('Test Data Loading')
        for i, path in enumerate(options.data.test):
            valid = split_gt(path)
            valid_data += valid
            print(f'From {path}')
            print(f'Val +:\t{len(valid)}')

    # Load data
    if options.data.use_small_data:
        old_train_len = len(train_data)
        old_valid_len = len(valid_data)
        train_data = train_data[:100]
        valid_data = valid_data[:10]
        print("Using Small Data")
        print(f"Train: {old_train_len} -> {len(train_data)}")
        print(f'Valid: {old_valid_len} -> {len(valid_data)}')

    train_dataset = LoadDataset(
        train_data, options.data.token_paths, crop=options.data.crop,
        transform=transformed, rgb=options.data.rgb,
        max_resolution=options.input_size.height * options.input_size.width,
        is_flexible=options.data.flexible_image_size,
    )

    valid_dataset = LoadDataset(
        valid_data, options.data.token_paths, crop=options.data.crop,
        transform=transformed, rgb=options.data.rgb,
        max_resolution=options.input_size.height * options.input_size.width,
        is_flexible=options.data.flexible_image_size,
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
