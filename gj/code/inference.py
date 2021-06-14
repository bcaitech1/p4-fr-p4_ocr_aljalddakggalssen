import sys
sys.path.insert(1, '/opt/ml/team/gj/code')

import torch
import os
from train_method import id_to_string
from metrics import word_error_rate, sentence_acc
from checkpoint import load_checkpoint
from torchvision import transforms
from dataset import LoadEvalDataset, collate_eval_batch, START, PAD, SizeBatchSampler
from flags import Flags
from utils import get_network, get_optimizer
import csv
from torch.utils.data import DataLoader
import argparse
import random
from tqdm import tqdm


def main(parser):
    is_cuda = torch.cuda.is_available()
    checkpoint = load_checkpoint(parser.checkpoint, cuda=is_cuda)
    options, _ = Flags(checkpoint["configs"]).get()
    torch.manual_seed(options.seed)
    random.seed(options.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    hardware = "cuda" if is_cuda else "cpu"
    device = torch.device(hardware)
    print("--------------------------------")
    print("Running {} on device {}\n".format(options.network, device))

    model_checkpoint = checkpoint["model"]
    if model_checkpoint:
        print(
            "[+] Checkpoint\n",
            "Resuming from epoch : {}\n".format(checkpoint["epoch"]),
        )
    print(options.input_size.height)

    if options.data.flexible_image_size:
        transformed = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    else:
        transformed = transforms.Compose(
            [
                transforms.Resize((options.input_size.height, options.input_size.width)),
                transforms.ToTensor(),
            ]
        )

    dummy_gt = "\sin " * parser.max_sequence  # set maximum inference sequence

    root = os.path.join(os.path.dirname(parser.file_path), "images")
    with open(parser.file_path, "r") as fd:
        reader = csv.reader(fd, delimiter="\t")
        data = list(reader)
    test_data = [[os.path.join(root, x[0]), x[0], dummy_gt] for x in data]
    test_dataset = LoadEvalDataset(
        test_data, checkpoint["token_to_id"], checkpoint["id_to_token"], crop=False, transform=transformed,
        rgb=options.data.rgb,
        max_resolution=options.input_size.height * options.input_size.width,
        is_flexible=options.data.flexible_image_size,
        is_reverse=options.data.is_reverse,
        use_flip_channel=options.data.use_flip_channel,
        apply_clihe=options.data.apply_clihe,
        rotate_img=options.data.rotate_img,
    )

    if options.data.flexible_image_size:
        test_sampler = SizeBatchSampler(test_dataset, parser.batch_size, is_random=False)
        test_data_loader = DataLoader(
            test_dataset,
            batch_sampler=test_sampler,
            num_workers=options.num_workers,
            collate_fn=collate_eval_batch,
        )
    else:
        test_data_loader = DataLoader(
            test_dataset,
            batch_size=parser.batch_size,
            shuffle=False,
            num_workers=options.num_workers,
            collate_fn=collate_eval_batch,
        )

    print(
        "[+] Data\n",
        "The number of test samples : {}\n".format(len(test_dataset)),
    )

    model = get_network(
        options.network,
        options,
        model_checkpoint,
        device,
        test_dataset,
    )
    model.eval()

    if parser.use_rotation == 'yes':
        print('using rotation inference')
        results = {}
        newresults={}
        crit=[]
        rot=[0,1,3,2]
        for i in rot[0:int(parser.direction_nums)]:
            print(f'{i} 번째 회전')
            newresults={}
            for d in tqdm(test_data_loader):
                input = d["image"].to(device)
        #         print(input.shape)
                input=torch.rot90(input, i, [2, 3])
                expected = d["truth"]["encoded"].to(device)

                output = model(input, expected, False, 0.0)
                    
                decoded_values = output.transpose(1, 2)
                # decoded_values = torch.softmax(decoded_values, dim=1)
                score, sequence = torch.topk(decoded_values, 1, dim=1)
                sequence = sequence.squeeze(1)
                score = score.squeeze(1)
                score=torch.log(score)
                score=torch.sum(score,1,True).tolist()

                if parser.rotation_print == 'yes':
                    print(score)
                sequence_str = id_to_string(sequence, test_data_loader, do_eval=1)

                if parser.rotation_print == 'yes':
                    print(sequence_str)
                if i ==0:
                    for path, predicted, score in zip(d["file_path"], sequence_str,score):
                        results[path] = [predicted,score[0]]
                else:
                    for path, predicted, score in zip(d["file_path"], sequence_str,score):
                        newresults[path] = [predicted,score[0]]
                    for k in newresults:
                        if (newresults[k][1]>results[k][1]) and (int(parser.ans_th)>results[k][1]):
                            results[k]=newresults[k]
    #             print('one rotation done')
                

        os.makedirs(parser.output_dir, exist_ok=True)
        with open(os.path.join(parser.output_dir, "output.csv"), "w") as w:
            for i in range(len(test_dataset)):
                path = test_dataset[i]['file_path']
                predicted, _ = results[path]
                w.write(path + "\t" + predicted + "\n")
    else:
        also_greedy = parser.also_greedy == 'yes'
        print('also_greedy', also_greedy)
        print('using normal inference')
        results = {}
        for d in tqdm(test_data_loader):
            input = d["image"].to(device)
            expected = d["truth"]["encoded"].to(device)

            output = model(input, expected, False, 0.0,
                beam_search_k=parser.beam_search_k, also_greedy=also_greedy)

            if parser.beam_search_k > 1:
                sequence = output
            else:
                decoded_values = output.transpose(1, 2)
                _, sequence = torch.topk(decoded_values, 1, dim=1)
                sequence = sequence.squeeze(1)

            sequence_str = id_to_string(sequence, test_data_loader, do_eval=1)
            for path, predicted in zip(d["file_path"], sequence_str):
                results[path] =  predicted

        os.makedirs(parser.output_dir, exist_ok=True)
        with open(os.path.join(parser.output_dir, "output.csv"), "w") as w:
            for i in range(len(test_dataset)):
                path = test_dataset[i]['file_path']
                predicted = results[path]
                w.write(path + "\t" + predicted + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        default="/opt/ml/code/satrn_cur_best_long/checkpoints/best.pth",
        type=str,
        help="Path of checkpoint file",
    )
    parser.add_argument(
        "--max_sequence",
        dest="max_sequence",
        default=230,
        type=int,
        help="maximun sequence when doing inference",
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        default=8,
        type=int,
        help="batch size when doing inference",
    )

    eval_dir = os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/')
    file_path = os.path.join(eval_dir, 'eval_dataset/input.txt')
    parser.add_argument(
        "--file_path",
        dest="file_path",
        default=file_path,
        type=str,
        help="file path when doing inference",
    )

    output_dir = os.environ.get('SM_OUTPUT_DATA_DIR', 'submit')
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        default=output_dir,
        type=str,
        help="output directory",
    )

    parser.add_argument(
        "--direction_nums",
        dest="direction_nums",
        default="1",
        type=str,
        help="none, left, right",
    )
    parser.add_argument(
        "--ans_th",
        dest="ans_th",
        default="600",
        type=str,
        help="changing ans threshhold",
    )
    
    parser.add_argument(
        "--use_rotation",
        dest="use_rotation",
        default="no",
        type=str,
    )
    
    parser.add_argument(
        "--rotation_print",
        dest="rotation_print",
        default="yes",
        type=str,
    )

    parser.add_argument(
        "--beam_search_k",
        dest="beam_search_k",
        default=1,
        type=int,
    )
    
    parser.add_argument(
        "--also_greedy",
        dest="also_greedy",
        default="no",
        type=str,
    )

    parser = parser.parse_args()
    main(parser)
