import torch
import os
from train import id_to_string
from metrics import word_error_rate, sentence_acc
from checkpoint import load_checkpoint
from torchvision import transforms
from dataset import LoadEvalDataset, collate_eval_batch, START, PAD
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
    options = Flags(checkpoint["configs"]).get()
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

    transformed = transforms.Compose(
        [
            transforms.Resize((options.input_size.height, options.input_size.width)),
#             transforms.functional.rotate(angle=90),
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
        rgb=options.data.rgb
    )
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
    results = []
    newresults=[]
    crit=[]
    rot=[0,1,3,2]
    for i in rot[0:int(parser.direction_nums)]:
        newresults=[]
        for d in tqdm(test_data_loader):
            input = d["image"].to(device)
    #         print(input.shape)
            input=torch.rot90(input, i, [2, 3])
            expected = d["truth"]["encoded"].to(device)

            output = model(input, expected, False, 0.0)
            decoded_values = output.transpose(1, 2)
            score, sequence = torch.topk(decoded_values, 1, dim=1)
            sequence = sequence.squeeze(1)
            score = score.squeeze(1)
            score=torch.log(score)
            score=torch.sum(score,1,True).tolist()
            print(score)
            sequence_str = id_to_string(sequence, test_data_loader, do_eval=1)
            print(sequence_str)
            if i ==0:
                for path, predicted, score in zip(d["file_path"], sequence_str,score):
                    results.append([path, predicted,score[0]])
            else:
                for path, predicted, score in zip(d["file_path"], sequence_str,score):
                    newresults.append([path, predicted,score[0]])
                for k in range(len(newresults)):
                    if (newresults[k][2]>results[k][2]) and (int(parser.ans_th)>results[k][2]):
                        results[k]=newresults[k]
#             print('one rotation done')
            

    os.makedirs(parser.output_dir, exist_ok=True)
    with open(os.path.join(parser.output_dir, "output.csv"), "w") as w:
        for path, predicted,score in results:
            w.write(path + "\t" + predicted  +"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        default="../input/C_0061.pth",
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
        default="2",
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


    parser = parser.parse_args()
    main(parser)
