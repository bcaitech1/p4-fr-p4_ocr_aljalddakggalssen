import os
import argparse
import multiprocessing
import numpy as np
import random
import shutil
import torch
import torch.nn as nn
from torchvision import transforms
import yaml
from checkpoint import (
    default_checkpoint,
    load_checkpoint,
    init_logging,
)
from psutil import virtual_memory

from flags import Flags
from utils import (
    get_network
)
from dataset import dataset_loader, PAD
from train_method import run_trial
from curriculum import run_curriculm

def main(config_file):
    """
    Train math formula recognition model
    """
    
    options, origin_config = Flags(config_file).get()

    #set random seed
    torch.manual_seed(options.seed)
    np.random.seed(options.seed)
    random.seed(options.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    is_cuda = torch.cuda.is_available()
    hardware = "cuda" if is_cuda and options.device_pref == 'cuda' else "cpu"
    device = torch.device(hardware)
    print("--------------------------------")
    print("Running {} on device {}\n".format(options.network, device))

    # Print system environments
    num_gpus = torch.cuda.device_count()
    num_cpus = os.cpu_count()
    mem_size = virtual_memory().available // (1024 ** 3)
    print(
        "[+] System environments\n",
        "The number of gpus : {}\n".format(num_gpus),
        "The number of cpus : {}\n".format(num_cpus),
        "Memory Size : {}G\n".format(mem_size),
    )

    # Load checkpoint and print result
    checkpoint = (
        load_checkpoint(options.checkpoint, cuda=is_cuda)
        if options.checkpoint != ""
        else default_checkpoint
    )
    model_checkpoint = checkpoint["model"]
    if model_checkpoint:
        print(
            "[+] Checkpoint\n",
            "Resuming from epoch : {}\n".format(checkpoint["epoch"]),
            "Train Symbol Accuracy : {:.5f}\n".format(checkpoint["train_symbol_accuracy"][-1]),
            "Train Sentence Accuracy : {:.5f}\n".format(checkpoint["train_sentence_accuracy"][-1]),
            "Train WER : {:.5f}\n".format(checkpoint["train_wer"][-1]),
            "Train Loss : {:.5f}\n".format(checkpoint["train_losses"][-1]),
            "Validation Symbol Accuracy : {:.5f}\n".format(
                checkpoint["validation_symbol_accuracy"][-1]
            ),
            "Validation Sentence Accuracy : {:.5f}\n".format(
                checkpoint["validation_sentence_accuracy"][-1]
            ),
            "Validation WER : {:.5f}\n".format(
                checkpoint["validation_wer"][-1]
            ),
            "Validation Loss : {:.5f}\n".format(checkpoint["validation_losses"][-1]),
        )

    # Get data
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

    # curriculum_learning.using = True -> train_data_loader -> maker임
    train_data_loader, validation_data_loader, train_dataset, valid_dataset = dataset_loader(options, transformed)

    if options.curriculum_learning.using:
        print("[+] Data")
        print("The number of classes : {}".format(len(train_dataset.token_to_id)))

        for i in range(options.curriculum_learning.max_level):
            print(f'----- Level {i + 1} -----')
            print("The number of train samples : {}".format(len(train_dataset.level_idxs[i])))
            print("The number of validation samples : {}".format(len(valid_dataset.level_idxs[i])))
            print()
            
    else:
        print(
            "[+] Data\n",
            "The number of train samples : {}\n".format(len(train_dataset)),
            "The number of validation samples : {}\n".format(len(valid_dataset)),
            "The number of classes : {}\n".format(len(train_dataset.token_to_id)),
        )

    # Get loss, model
    model = get_network(
        options.network,
        options,
        model_checkpoint,
        device,
        train_dataset,
    )
    model.train()

    if options.SATRN.solve_extra_pb:
        criterion = [x.to(device) for x in model.criterion]
    else:
        criterion = model.criterion.to(device)
 
    enc_params_to_optimise = [
        param for param in model.encoder.parameters() if param.requires_grad
    ]
    dec_params_to_optimise = [
        param for param in model.decoder.parameters() if param.requires_grad
    ]
    params_to_optimise = [*enc_params_to_optimise, *dec_params_to_optimise]
    print(
        "[+] Network\n",
        "Type: {}\n".format(options.network),
        "Encoder parameters: {}\n".format(
            sum(p.numel() for p in enc_params_to_optimise),
        ),
        "Decoder parameters: {} \n".format(
            sum(p.numel() for p in dec_params_to_optimise),
        ),
    )

     # Log
    cur_log_dir = os.path.join(options.log_dir, options.prefix)
    if not os.path.exists(cur_log_dir):
        os.makedirs(cur_log_dir)
    log_file = open(os.path.join(cur_log_dir, "log.txt"), "w")
    shutil.copy(config_file, os.path.join(cur_log_dir, "train_config.yaml"))
    if options.print_epochs is None:
        options.print_epochs = options.num_epochs

    # Wandb or Tensorboard
    writer = init_logging(options, model, origin_config)

    start_epoch = checkpoint["epoch"]
    train_symbol_accuracy = checkpoint["train_symbol_accuracy"]
    train_sentence_accuracy=checkpoint["train_sentence_accuracy"]
    train_wer=checkpoint["train_wer"]
    
    train_losses = checkpoint["train_losses"]
    validation_symbol_accuracy = checkpoint["validation_symbol_accuracy"]
    validation_sentence_accuracy=checkpoint["validation_sentence_accuracy"]
    validation_wer=checkpoint["validation_wer"]
    validation_losses = checkpoint["validation_losses"]
    learning_rates = checkpoint["lr"]

    best_metric = [0]

    if options.curriculum_learning.using:
        run_curriculm(
            options,
            train_data_loader, # 사실 maker임
            validation_data_loader,
            model,
            criterion,
            enc_params_to_optimise,
            dec_params_to_optimise,
            device,
            train_losses,
            validation_losses,
            train_symbol_accuracy,
            validation_symbol_accuracy,
            train_sentence_accuracy,
            validation_sentence_accuracy,
            train_wer,
            validation_wer,
            best_metric,
            learning_rates,
            cur_log_dir,
            log_file,
            config_file,
            writer,
            checkpoint,
            set_optimizer_from_checkpoint=True
        )
    
    else:
        run_trial(
            options,
            options.num_epochs,
            train_data_loader,
            validation_data_loader,
            model,
            criterion,
            enc_params_to_optimise,
            dec_params_to_optimise,
            device,
            train_losses,
            validation_losses,
            train_symbol_accuracy,
            validation_symbol_accuracy,
            train_sentence_accuracy,
            validation_sentence_accuracy,
            train_wer,
            validation_wer,
            best_metric,
            learning_rates,
            cur_log_dir,
            log_file,
            config_file,
            writer,
            checkpoint,
            start_epoch,
            set_optimizer_from_checkpoint=options.optimizer.set_optimizer_from_checkpoint,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_file",
        dest="config_file",
        default="/opt/ml/team/gj/code/configs/Default.yaml",
        type=str,
        help="Path of configuration file",
    )
    parser = parser.parse_args()
    main(parser.config_file)
