import os
import argparse
import multiprocessing
import numpy as np
import random
import time
import shutil
import torch
import torch.nn as nn
from torchvision import transforms
import yaml
from tqdm import tqdm
from checkpoint import (
    default_checkpoint,
    load_checkpoint,
    save_checkpoint,
    init_logging,
    log_stuff,
    log_best_stuff,
)
from psutil import virtual_memory

from flags import Flags
from utils import (
    get_network,
    get_enc_dec_optimizer,
    setup_enc_dec_optimizer,
    get_enc_dec_lr_scheduler,
)
from dataset import dataset_loader, START, PAD,load_vocab

from metrics import word_error_rate,sentence_acc

from torch.cuda.amp import (
    GradScaler, 
    autocast,
)

def id_to_string(tokens, data_loader,do_eval=0):
    result = []
    if do_eval:
        special_ids = [data_loader.dataset.token_to_id["<PAD>"], data_loader.dataset.token_to_id["<SOS>"],
                       data_loader.dataset.token_to_id["<EOS>"]]

    for example in tokens:
        string = ""
        if do_eval:
            for token in example:
                token = token.item()
                if token not in special_ids:
                    if token != -1:
                        string += data_loader.dataset.id_to_token[token] + " "
        else:
            for token in example:
                token = token.item()
                if token != -1:
                    string += data_loader.dataset.id_to_token[token] + " "

        result.append(string)
    return result

def run_epoch(
    data_loader,
    model,
    epoch_text,
    criterion,
    enc_optimizer, 
    dec_optimizer,
    enc_lr_scheduler,
    dec_lr_scheduler,
    teacher_forcing_ratio,
    max_grad_norm,
    device,
    options,
    use_amp=False,
    train=True,
):
    # Disables autograd during validation mode
    torch.set_grad_enabled(train)
    if train:
        model.train()
        scaler = GradScaler(enabled=use_amp)
    else:
        model.eval()

    if options.SATRN.solve_extra_pb:
        losses_satrn = []
        losses_level = []
        losses_source = []
    losses = []
    total_inputs = 0
    # grad_norms = []
    correct_symbols = 0
    total_symbols = 0
    wer=0
    num_wer=0
    sent_acc=0
    num_sent_acc=0

    with tqdm(
        desc="{} ({})".format(epoch_text, "Train" if train else "Validation"),
        total=len(data_loader.dataset),
        dynamic_ncols=True,
        leave=False,
    ) as pbar:
        for d in data_loader:
            input = d["image"].to(device)

            # The last batch may not be a full batch
            curr_batch_size = len(input)
            expected = d["truth"]["encoded"].to(device)
            levels_expected = d['level'].to(device)
            sources_expected = d['source'].to(device)

            # Replace -1 with the PAD token
            expected[expected == -1] = data_loader.dataset.token_to_id[PAD]

            with autocast(enabled=use_amp):
                output_dict = model(input, expected, train, teacher_forcing_ratio)
                output = output_dict['out']
                if options.SATRN.solve_extra_pb:
                    level_result = output_dict['level_out']
                    source_result = output_dict['source_out']

                decoded_values = output.transpose(1, 2)
                _, sequence = torch.topk(decoded_values, 1, dim=1)
                sequence = sequence.squeeze(1)

                if options.SATRN.solve_extra_pb:
                    loss_satrn = criterion[0](decoded_values, expected[:, 1:])
                    loss_level = criterion[1](level_result, levels_expected)
                    loss_source = criterion[2](source_result, sources_expected)
                    loss = loss_satrn + loss_level + loss_source
                else:
                    loss = criterion(decoded_values, expected[:, 1:])

            if train:
                enc_optim_params = [
                    p
                    for param_group in enc_optimizer.param_groups
                    for p in param_group["params"]
                ]

                dec_optim_params = [
                    p
                    for param_group in dec_optimizer.param_groups
                    for p in param_group["params"]
                ]

                enc_optimizer.zero_grad()
                dec_optimizer.zero_grad()
                
                scaler.scale(loss).backward()
                scaler.unscale_(enc_optimizer)
                scaler.unscale_(dec_optimizer)
                # Clip gradients, it returns the total norm of all parameters
                nn.utils.clip_grad_norm_(
                    enc_optim_params, max_norm=max_grad_norm
                )

                nn.utils.clip_grad_norm_(
                    dec_optim_params, max_norm=max_grad_norm
                )
                # grad_norms.append(grad_norm)

                # cycle
                scaler.step(enc_optimizer)
                scaler.step(dec_optimizer)
                scale = scaler.get_scale()
                scaler.update()
                step_scheduler = scaler.get_scale() == scale

                if step_scheduler:
                    enc_lr_scheduler.step()
                    dec_lr_scheduler.step()

            if options.SATRN.solve_extra_pb:
                losses.append(loss.item() * len(input))
                losses_satrn.append(loss_satrn.item() * len(input))
                losses_level.append(loss_level.item() * len(input))
                losses_source.append(loss_source.item() * len(input))
            else:
                losses.append(loss.item() * len(input))
            total_inputs += len(input)

            expected[expected == data_loader.dataset.token_to_id[PAD]] = -1
            expected_str = id_to_string(expected, data_loader,do_eval=1)
            sequence_str = id_to_string(sequence, data_loader,do_eval=1)
            wer += word_error_rate(sequence_str,expected_str)
            num_wer += len(expected_str)
            sent_acc += sentence_acc(sequence_str,expected_str)
            num_sent_acc += len(expected_str)
            correct_symbols += torch.sum(sequence == expected[:, 1:], dim=(0, 1)).item()
            total_symbols += torch.sum(expected[:, 1:] != -1, dim=(0, 1)).item()

            pbar.update(curr_batch_size)

    expected = id_to_string(expected, data_loader)
    sequence = id_to_string(sequence, data_loader)
    print("-" * 10 + "GT ({})".format("train" if train else "valid"))
    print(*expected[:3], sep="\n")
    print("-" * 10 + "PR ({})".format("train" if train else "valid"))
    print(*sequence[:3], sep="\n")
    
    if options.SATRN.solve_extra_pb:
        result = {
            "loss": np.sum(losses_satrn) / total_inputs,
            'loss_total': np.sum(losses) / total_inputs,
            'loss_level': np.sum(losses_level) / total_inputs,
            'loss_source': np.sum(losses_source) / total_inputs,
            "correct_symbols": correct_symbols,
            "total_symbols": total_symbols,
            "wer": wer,
            "num_wer":num_wer,
            "sent_acc": sent_acc,
            "num_sent_acc":num_sent_acc
        }
    else:
        result = {
            "loss": np.sum(losses) / total_inputs,
            "correct_symbols": correct_symbols,
            "total_symbols": total_symbols,
            "wer": wer,
            "num_wer":num_wer,
            "sent_acc": sent_acc,
            "num_sent_acc":num_sent_acc
        }

    # if train:
    #     try:
    #         result["grad_norm"] = np.mean([tensor.cpu() for tensor in grad_norms])
    #     except:
    #         result["grad_norm"] = np.mean(grad_norms)

    return result


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

    train_data_loader, validation_data_loader, train_dataset, valid_dataset = dataset_loader(options, transformed)
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

    # Get optimizer
    enc_optimizer, dec_optimizer = get_enc_dec_optimizer(
        options, 
        enc_params_to_optimise,
        dec_params_to_optimise,
    )

    setup_enc_dec_optimizer(options, checkpoint,
         enc_optimizer, dec_optimizer)
        
    enc_lr_scheduler, dec_lr_scheduler = get_enc_dec_lr_scheduler(
        options, enc_optimizer, dec_optimizer, train_data_loader
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
    # grad_norms = checkpoint["grad_norm"]

    best_metric = 0
    # Train

    for epoch in range(options.num_epochs):
        start_time = time.time()

        epoch_text = "[{current:>{pad}}/{end}] Epoch {epoch}".format(
            current=epoch + 1,
            end=options.num_epochs,
            epoch=start_epoch + epoch + 1,
            pad=len(str(options.num_epochs)),
        )

        # Train
        if options.num_epochs == 1:
            teacher_forcing_ratio = options.teacher_forcing_ratio
        else:
            if hasattr(options, 'teacher_forcing_ratio_drop'):
                tf_ratio_drop = options.teacher_forcing_ratio_drop
            else:
                tf_ratio_drop = 0
            teacher_forcing_ratio = options.teacher_forcing_ratio- \
                    (epoch/(options.num_epochs-1))*tf_ratio_drop
        # print(teacher_forcing_ratio)
        train_result = run_epoch(
            train_data_loader,
            model,
            epoch_text,
            criterion,
            enc_optimizer,
            dec_optimizer,
            enc_lr_scheduler,
            dec_lr_scheduler,
            teacher_forcing_ratio,
            options.max_grad_norm,
            device,
            options=options,
            use_amp=options.use_amp and device.type == 'cuda',
            train=True,
        )



        train_losses.append(train_result["loss"])
        # grad_norms.append(train_result["grad_norm"])
        train_epoch_symbol_accuracy = (
            train_result["correct_symbols"] / train_result["total_symbols"]
        )
        train_symbol_accuracy.append(train_epoch_symbol_accuracy)
        train_epoch_sentence_accuracy = (
                train_result["sent_acc"] / train_result["num_sent_acc"]
        )

        train_sentence_accuracy.append(train_epoch_sentence_accuracy)
        train_epoch_wer = (
                train_result["wer"] / train_result["num_wer"]
        )
        train_wer.append(train_epoch_wer)
        enc_epoch_lr = enc_lr_scheduler.get_lr()  # cycle
        dec_epoch_lr = enc_lr_scheduler.get_lr() 

        # Validation
        validation_result = run_epoch(
            validation_data_loader,
            model,
            epoch_text,
            criterion,
            enc_optimizer,
            dec_optimizer,
            enc_lr_scheduler,
            dec_lr_scheduler,
            options.teacher_forcing_ratio,
            options.max_grad_norm,
            device,
            options=options,
            use_amp=options.use_amp and device.type == 'cuda',
            train=False,
        )
        validation_losses.append(validation_result["loss"])
        validation_epoch_symbol_accuracy = (
            validation_result["correct_symbols"] / validation_result["total_symbols"]
        )
        validation_symbol_accuracy.append(validation_epoch_symbol_accuracy)

        validation_epoch_sentence_accuracy = (
            validation_result["sent_acc"] / validation_result["num_sent_acc"]
        )
        validation_sentence_accuracy.append(validation_epoch_sentence_accuracy)
        validation_epoch_wer = (
                validation_result["wer"] / validation_result["num_wer"]
        )
        validation_wer.append(validation_epoch_wer)

        # Save checkpoint
        #make config

        new_metric = validation_epoch_sentence_accuracy * 0.9 + (1-validation_epoch_wer) * 0.1
        best_changed = False
        if best_metric <= new_metric:
            best_metric = new_metric
            best_changed = True

        if options.save_type == 'best':
            if best_changed:
                with open(config_file, 'r') as f:
                    option_dict = yaml.safe_load(f)

                save_checkpoint(
                    {
                        "epoch": start_epoch + epoch + 1,
                        "train_losses": train_losses,
                        "train_symbol_accuracy": train_symbol_accuracy,
                        "train_sentence_accuracy": train_sentence_accuracy,
                        "train_wer":train_wer,
                        "validation_losses": validation_losses,
                        "validation_symbol_accuracy": validation_symbol_accuracy,
                        "validation_sentence_accuracy":validation_sentence_accuracy,
                        "validation_wer":validation_wer,
                        "lr": learning_rates,
                        # "grad_norm": grad_norms,
                        "model": model.state_dict(),
                        "enc_optimizer": enc_optimizer.state_dict(),
                        "dec_optimizer": dec_optimizer.state_dict(),
                        "configs": option_dict,
                        "token_to_id":train_data_loader.dataset.token_to_id,
                        "id_to_token":train_data_loader.dataset.id_to_token
                    },
                    prefix=cur_log_dir,
                    save_best=True,
                )

        # Summary
        elapsed_time = time.time() - start_time
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        if epoch % options.print_epochs == 0 or epoch == options.num_epochs - 1:
            output_string = (
                "{epoch_text}: "
                "Train Symbol Accuracy = {train_symbol_accuracy:.5f}, "
                "Train Sentence Accuracy = {train_sentence_accuracy:.5f}, "
                "Train WER = {train_wer:.5f}, "
                "Train Loss = {train_loss:.5f}, "
                "Validation Symbol Accuracy = {validation_symbol_accuracy:.5f}, "
                "Validation Sentence Accuracy = {validation_sentence_accuracy:.5f}, "
                "Validation WER = {validation_wer:.5f}, "
                "Validation Loss = {validation_loss:.5f}, "
                "enc_lr = {enc_lr} "
                "dec_lr = {dec_lr} "
                "(time elapsed {time})"
            ).format(
                epoch_text=epoch_text,
                train_symbol_accuracy=train_epoch_symbol_accuracy,
                train_sentence_accuracy=train_epoch_sentence_accuracy,
                train_wer=train_epoch_wer,
                train_loss=train_result["loss"],
                validation_symbol_accuracy=validation_epoch_symbol_accuracy,
                validation_sentence_accuracy=validation_epoch_sentence_accuracy,
                validation_wer=validation_epoch_wer,
                validation_loss=validation_result["loss"],
                enc_lr=enc_epoch_lr,
                dec_lr=dec_epoch_lr,
                time=elapsed_time,
            )
            print(output_string)
            log_file.write(output_string + "\n")

            if options.SATRN.solve_extra_pb:
                log_stuff(
                    options,
                    writer,
                    start_epoch + epoch + 1,
                    # train_result["grad_norm"],
                    train_result["loss"],
                    train_epoch_symbol_accuracy,
                    train_epoch_sentence_accuracy,
                    train_epoch_wer,
                    validation_result["loss"],
                    validation_epoch_symbol_accuracy,
                    validation_epoch_sentence_accuracy,
                    validation_epoch_wer,
                    model,
                    train_total_loss=train_result['loss_total'],
                    train_level_loss=train_result['loss_level'],
                    train_source_loss=train_result['loss_source'],
                    validation_total_loss=validation_result["loss_total"],
                    validation_level_loss=validation_result["loss_level"],
                    validation_source_loss=validation_result["loss_source"],
                )
            else:
                log_stuff(
                    options,
                    writer,
                    start_epoch + epoch + 1,
                    # train_result["grad_norm"],
                    train_result["loss"],
                    train_epoch_symbol_accuracy,
                    train_epoch_sentence_accuracy,
                    train_epoch_wer,
                    validation_result["loss"],
                    validation_epoch_symbol_accuracy,
                    validation_epoch_sentence_accuracy,
                    validation_epoch_wer,
                    enc_epoch_lr,
                    dec_epoch_lr,
                    model,
                    teacher_forcing_ratio=teacher_forcing_ratio,
                )

            if best_changed:
                log_best_stuff(options, best_metric)


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
