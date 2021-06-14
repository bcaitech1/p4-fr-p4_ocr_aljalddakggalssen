import numpy as np
import time
import torch
import torch.nn as nn
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

from utils import (
    get_enc_dec_optimizer,
    setup_enc_dec_optimizer,
    get_enc_dec_lr_scheduler,
)
from dataset import PAD

from metrics import (
    word_error_rate,
    sentence_acc,
    correct_symbol,
)

from torch.cuda.amp import (
    GradScaler, 
    autocast,
)

def id_to_string(tokens, data_loader,do_eval=0):
    result = []

    if do_eval:
        eos_id = data_loader.dataset.token_to_id["<EOS>"]
        special_ids = set([data_loader.dataset.token_to_id["<PAD>"], data_loader.dataset.token_to_id["<SOS>"],
                       eos_id])

    for example in tokens:
        string = ""
        if do_eval:
            for token in example:
                token = token.item()
                if token not in special_ids:
                    if token != -1:
                        string += data_loader.dataset.id_to_token[token] + " "
                elif token == eos_id:
                    break
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
    beam_search_k=1,
    also_greedy=False,
):
    # Disables autograd during validation mode
    torch.set_grad_enabled(train)
    if train:
        model.train()
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
            if train:
                enc_optimizer.zero_grad()
                dec_optimizer.zero_grad()
                
            input = d["image"].to(device)

            # The last batch may not be a full batch
            curr_batch_size = len(input)
            expected = d["truth"]["encoded"].to(device)
            # levels_expected = d['level'].to(device)
            # sources_expected = d['source'].to(device)

            # Replace -1 with the PAD token
            expected[expected == -1] = data_loader.dataset.token_to_id[PAD]

            with autocast(enabled=use_amp):
                output = model(input, expected, train, teacher_forcing_ratio,
                    beam_search_k=beam_search_k, also_greedy=also_greedy)
                # output = output_dict['out']
                # if options.SATRN.solve_extra_pb:
                #     level_result = output_dict['level_out']
                #     source_result = output_dict['source_out']

                if beam_search_k > 1:
                    sequence = output
                else:
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
                    
                    del decoded_values

            if train:
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

            if beam_search_k ==  1:
                if options.SATRN.solve_extra_pb:
                    losses.append(loss.item() * len(input))
                    losses_satrn.append(loss_satrn.item() * len(input))
                    losses_level.append(loss_level.item() * len(input))
                    losses_source.append(loss_source.item() * len(input))
                else:
                    losses.append(loss.item() * len(input))
                
                del loss, output
            total_inputs += len(input)

            expected = expected.cpu()
            sequence = sequence.cpu()
            expected[expected == data_loader.dataset.token_to_id[PAD]] = -1
            expected_str = id_to_string(expected, data_loader,do_eval=1)
            sequence_str = id_to_string(sequence, data_loader,do_eval=1)
            wer += word_error_rate(sequence_str,expected_str)
            num_wer += len(expected_str)
            sent_acc += sentence_acc(sequence_str,expected_str)
            num_sent_acc += len(expected_str)
            correct_symbols += correct_symbol(sequence, expected[:, 1:])
            total_symbols += torch.sum(expected[:, 1:] != -1, dim=(0, 1)).item()

            pbar.update(curr_batch_size)

    expected = id_to_string(expected, data_loader)
    sequence = id_to_string(sequence, data_loader)
    print("-" * 10 + "GT ({})".format("train" if train else "valid"))
    print(*expected[:3], sep="\n")
    print("-" * 10 + "PR ({})".format("train" if train else "valid"))
    print(*sequence[:3], sep="\n")

    del expected, sequence
    
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
            "correct_symbols": correct_symbols,
            "total_symbols": total_symbols,
            "wer": wer,
            "num_wer":num_wer,
            "sent_acc": sent_acc,
            "num_sent_acc":num_sent_acc
        }

        if beam_search_k == 1:
            result['loss'] = np.sum(losses) / total_inputs

    # if train:
    #     try:
    #         result["grad_norm"] = np.mean([tensor.cpu() for tensor in grad_norms])
    #     except:
    #         result["grad_norm"] = np.mean(grad_norms)

    return result

def run_trial(
    options,
    num_epochs,
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
    set_optimizer_from_checkpoint=True,
    level_info='',
    is_first=False,
):
    # Get optimizer
    enc_optimizer, dec_optimizer = get_enc_dec_optimizer(
        options, 
        enc_params_to_optimise,
        dec_params_to_optimise,
    )

    if set_optimizer_from_checkpoint:
        setup_enc_dec_optimizer(options, checkpoint,
            enc_optimizer, dec_optimizer)
        
    enc_lr_scheduler, dec_lr_scheduler = get_enc_dec_lr_scheduler(
        options, enc_optimizer, dec_optimizer, train_data_loader, is_first
    )

    for epoch in range(num_epochs):
        start_time = time.time()

        epoch_text = "[{current:>{pad}}/{end}] {level_info}Epoch {epoch}".format(
            current=epoch + 1,
            end=num_epochs,
            epoch=start_epoch + epoch + 1,
            pad=len(str(num_epochs)),
            level_info=level_info,
        )

        # Train
        if num_epochs == 1:
            teacher_forcing_ratio = options.teacher_forcing_ratio
        else:
            if hasattr(options, 'teacher_forcing_ratio_drop'):
                tf_ratio_drop = options.teacher_forcing_ratio_drop
            else:
                tf_ratio_drop = 0
            teacher_forcing_ratio = options.teacher_forcing_ratio- \
                    (epoch/(num_epochs-1))*tf_ratio_drop
        
        if not options.run_only_valid:
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
            enc_epoch_lr = enc_lr_scheduler.get_last_lr()[0]  # cycle
            dec_epoch_lr = enc_lr_scheduler.get_last_lr()[0] 

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
            beam_search_k=options.beam_search_k,
            also_greedy=options.also_greedy,
        )

        if options.beam_search_k == 1:
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
        if best_metric[0] <= new_metric:
            best_metric[0] = new_metric
            best_changed = True

        if options.save_type == 'latest':
            should_save = True
        elif options.save_type == 'best':
            should_save = best_changed
        else:
            should_save = False

        if should_save and not options.run_only_valid:
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
                save_type=options.save_type,
            )

        # Summary
        elapsed_time = time.time() - start_time
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        if epoch % options.print_epochs == 0 or epoch == options.num_epochs - 1:
            if options.run_only_valid:
                if options.beam_search_k == 1:
                    val_loss_result = validation_result["loss"]
                else:
                    val_loss_result = 0
                output_string = (
                "{epoch_text}: "
                # "Train Symbol Accuracy = {train_symbol_accuracy:.5f}, "
                # "Train Sentence Accuracy = {train_sentence_accuracy:.5f}, "
                # "Train WER = {train_wer:.5f}, "
                # "Train Loss = {train_loss:.5f}, "
                "Validation Symbol Accuracy = {validation_symbol_accuracy:.5f}, "
                "Validation Sentence Accuracy = {validation_sentence_accuracy:.5f}, "
                "Validation WER = {validation_wer:.5f}, "
                "Validation Loss = {validation_loss:.5f}, "
                # "enc_lr = {enc_lr} "
                # "dec_lr = {dec_lr} "
                "(time elapsed {time})"
            ).format(
                epoch_text=epoch_text,
                # train_symbol_accuracy=train_epoch_symbol_accuracy,
                # train_sentence_accuracy=train_epoch_sentence_accuracy,
                # train_wer=train_epoch_wer,
                # train_loss=train_result["loss"],
                validation_symbol_accuracy=validation_epoch_symbol_accuracy,
                validation_sentence_accuracy=validation_epoch_sentence_accuracy,
                validation_wer=validation_epoch_wer,
                validation_loss=val_loss_result,
                # enc_lr=enc_epoch_lr,
                # dec_lr=dec_epoch_lr,
                time=elapsed_time,
            )
            else:
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