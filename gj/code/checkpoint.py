import os
import torch
from tensorboardX import SummaryWriter
import wandb

from flatten_dict import flatten

use_cuda = torch.cuda.is_available()

default_checkpoint = {
    "epoch": 0,
    "train_losses": [],
    "train_symbol_accuracy": [],
    "train_sentence_accuracy": [],
    "train_wer": [],
    "validation_losses": [],
    "validation_symbol_accuracy": [],
    "validation_sentence_accuracy": [],
    "validation_wer": [],
    "lr": [],
    # "grad_norm": [],
    "model": {},
    "configs":{},
    "token_to_id":{},
    "id_to_token":{},
}


def save_checkpoint(checkpoint, dir="./checkpoints", prefix="", save_best=False):
    # Padded to 4 digits because of lexical sorting of numbers.
    # e.g. 0009.pth
    if save_best:
        filename = "best.pth"
    else:
        filename = "{num:0>4}.pth".format(num=checkpoint["epoch"])
    if not os.path.exists(os.path.join(prefix, dir)):
        os.makedirs(os.path.join(prefix, dir))
    torch.save(checkpoint, os.path.join(prefix, dir, filename))


def load_checkpoint(path, cuda=use_cuda):
    if cuda:
        return torch.load(path)
    else:
        # Load GPU model on CPU
        return torch.load(path, map_location=lambda storage, loc: storage)

def init_logging(options, model, origin_config):
    if options.use_log_type is None:
        return None

    if options.use_log_type == 'tensorboard':
        cur_log_dir = os.path.join(options.log_dir, options.prefix)
        return init_tensorboard(cur_log_dir)
    elif options.use_log_type == 'wandb':
        if options.wandb.name is None:
            name = options.prefix
        else:
            name = options.wandb.name

        wandb.init(project=options.wandb.project,
                entity=options.wandb.entity,
               tags=options.wandb.tags, name=name,
               config=flatten(origin_config, reducer='path'))
        wandb.watch(model)
        return None
    else:
        raise NotImplementedError('Not supported logging')

def log_stuff(
    options,
    writer,
    epoch,
    # grad_norm,
    train_loss,
    train_symbol_accuracy,
    train_sentence_accuracy,
    train_wer,
    validation_loss,
    validation_symbol_accuracy,
    validation_sentence_accuracy,
    validation_wer,
    enc_lr,
    dec_lr,
    model,
    train_total_loss=None,
    train_level_loss=None,
    train_source_loss=None,
    validation_total_loss=None,
    validation_level_loss=None,
    validation_source_loss=None,
    teacher_forcing_ratio=None,
):
    if options.use_log_type is None:
        return None

    if options.use_log_type == 'tensorboard':
        write_tensorboard(writer, epoch, 
        #grad_norm,
         train_loss, 
            train_symbol_accuracy, train_sentence_accuracy, train_wer,
             validation_loss, validation_symbol_accuracy,
              validation_sentence_accuracy, validation_wer, model)
    elif options.use_log_type == 'wandb':
        logging_stuff = {
            'train_loss': train_loss,
            'train_symbol_accuracy': train_symbol_accuracy,
            'train_sentence_accuracy': train_sentence_accuracy,
            'train_wer': train_wer,
            'validation_loss': validation_loss,
            'validation_symbol_accuracy': validation_symbol_accuracy,
            'validation_sentence_accuracy': validation_sentence_accuracy,
            'validation_wer': validation_wer,
            'enc_lr': enc_lr,
            'dec_lr': dec_lr,
            'teacher_forcing_ratio': teacher_forcing_ratio,
        }

        if train_total_loss is not None:
            logging_stuff['train_total_loss'] = train_total_loss
        if train_level_loss is not None:
            logging_stuff['train_level_loss'] = train_level_loss
        if train_source_loss is not None:
            logging_stuff['train_source_loss'] = train_source_loss

        if validation_total_loss is not None:
            logging_stuff['validation_total_loss'] = validation_total_loss
        if validation_level_loss is not None:
            logging_stuff['validation_level_loss'] = validation_level_loss
        if validation_source_loss is not None:
            logging_stuff['validation_source_loss'] = validation_source_loss

        wandb.log(logging_stuff, step=epoch, commit=True)
    else:
        raise NotImplementedError('Not supported logging')

def log_best_stuff(
    options,
    best_metric,
):
    if options.use_log_type is None:    
        return None

    if options.use_log_type == 'tensorboard':
        pass
    elif options.use_log_type == 'wandb':
        wandb.run.summary["best_metric"] = best_metric
    else:
        raise NotImplementedError('Not supported logging')

def init_tensorboard(name="", base_dir="./tensorboard"):
    return SummaryWriter(os.path.join(name, base_dir))

def write_tensorboard(
    writer,
    epoch,
    # grad_norm,
    train_loss,
    train_symbol_accuracy,
    train_sentence_accuracy,
    train_wer,
    validation_loss,
    validation_symbol_accuracy,
    validation_sentence_accuracy,
    validation_wer,
    model,
):
    writer.add_scalar("train_loss", train_loss, epoch)
    writer.add_scalar("train_symbol_accuracy", train_symbol_accuracy, epoch)
    writer.add_scalar("train_sentence_accuracy",train_sentence_accuracy,epoch)
    writer.add_scalar("train_wer", train_wer, epoch)
    writer.add_scalar("validation_loss", validation_loss, epoch)
    writer.add_scalar("validation_symbol_accuracy", validation_symbol_accuracy, epoch)
    writer.add_scalar("validation_sentence_accuracy",validation_sentence_accuracy,epoch)
    writer.add_scalar("validation_wer",validation_wer,epoch)
    # writer.add_scalar("grad_norm", grad_norm, epoch)

    for name, param in model.encoder.named_parameters():
        writer.add_histogram(
            "encoder/{}".format(name), param.detach().cpu().numpy(), epoch
        )
        if param.grad is not None:
            writer.add_histogram(
                "encoder/{}/grad".format(name), param.grad.detach().cpu().numpy(), epoch
            )

    for name, param in model.decoder.named_parameters():
        writer.add_histogram(
            "decoder/{}".format(name), param.detach().cpu().numpy(), epoch
        )
        if param.grad is not None:
            writer.add_histogram(
                "decoder/{}/grad".format(name), param.grad.detach().cpu().numpy(), epoch
            )
