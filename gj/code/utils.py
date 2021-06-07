import torch.optim as optim
from adamp import (
    AdamP,
    SGDP,
)

from scheduler import CircularLRBeta

from networks.Attention import Attention
from networks.SATRN import SATRN


def get_network(
    model_type,
    FLAGS,
    model_checkpoint,
    device,
    train_dataset,
):
    model = None

    if model_type == "Attention":
        model = Attention(FLAGS, train_dataset, checkpoint=model_checkpoint,
             device=device).to(device)
    elif model_type == "SATRN":
        model = SATRN(FLAGS, train_dataset, checkpoint=model_checkpoint,
         device=device).to(device)

        if FLAGS.SATRN.flexible_stn.use and FLAGS.SATRN.flexible_stn.train_stn_only:
            for param in model.encoder.parameters():
                param.requires_grad_ = False
            for param in model.decoder.parameters():
                param.requires_grad_ = False
    else:
        raise NotImplementedError

    return model

def get_optimizer(optimizer, params, lr, weight_decay=None):
    if optimizer == "Adam":
        optimizer = optim.Adam(params, lr=lr)
    elif optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=lr, betas=(0.9, 0.999),
             eps=1e-08, weight_decay=weight_decay, amsgrad=False)
    elif optimizer == "Adadelta":
        optimizer = optim.Adadelta(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == 'AdamP':
        optimizer = AdamP(params, lr=lr, betas=(0.9, 0.999),
            weight_decay=weight_decay)
    elif optimizer == 'SGDP':
        optimizer = SGDP(params, lr=lr, weight_decay=weight_decay,
             momentum=0.9, nesterov=True)
    else:
        raise NotImplementedError
    return optimizer

def get_enc_dec_optimizer(options, enc_params, dec_params):
    enc_optimizer = get_optimizer(
        options.optimizer.encoder.type,
        enc_params,
        lr=options.optimizer.encoder.lr,
        weight_decay=options.optimizer.encoder.weight_decay,
    )
    dec_optimizer = get_optimizer(
        options.optimizer.decoder.type,
        dec_params,
        lr=options.optimizer.decoder.lr,
        weight_decay=options.optimizer.decoder.weight_decay,
    )

    return enc_optimizer, dec_optimizer

def setup_optimizer(optimizer, lr,  optimizer_state):
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)
    for param_group in optimizer.param_groups:
        param_group["initial_lr"] = lr

def setup_enc_dec_optimizer(options, checkpoint,
         enc_optimizer, dec_optimizer):
    enc_optimizer_state = checkpoint.get("enc_optimizer")
    dec_optimizer_state = checkpoint.get("dec_optimizer")
    setup_optimizer(enc_optimizer, options.optimizer.encoder.lr, enc_optimizer_state)
    setup_optimizer(dec_optimizer, options.optimizer.decoder.lr, dec_optimizer_state)

def get_lr_scheduler(options, options_optimizer, optimizer, data_loader):
    if options_optimizer.lr_scheduler == 'CircularLRBeta':
        cycle = len(data_loader) * options.num_epochs
        lr_scheduler = CircularLRBeta(
            optimizer, options_optimizer.lr, 10, 10, cycle, [0.95, 0.85]
        )
    elif options_optimizer.lr_scheduler == 'StepLR':
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=options_optimizer.lr_epochs,
            gamma=options_optimizer.lr_factor,
        )
    elif options_optimizer.lr_scheduler == 'OneCycleLR':
        lr_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=options_optimizer.lr,
            steps_per_epoch=len(data_loader),
            epochs=options_optimizer.lr_epochs,
            pct_start=options_optimizer.pct_start,
        )
    elif options_optimizer.lr_scheduler == 'Same':
        lr = options_optimizer.lr
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=options_optimizer.lr_epochs, gamma=0.1
        )
    else:
        raise NotImplementedError(options_optimizer.lr_scheduler)
        
    return lr_scheduler

def get_enc_dec_lr_scheduler(options, enc_optimizer, dec_optimizer, train_data_loader,
    is_first=False):
    """ is_first curriculm 용"""
    if is_first:
        enc_option = options.optimizer.first_encoder
        dec_option = options.optimizer.first_decoder
    else:
        enc_option = options.optimizer.encoder
        dec_option = options.optimizer.decoder

    enc_lr_scheduler = get_lr_scheduler(
        options, enc_option, enc_optimizer, train_data_loader
    )

    dec_lr_scheduler = get_lr_scheduler(
        options, dec_option, dec_optimizer, train_data_loader
    )

    return enc_lr_scheduler, dec_lr_scheduler
