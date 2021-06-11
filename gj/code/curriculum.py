from train_method import run_trial

def run_curriculm(
    options,
    train_data_loader_maker,
    validation_data_loader_maker,
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
    set_optimizer_from_checkpoint=True,
):
    is_first = True
    for level in range(options.curriculum_learning.max_level):
        start_epoch = level * options.num_epochs
        num_epochs = options.num_epochs

        train_data_loader = train_data_loader_maker.get_level_data_loader(level)
        validation_data_loader = validation_data_loader_maker.get_lower_level_loader(level)
        run_trial(
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
            set_optimizer_from_checkpoint=False,
            level_info=f'Level {level+1} ',
            is_first=is_first,
        )

        is_first = False