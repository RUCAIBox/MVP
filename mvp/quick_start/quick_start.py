import torch
import logging
from logging import getLogger
from mvp.utils import init_logger, get_model, get_trainer, init_seed
from mvp.config import Config
from mvp.data import data_preparation


def run_mvp(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str): model name
        dataset (str): dataset name
        config_file_list (list): config files used to modify experiment parameters
        config_dict (dict): parameters dictionary used to modify experiment parameters
        saved (bool): whether to save the model
    """

    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)

    init_seed(config['seed'], config['reproducibility'])
    # logger initialization

    init_logger(config)
    logger = getLogger()
    logger.info(config)
    logger.setLevel(logging.INFO)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config)

    # model loading and initialization
    model = get_model(config['model'])(config, train_data).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model, len(train_data))

    if config['test_only']:
        logger.info('Test only')
        test_result = trainer.evaluate(test_data, load_best_model=saved, model_file=config['load_experiment'])
    else:
        if config['load_experiment'] is not None:
            trainer.resume_checkpoint(resume_file=config['load_experiment'])
        # model training
        test_result = trainer.fit(train_data, valid_data, test_data, saved=saved)

    logger.info('\nbest result: {}'.format(test_result))
    return test_result