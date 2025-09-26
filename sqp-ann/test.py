import logging
import dotenv
import hydra

dotenv.load_dotenv(override=True)
logging.getLogger('numexpr.utils').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config/", config_name="train.yaml")
def main(config):
    from torch.utils.data import DataLoader
    from accelerate import Accelerator
    from sqp_ann.utils import seed_everything, register_resolvers, pretty_configs, model_summary

    accelerator = Accelerator()

    # preamble
    seed_everything(config.seed)
    register_resolvers()
    logger.info(f"Current configs:\n{pretty_configs(config)}")

    # instantiate test dataset and dataloaders
    logger.info(f"Initializing test dataset {config.dataset_test._target_}")
    dataset_test = hydra.utils.instantiate(config.dataset_test)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=config.valid_batch_size, num_workers=config.workers, shuffle=False)

    # instantiate callbacks
    callbacks = []
    for callback_name in config.callbacks:
        logger.info(f"Initializing callback {config.callbacks[callback_name]._target_}")
        curr_callback = hydra.utils.instantiate(config.callbacks[callback_name])
        callbacks.append(curr_callback)

    # instantiate model
    logger.info(f"Initializing model {config.model._target_}")
    model = hydra.utils.instantiate(config.model)
    logger.info(f"Model architecture:\n{model_summary(model, dataloader_test)}")
    assert config.checkpoint_path, "Missing checkpoint path"
    logger.info(f"Loading checkpoint from {config.checkpoint_path}")

    # instantiate trainer
    logger.info(f"Initializing trainer {config.trainer._target_}")
    trainer = hydra.utils.instantiate(config.trainer, all_config=config, accelerator=accelerator, callbacks=callbacks, _recursive_=False)

    # test
    trainer.test(
        model=model, 
        dataloader_test=dataloader_test, 
        checkpoint_path=config.checkpoint_path)

if __name__ == "__main__":
    main()
