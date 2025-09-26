import logging
import dotenv
import hydra
import os

dotenv.load_dotenv(override=True)
logging.getLogger('numexpr.utils').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config/", config_name="train.yaml")
def main(config):
    from torch.utils.data import DataLoader
    from accelerate import Accelerator
    from sqp_ann.utils import seed_everything, register_resolvers, pretty_configs, model_summary, train_valid_split

    # Set CUDA_VISIBLE_DEVICES before initializing Accelerator
    if 'gpu' in config and config.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)
        logger.info(f"Using GPU: {config.gpu}")
    else:
        logger.info("No specific GPU selected, using default")

    accelerator = Accelerator()

    # preamble
    seed_everything(config.seed)
    register_resolvers()
    logger.info(f"Current configs:\n{pretty_configs(config)}")

    # instantiate dataset
    logger.info(f"Initializing training/validation dataset {config.dataset._target_}")
    dataset = hydra.utils.instantiate(config.dataset)

    # split into train/validation and instantiate dataloaders
    dataset_train, dataset_valid = train_valid_split(dataset=dataset, readers_file=config.readers_file, seed=config.seed, valid_split=config.train_valid_split)
    logger.info(f"Splitting dataset into training/validation sets: {len(dataset_train)} / {len(dataset_valid)}")
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=config.batch_size, num_workers=config.workers, shuffle=True)
    dataloader_valid = DataLoader(dataset=dataset_valid, batch_size=config.valid_batch_size, num_workers=config.workers, shuffle=False)

    # instantiate callbacks
    callbacks = []
    for callback_name in config.callbacks:
        logger.info(f"Initializing callback {config.callbacks[callback_name]._target_}")
        curr_callback = hydra.utils.instantiate(config.callbacks[callback_name])
        callbacks.append(curr_callback)

    # instantiate model
    logger.info(f"Initializing model {config.model._target_}")
    model = hydra.utils.instantiate(config.model)
    logger.info(f"Model architecture:\n{model_summary(model, dataloader_valid)}")


    # instantiate trainer
    logger.info(f"Initializing trainer {config.trainer._target_}")
    trainer = hydra.utils.instantiate(config.trainer, all_config=config, accelerator=accelerator, callbacks=callbacks, _recursive_=False)

    # train
    if config.checkpoint_path:
        logger.info(f"Loading checkpoint from {config.checkpoint_path}")
    trainer.train(model=model, dataloader_train=dataloader_train, dataloader_valid=dataloader_valid, checkpoint_path=config.checkpoint_path)

    if config.run_test:
        # find and load best checkpoint
        logger.info("Finding best checkpoint...")
        checkpoint_callbacks = [c for c in callbacks if hasattr(c, 'best_path')]
        assert len(checkpoint_callbacks) <= 1, "There appear to be several checkpoint callbacks"
        if len(checkpoint_callbacks) == 1 and checkpoint_callbacks[0].best_path is not None:
            checkpoint_best_path = checkpoint_callbacks[0].best_path
            logger.info(f"Best checkpoint found at {checkpoint_best_path}")
        else:
            checkpoint_best_path = None
            logger.info("Best checkpoint not found! Using current state")

        # instantiate test dataset and dataloaders
        logger.info(f"Initializing test dataset {config.dataset_test._target_}")
        dataset_test = hydra.utils.instantiate(config.dataset_test)
        dataloader_test = DataLoader(dataset=dataset_test, batch_size=config.valid_batch_size, num_workers=config.workers, shuffle=False)

        # test
        trainer.test(
            model=model, 
            dataloader_test=dataloader_test, 
            checkpoint_path=checkpoint_best_path)

if __name__ == "__main__":
    main()
