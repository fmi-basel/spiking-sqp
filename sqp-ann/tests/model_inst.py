import logging
import dotenv
import hydra

dotenv.load_dotenv(override=True)
logging.getLogger('numexpr.utils').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config/", config_name="train.yaml")
def main(config):
    from torch.utils.data import DataLoader
    from accelerate import Accelerator
    from sqp_ann.utils import seed_everything, register_resolvers, pretty_configs

    accelerator = Accelerator()

    # preamble
    seed_everything(config.seed)
    register_resolvers()
    logger.info(f"Current configs:\n{pretty_configs(config)}")

    # instantiate validation dataset and dataloaders
    logger.info(f"Initializing validation dataset {config.dataset._target_}")
    dataset = hydra.utils.instantiate(config.dataset)
    dataloader = DataLoader(dataset=dataset, batch_size=config.valid_batch_size, num_workers=config.workers, shuffle=False)

    # instantiate model
    logger.info(f"Initializing model {config.model._target_}")
    model = hydra.utils.instantiate(config.model)

    # run forward step
    dummy_data = next(iter(dataloader))
    dummy_preproc = model.preproc(dummy_data[0])
    pred = model.forward(dummy_preproc)

    # run train step
    pred = model.train_step(*dummy_data)

    # run valid step
    pred = model.valid_step(*dummy_data)



if __name__ == "__main__":
    main()