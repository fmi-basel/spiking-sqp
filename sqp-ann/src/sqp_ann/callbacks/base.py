class BaseCallback():
    def __init__(self):
        pass
    
    # trainer init
    def on_init(self, trainer):
        self.trainer = trainer


    # during training
    def on_training_start(self, model, dataloader_train, dataloader_valid):
        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_valid = dataloader_valid

    def on_training_epoch_end(self, **kw_args):
        pass

    def on_training_end(self, **kw_args):
        pass


    # during validation
    def on_validation_start(self, **kw_args):
        pass

    def on_validation_end(self, **kw_args):
        pass

    # during testing
    def on_testing_start(self, model, dataloader_test, **kw_args):
        self.model = model
        self.dataloader_test = dataloader_test

    def on_testing_end(self, **kw_args):
        pass
