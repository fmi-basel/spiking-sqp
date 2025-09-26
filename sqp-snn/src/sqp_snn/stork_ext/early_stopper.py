class EarlyStopper:
    def __init__(self,
                 patience = 10,
                 min_delta = 0.0,
                 mode = 'min',
                 verbose = True):
        
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.reset()
        
    def reset(self):
        self.best_metric = float('inf') if self.mode == 'min' else float('-inf')
        self.counter = 0
        self.best_state = None
        self.stop_training = False
        
    def __call__(self, current_metric, model):
        improved = ((self.mode == 'min' and current_metric < self.best_metric - self.min_delta) or
                    (self.mode == 'max' and current_metric > self.best_metric + self.min_delta))
        
        if improved:
            if self.verbose:
                print(f'Validation metric improved from {self.best_metric:.6f} to {current_metric:.6f}')
            self.best_metric = current_metric
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'Early-stopping counter: {self.counter}/{self.patience}')
            
        self.stop_training = self.counter >= self.patience
        return self.stop_training
    
    def restore_best_state(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
