
# Technique Early Stopping

import copy

class EarlyStopping:
    def __init__(self, patience: int=3, min_delta: float=1e-3, mode="min") -> None:
        self.patience = patience
        self.min_delta = min_delta
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.mode = mode

        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.early_stop = False

        self.best_model_state = None

    def __call__(self,model, score: float):
        improved = (
            score < self.best_score - self.min_delta
            if self.mode == "min"
            else score > self.best_score + self.min_delta
        )
        if improved:
            self.best_score = score
            self.counter = 0
            self.best_model_state = copy.deepcopy(model.state_dict()) # lo que hace deepcopy es crear una copia completa del objeto, incluyendo todos los objetos anidados, en lugar de solo copiar las referencias a esos objetos. 
            #Esto significa que cualquier cambio realizado en la copia no afectará al objeto original y viceversa.
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)