from .model import Model


class PreferenceModel(Model):
    """Model concept promotion/suppression as preference."""

    # the base class for all preference models
    preference_pairs = ["orig_add"] # "orig_add", "orig_sub", "steered_add", "steered_sub"
    def __str__(self):
        return 'PreferenceModel'

    def train(self, examples, **kwargs):
        raise NotImplementedError(f"Training is not supported for {self.__str__()}.")
