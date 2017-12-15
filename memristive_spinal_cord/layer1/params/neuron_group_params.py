from memristive_spinal_cord.layer1.params.entity_params import EntityParams


class NeuronGroupParams(EntityParams):
    def __init__(self, model, number, params):
        self.model = model
        self.number = number
        self.params = params

    def to_nest_params(self):
        return dict(
            model=self.model,
            n=self.number,
            params=self.params,
        )
