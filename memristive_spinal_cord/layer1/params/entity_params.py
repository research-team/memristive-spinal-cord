class EntityParams:
    def to_nest_params(self):
        raise NotImplementedError("Using of abstract method of " + self.__name__)
