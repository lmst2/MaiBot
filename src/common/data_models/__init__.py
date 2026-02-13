import copy


class BaseDataModel:
    def deepcopy(self):
        return copy.deepcopy(self)
