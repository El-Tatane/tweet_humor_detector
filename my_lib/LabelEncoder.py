from copy import deepcopy


class LabelEncoder:

    def __init__(self):
        self.label = ['pos', 'neu', 'neg', 'irr']
        self.enterprise = ['apl', 'msf', 'ggl', 'twt']

    def encode(self, array_like, type):
        if type == "label":
            tmp_list = self.label
        elif type == "enterprise":
            tmp_list = self.enterprise
        else:
            raise NotImplementedError()

        array_like_cp = deepcopy(array_like)
        for idx, value in enumerate(array_like_cp):
            array_like_cp[idx] = tmp_list.index(value)
        return array_like_cp

    def decode(self, int_list):
        raise NotImplementedError
        # a finir
        # if isinstance(int_list, int):
        #     int_list = [int_list]
        #
        # res = []
        #
        # for word in int_list:
        #     res.append()