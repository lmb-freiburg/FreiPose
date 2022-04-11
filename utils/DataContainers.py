
def _check_str(txt):
    assert type(txt) == str, 'String type mismatch.'
    name_replace_list = [
        ('\n', ''),
        ('\r', ''),
        (' ', '_'),
    ]
    for a, b in name_replace_list:
        txt = txt.replace(a, b)
    return txt


def _check_color(color):
    if type(color) == list:
        color = tuple(color)
    assert type(color) == tuple, 'Color type mismatch.'
    assert len(color) == 3, 'Color shape mismatch.'
    assert all([0 <= c <= 255 for c in color]), 'Color value range mismatch.'
    return color


class Keypoint(object):
    def __init__(self, name, color):
        name, color = self.check_input(name, color)
        self.name = name
        self.color = color

    def serialize(self):
        return '{name: %s, color: %s}' % (self.name, self.color)

    @staticmethod
    def check_input(name, color):
        # Check validity of input quantities
        name = _check_str(name)
        color = _check_color(color)
        return name, color


class Limb(object):
    def __init__(self, kp_p, kp_c, color):
        kp_p, kp_c, color = self.check_input(kp_p, kp_c, color)
        self.kp_p = kp_p
        self.kp_c = kp_c
        self.color = color

    @staticmethod
    def check_input(kp_p, kp_c, color):
        # Check validity of input quantities
        assert type(kp_p) == Keypoint, 'Parent keypoint type mismatch.'
        assert type(kp_c) == Keypoint, 'Child keypoint type mismatch.'
        color = _check_color(color)
        return kp_p, kp_c, color


class Dataset(object):
    def __init__(self, kp_p, kp_c, color):
        kp_p, kp_c, color = self.check_input(kp_p, kp_c, color)
        self.kp_p = kp_p
        self.kp_c = kp_c
        self.color = color
