
class AttributeDict(dict):
    __slots__ = ()
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


math = AttributeDict({
    "e": 2.718281828459045
})
CCE = "categorical_crossentropy"

