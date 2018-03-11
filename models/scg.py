from abc import ABCMeta, abstractmethod, abstractproperty

class SCGMeta(type):
    def __call__(cls, *args, **kwargs):
        """Called when you call Foo(*args, **kwargs) """
        obj = type.__call__(cls, *args, **kwargs)
        obj.check_obj()
        return obj


class SCG:
    
    __metaclass__ = SCGMeta
    config = None
    X = None
    y = None
    init = None
    saver = None
    fetches = None
    opt = None

    def __init__(self):
        pass

    @abstractmethod
    def define(self):
        #method where forward pass of the graph is defined
        pass

    def check_obj(self):
        if self.config is None:
            raise NotImplementedError('Subclasses must define config')
        if self.X is None:
            raise NotImplementedError('Subclasses must define X')
        if self.y is None:
            raise NotImplementedError('Subclasses must define y')
        if self.init is None:
            raise NotImplementedError('Subclasses must define init')
        if self.saver is None:
            raise NotImplementedError('Subclasses must define saver')
        if self.fetches is None:
            raise NotImplementedError('Subclasses must define fetches')
        if self.opt is None:
            raise NotImplementedError('Subclasses must define opt')
