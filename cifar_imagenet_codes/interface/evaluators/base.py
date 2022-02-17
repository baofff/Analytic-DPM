

class Evaluator(object):
    def __init__(self, options: dict):
        r""" Evaluate models
        """
        self.options = options

    def evaluate_train(self, it):
        r""" Evaluate during training
        Args:
            it: the iteration of training
        """
        for fn, val in self.options.items():
            period = val["period"]
            kwargs = val.get("kwargs", {})
            if it % period == 0:
                eval("self.%s" % fn)(it=it, **kwargs)

    def evaluate(self, it=None):
        r"""
        Args:
            it: the iteration when the evaluated models is saved
        """
        if it is None:
            it = 0
        for fn, val in self.options.items():
            kwargs = val.get("kwargs", {})
            eval("self.%s" % fn)(it=it, **kwargs)
