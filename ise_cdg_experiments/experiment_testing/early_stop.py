
class MetricBasedEarlyStop:
    def __init__(self):
        self.best_metric_set: dict = None

    def majority(self, l: list):
        l.sort()
        return l[len(l)//2]

    def __call__(self, metric_set: dict):
        # Returns True if Model Should Continue Training
        if self.best_metric_set is None:
            self.best_metric_set = metric_set
            return True
        else:
            should_stop_dict = {}
            for key, best_result in self.best_metric_set.items():
                should_stop_dict[key] = bool(best_result < metric_set[key])
            should_save = self.majority(list(should_stop_dict.values()))
            if should_save:
                self.best_metric_set = metric_set
            return should_save
