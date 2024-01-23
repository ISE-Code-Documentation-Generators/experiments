class EarlyStopDetector:
    def __init__(self, model, threshold=0.02, allowed_chances=1):
        self.model = model
        self.threshold = threshold
        self.allowed_chances, self.chances = allowed_chances, 0
        self.last_bleu_metrics = None

    def majority(self, l):
        l.sort()
        return l[len(l)//2]
    
    def should_stop(self, bleu_metrics: dict):
        if self.last_bleu_metrics is None:
            self.last_bleu_metrics = bleu_metrics
            return False
        else:
            should_stop_dict = {}
            for key, last_result in self.last_bleu_metrics.items():
                should_stop_dict[key] = last_result - bleu_metrics[key] > self.threshold
            should_stop = self.majority(list(should_stop_dict.values()))
            if should_stop:
                self.chances += 1
                return self.chances >= self.allowed_chances
            else:
                self.chances = 0
        return False