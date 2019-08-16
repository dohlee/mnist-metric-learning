import pandas as pd

from collections import defaultdict

class Logger(object):

    def __init__(self, log_file, log_loss_every):
        self.log = defaultdict(list)
        self.log_file = log_file
        self.log_loss_every = log_loss_every
        self.num_rows = 0

    def _pad(self, k):
        num_paddings = len(self.log['epoch']) - len(self.log[k])
        for _ in range(num_paddings):
            self.log[k].append(None)

    def update_log(self, **kwargs):
        # Append None to columns that are not updated.
        for col in self.log:
            if col not in kwargs:
                self.log[col].append(None)

        for col, v in kwargs.items():
            # Append to existing column.
            if col in self.log:
                self.log[col].append(v)
            # Add new column.
            else:
                self.log[col] = [None] * self.num_rows + [v]
        
        self.num_rows += 1
        pd.DataFrame(self.log).to_csv(self.log_file, sep='\t', index=False)