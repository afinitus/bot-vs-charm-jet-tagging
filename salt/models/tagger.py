import torch.nn as nn

# these need to be directly imported from their modules
from salt.models.dense import Dense
from salt.models.pooling import Pooling
from salt.models.task import Task


class JetTagger(nn.Module):
    def __init__(
        self,
        init_net: Dense = None,
        gnn: nn.Module = None,
        pool_net: Pooling = None,
        jet_net: Task = None,
        track_net: Task = None,
    ):
        """Jet constituent tagger model.

        # TODO: add option to pool separately for each task

        Parameters
        ----------
        init_net : Dense
            Initialisation network
        gnn : nn.Module
            Graph neural network
        pool_net : nn.Module
            Pooling network
        jet_net : Task
            Jet classification task
        track_net : Task
            Track classification task
        """
        super().__init__()

        self.init_net = init_net
        self.gnn = gnn
        self.jet_net = jet_net
        self.pool_net = pool_net
        self.track_net = track_net

    def forward(self, x, mask, labels):
        mask[..., 0] = False  # hack to make the MHA work
        embd_x = self.init_net(x)
        if self.gnn:
            embd_x = self.gnn(embd_x, mask=mask)
        pooled = self.pool_net(embd_x, mask=mask)

        # run tasks
        preds, loss = self.tasks(pooled, embd_x, mask, labels)

        return preds, loss

    def tasks(self, pooled, embd_x, mask, labels):
        preds = {}
        loss = {}

        for task in self.get_tasks():
            inputs = pooled if "jet" in task.name else embd_x  # TODO: make robust
            p, subloss = task(inputs, labels[task.name] if labels is not None else None)
            preds[task.name] = p
            loss[task.name] = subloss

        return preds, loss

    def get_tasks(self):
        tasks = []
        for n in dir(self):
            task = getattr(self, n)
            if not isinstance(task, Task) or task is None:
                continue
            tasks.append(task)
        return tasks
