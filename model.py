import torch
import torch.nn as nn
import torch.nn.functional as F

from weight_drop_lstm import WeightDropLSTM


class WDropModel(nn.Module):
    def __init__(self, config):
        super(WDropModel, self).__init__()
        dataset = config.dataset
        target_class = config.target_class
        self.mode = config.mode
        self.wdrop_lstm = WeightDropLSTM(config.words_num, config.words_dim, config.hidden_dim, config.num_layers, config.dropout)
        self.hidden = self.model.init_hidden(config.batch_size)

        if self.is_bidirectional:
            self.fc1 = nn.Linear(2 * config.hidden_dim, target_class)
        else:
            self.fc1 = nn.Linear(config.hidden_dim, target_class)

    @staticmethod
    def repackage_hidden(hidden):
        """
        Wraps hidden states in new Tensors to detach them from their history.
        :return:
        """
        if isinstance(hidden, torch.Tensor):
            return hidden.detach()
        else:
            return tuple(WDropModel.repackage_hidden(v) for v in hidden)

    def forward(self, x, lengths=None):
        if lengths is not None:
            x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        self.hidden = WDropModel.repackage_hidden(self.hidden)
        output, self.hidden, rnn_hs, dropped_rnn_hs = self.wdrop_lstm(x, self.hidden, return_h=True)
        if lengths is not None:
            x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = F.relu(torch.transpose(x, 1, 2))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        logit = self.fc1(x) # (batch, target_size)
        return logit
