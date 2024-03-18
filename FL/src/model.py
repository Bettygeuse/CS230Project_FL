import torch
import torch.nn as nn

class RNNPredictor(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, use_cuda):
        """Define layers for a vanilla rnn decoder"""
        super(RNNPredictor, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        self.rnn = nn.GRU(input_size=input_size
            ,hidden_size=hidden_size,num_layers=2, dropout=0.15, batch_first=True, bidirectional=True)
        self.out = nn.Linear(hidden_size * 2 * 40, output_size * 10)
        #self.log_softmax = nn.LogSoftmax()  # work with NLLLoss = CrossEntropyLoss
        self.tanh = nn.Tanh()
        self.use_cuda = use_cuda

    def forward(self, inputs, targets):
        if self.use_cuda:
            inputs = inputs.cuda()
        batch_size = inputs.size(0)
        rnn_out, _ = self.rnn(inputs, None)
        #print(rnn_out.size())
        fc_out = self.out(rnn_out.contiguous().view(batch_size, -1))
        outputs = self.tanh(fc_out)
        outputs = outputs.view(-1, 10, self.output_size)
        return outputs