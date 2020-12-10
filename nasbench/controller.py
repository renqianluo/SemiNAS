import os
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder
from decoder import Decoder
import utils


SOS_ID = 0
EOS_ID = 0

class NAO(nn.Module):
    def __init__(self,
                 encoder_layers,
                 decoder_layers,
                 mlp_layers,
                 hidden_size,
                 mlp_hidden_size,
                 vocab_size,
                 dropout,
                 source_length,
                 encoder_length,
                 decoder_length,
                 ):
        super(NAO, self).__init__()
        self.encoder = Encoder(
            encoder_layers,
            mlp_layers,
            hidden_size,
            mlp_hidden_size,
            vocab_size,
            dropout,
            source_length,
            encoder_length,
        )
        self.decoder = Decoder(
            decoder_layers,
            hidden_size,
            vocab_size,
            dropout,
            decoder_length,
        )

        self.flatten_parameters()
    
    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()
    
    def forward(self, input_variable, target_variable=None):
        encoder_outputs, encoder_hidden, arch_emb, predict_value = self.encoder(input_variable)
        decoder_hidden = (arch_emb.unsqueeze(0), arch_emb.unsqueeze(0))
        decoder_outputs, archs = self.decoder(target_variable, decoder_hidden, encoder_outputs)
        return predict_value, decoder_outputs, archs
    
    def generate_new_arch(self, input_variable, predict_lambda=1, direction='-'):
        encoder_outputs, encoder_hidden, arch_emb, predict_value, new_encoder_outputs, new_arch_emb, new_predict_value = self.encoder.infer(
            input_variable, predict_lambda, direction=direction)
        new_encoder_hidden = (new_arch_emb.unsqueeze(0), new_arch_emb.unsqueeze(0))
        decoder_outputs, new_archs = self.decoder(None, new_encoder_hidden, new_encoder_outputs)
        return new_archs, new_predict_value
