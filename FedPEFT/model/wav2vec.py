# part of the code was referenced from SUPERB: https://github.com/s3prl/s3prl
# and https://github.com/wngh1187/IPET/blob/main/Speechcommands_V2/W2V2/models/W2V2.py

##【！！！重要】程序无法访问hugggingface.co时使用替代方案
##chenhj
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import os
import pdb
import copy
import torch
import argparse
import numpy as np
import loralib as lora
import transformers.models.wav2vec2.modeling_wav2vec2 as w2v2

from functools import lru_cache
from torchaudio.compliance import kaldi

from torch import nn
from adapter import Adapter
from collections import OrderedDict
from typing import Optional, Callable
from torch.nn import functional as F
from torch.nn.functional import normalize
from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2Processor, AutoProcessor, WavLMModel, WhisperModel, AutoFeatureExtractor


class Wav2Vec2EncoderLayer(nn.Module):
    def __init__(
        self, 
        config, 
        i
    ):
        super().__init__()
        self.attention = w2v2.Wav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = w2v2.Wav2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.config = config
        
        if self.config.finetune_method == "embedding_prompt" or self.config.finetune_method == "combined":
            self.embed_prompt = nn.Parameter(torch.randn([1, self.config.embedding_prompt_dim, 768]))
            nn.init.xavier_uniform_(self.embed_prompt)
        if self.config.finetune_method == "lora" or self.config.finetune_method == "combined":
            self.feed_forward.intermediate_dense    = lora.Linear(config.hidden_size, config.intermediate_size, r=config.lora_rank)
            self.feed_forward.output_dense          = lora.Linear(config.intermediate_size, config.hidden_size, r=config.lora_rank)
            
        if self.config.finetune_method == "adapter" or self.config.finetune_method == "adapter_l" or self.config.finetune_method == "combined":
            self.adapter = Adapter(
                config, 
                dropout=0.1, 
                bottleneck=config.adapter_hidden_dim, 
                adapter_scalar=0.1
            )
        self.i = i

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        if self.config.finetune_method == "embedding_prompt" or self.config.finetune_method == "combined":
            hidden_states = torch.cat((self.embed_prompt.repeat(hidden_states.size(0), 1, 1), hidden_states), dim=1)
        attn_residual = hidden_states
        
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        # Adapter
        if self.config.finetune_method == "adapter":
            adapt_h = self.adapter(hidden_states)

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states) 
        # Adapter
        if self.config.finetune_method == "adapter": 
            hidden_states = hidden_states+ adapt_h
        if self.config.finetune_method == "adapter_l" or self.config.finetune_method == "combined": 
            hidden_states = hidden_states + self.adapter(hidden_states)
            
        hidden_states = self.final_layer_norm(hidden_states)
        if self.config.finetune_method == "embedding_prompt" or self.config.finetune_method == "combined":
            hidden_states = hidden_states[:, self.config.embedding_prompt_dim:, :]

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)
        return outputs

class Wav2VecWrapper(nn.Module):
    def __init__(
            self,
            args,
            hidden_dim=256,
            output_class_num=4
    ):
        super(Wav2VecWrapper, self).__init__()
        # 1. We Load the model first with weights
        self.args = args
        # chenhj
        # 每次强制重新下载：force_download=True, resume_download=False
        self.backbone_model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base-960h",
            output_hidden_states=True
            # force_download=True,
            # resume_download=False
        )
        state_dict = self.backbone_model.state_dict()
        # 2. Read the model config
        self.model_config = self.backbone_model.config
        self.model_config.finetune_method = args.finetune_method
        self.model_config.adapter_hidden_dim = args.adapter_hidden_dim
        self.model_config.embedding_prompt_dim = args.embedding_prompt_dim
        self.model_config.lora_rank = args.lora_rank

        # 3. Config encoder layers with adapter or embedding prompt
        # pdb.set_trace()
        self.backbone_model.encoder.layers = nn.ModuleList(
            [Wav2Vec2EncoderLayer(self.model_config, i) for i in range(self.model_config.num_hidden_layers)])
        # 4. Load the weights back
        msg = self.backbone_model.load_state_dict(state_dict, strict=False)
        # 5. Freeze the weights
        if self.args.finetune_method == "adapter" or self.args.finetune_method == "adapter_l" or self.args.finetune_method == "embedding_prompt" or self.args.finetune_method == "finetune" or self.args.finetune_method == "lora" or self.args.finetune_method == "combined":
            for name, p in self.backbone_model.named_parameters():
                if name in msg.missing_keys:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        self.finetune_method = self.args.finetune_method

        # 6. Downstream models
        self.model_seq = nn.Sequential(
            nn.Conv1d(self.model_config.hidden_size, hidden_dim, 1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(hidden_dim, hidden_dim, 1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(hidden_dim, hidden_dim, 1, padding=0)
        )
        self.weights = nn.Parameter(torch.zeros(self.model_config.num_hidden_layers))

        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_class_num),
        )

    def forward(self, x, length=None):

        device = torch.device('cuda:2')

        # 1. feature extraction and projections
        with torch.no_grad():
            x = self.backbone_model.feature_extractor(x)
            x = x.transpose(1, 2)  # New version of huggingface
            x, _ = self.backbone_model.feature_projection(x)  # New version of huggingface

        # 2. get length and mask
        if length is not None:
            length = self.get_feat_extract_output_lengths(length.detach().cpu())
            length = length.cuda()

        # 3. transformer encoding features
        x = self.backbone_model.encoder(
            x, output_hidden_states=True
        ).hidden_states

        # 4. stacked feature
        stacked_feature = torch.stack(x, dim=0)[1:]

        # 5. Weighted sum
        _, *origin_shape = stacked_feature.shape
        # Return transformer enc outputs [num_enc_layers, B, T, D]
        stacked_feature = stacked_feature.view(self.backbone_model.config.num_hidden_layers, -1)
        norm_weights = F.softmax(self.weights, dim=-1)

        # Perform weighted average
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        features = weighted_feature.view(*origin_shape)

        # 6. Pass the weighted average to point-wise 1D Conv
        # B x T x D
        features = features.transpose(1, 2)
        features = self.model_seq(features)
        features = features.transpose(1, 2)

        # 7. Pooling
        if length is not None:
            # chenhj
            features = features.to(device)
            length = length.to(device)
            # masks = torch.arange(features.size(1)).expand(length.size(0), -1).cuda() < length.unsqueeze(1)
            masks = torch.arange(features.size(1), device=device).expand(length.size(0), -1) < length.unsqueeze(1)
            masks = masks.float()
            features = (features * masks.unsqueeze(-1)).sum(1) / length.unsqueeze(1)
        else:
            features = torch.mean(features, dim=1)

        # 8. Output predictions
        # B x D
        predicted = self.out_layer(features)
        return predicted

    # From huggingface
    def get_feat_extract_output_lengths(self, input_length):
        """
        Computes the output length of the convolutional layers
        """
        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1
        for kernel_size, stride in zip(self.backbone_model.config.conv_kernel, self.backbone_model.config.conv_stride):
            input_length = _conv_out_length(input_length, kernel_size, stride)
        return input_length

class ShallowModel(nn.Module):
    def __init__(self, args, num_shallow_layers=6):
        super(ShallowModel, self).__init__()
        self.args = args

        self.backbone_model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base-960h",
            output_hidden_states=True
            # force_download=True,
            # resume_download=False
        )
        state_dict = self.backbone_model.state_dict()
        # 2. Read the model config
        self.model_config = self.backbone_model.config
        self.model_config.finetune_method = args.finetune_method
        self.model_config.adapter_hidden_dim = args.adapter_hidden_dim
        self.model_config.embedding_prompt_dim = args.embedding_prompt_dim
        self.model_config.lora_rank = args.lora_rank

        self.num_shallow_layers = num_shallow_layers

        # Only use the feature extractor, projection and shallow encoder layers
        self.feature_extractor = self.backbone_model.feature_extractor
        self.feature_projection = self.backbone_model.feature_projection
        # self.positional_embedding = self.backbone_model.encoder.pos_conv_embed
        # self.layer_norm = nn.LayerNorm(768, eps=1e-5)
        # self.dropout = nn.Dropout(p=0.1)
        self.shallow_encoder = nn.ModuleList(
            [Wav2Vec2EncoderLayer(self.model_config, i) for i in range(self.num_shallow_layers)]
        )
        self.backbone_model.encoder.layers = self.shallow_encoder

        msg = self.backbone_model.load_state_dict(state_dict, strict=False)
        # 5. Freeze the weights
        if self.args.finetune_method == "adapter" or self.args.finetune_method == "adapter_l" or self.args.finetune_method == "embedding_prompt" or self.args.finetune_method == "finetune" or self.args.finetune_method == "lora" or self.args.finetune_method == "combined":
            for name, p in self.backbone_model.named_parameters():
                if name in msg.missing_keys:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        self.finetune_method = self.args.finetune_method

        self.weights = nn.Parameter(torch.zeros(self.num_shallow_layers))


    def forward(self, x, length=None):
        device = torch.device('cuda:2')

        # Feature extraction and projection
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)  # New version of huggingface
        x, _ = self.feature_projection(x)  # Feature projection
        # x = self.positional_embedding(x)
        # x = self.layer_norm(x)
        # x = self.dropout(x)

        # # Shallow encoder layers
        # for layer in self.shallow_encoder:
        #     x= layer(x, output_hidden_states=True)

        x = self.backbone_model.encoder(x, output_hidden_states=True).hidden_states

        stacked_feature = torch.stack(x, dim=0)[1:]

        _, *origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.view(self.num_shallow_layers, -1)
        norm_weights = F.softmax(self.weights, dim=-1)

        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        features = weighted_feature.view(*origin_shape)

        return features


class DeepModel(nn.Module):
    def __init__(self, args, hidden_dim=256, output_class_num=4, num_shallow_layers=6):
        super(DeepModel, self).__init__()
        self.args = args
        self.backbone_model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base-960h",
            output_hidden_states=True
            # force_download=True,
            # resume_download=False
        )
        state_dict = self.backbone_model.state_dict()
        # 2. Read the model config
        self.model_config = self.backbone_model.config
        self.model_config.finetune_method = args.finetune_method
        self.model_config.adapter_hidden_dim = args.adapter_hidden_dim
        self.model_config.embedding_prompt_dim = args.embedding_prompt_dim
        self.model_config.lora_rank = args.lora_rank

        self.num_shallow_layers = num_shallow_layers

        self.feature_extractor = self.backbone_model.feature_extractor
        self.feature_projection = self.backbone_model.feature_projection

        # Deep model includes the second half of the encoder layers and downstream layers
        self.deep_encoder = nn.ModuleList(
            [Wav2Vec2EncoderLayer(self.model_config, i) for i in range(self.model_config.num_hidden_layers - self.num_shallow_layers, self.model_config.num_hidden_layers)]
        )

        self.backbone_model.encoder.layers = self.deep_encoder

        msg = self.backbone_model.load_state_dict(state_dict, strict=False)
        # 5. Freeze the weights
        if self.args.finetune_method == "adapter" or self.args.finetune_method == "adapter_l" or self.args.finetune_method == "embedding_prompt" or self.args.finetune_method == "finetune" or self.args.finetune_method == "lora" or self.args.finetune_method == "combined":
            for name, p in self.backbone_model.named_parameters():
                if name in msg.missing_keys:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        self.finetune_method = self.args.finetune_method

        self.model_seq = nn.Sequential(
            nn.Conv1d(self.model_config.hidden_size, hidden_dim, 1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(hidden_dim, hidden_dim, 1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(hidden_dim, hidden_dim, 1, padding=0)
        )

        self.weights = nn.Parameter(torch.zeros(self.model_config.num_hidden_layers - self.num_shallow_layers))

        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_class_num),
        )

    def forward(self, x, shallow_features, length=None):
        # # Fusion of shallow features and current input
        # # Concatenate shallow features with the current input (e.g., from feature extractor)
        # x = torch.cat((x, shallow_features), dim=-1)  # Concatenate shallow features with current features

        # # Deep encoder layers (second half of the encoder)
        # for layer in self.deep_encoder:
        #     x, _ = layer(x)
        device = torch.device('cuda:2')

        x = self.feature_extractor(x)
        x = x.transpose(1, 2)  # New version of huggingface
        x, _ = self.feature_projection(x)  # Feature projection
        # Concatenate shallow features with the current input
        x = torch.cat((x, shallow_features), dim=-1)

        x = self.backbone_model.encoder(x, output_hidden_states=True).hidden_states

        # 4. stacked feature
        stacked_feature = torch.stack(x, dim=0)[1:]

        # 5. Weighted sum
        _, *origin_shape = stacked_feature.shape
        # Return transformer enc outputs [num_enc_layers, B, T, D]
        stacked_feature = stacked_feature.view(self.model_config.num_hidden_layers - self.num_shallow_layers, -1)
        norm_weights = F.softmax(self.weights, dim=-1)

        # Perform weighted average
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        features = weighted_feature.view(*origin_shape)

        # Pass the features through the downstream layers
        features = features.transpose(1, 2)
        features = self.model_seq(features)
        features = features.transpose(1, 2)

        # Pooling
        if length is not None:
            length = length.to(device)
            masks = torch.arange(features.size(1), device=device).expand(length.size(0), -1) < length.unsqueeze(1)
            masks = masks.float()
            features = (features * masks.unsqueeze(-1)).sum(1) / length.unsqueeze(1)
        else:
            features = torch.mean(features, dim=1)

        # Output predictions
        predicted = self.out_layer(features)
        return predicted

# class Wav2VecWrapper(nn.Module):
#     def __init__(self, args, hidden_dim=256, output_class_num=4):
#         super(Wav2VecWrapper, self).__init__()
#         self.args = args
#
#         # Load the pre-trained Wav2Vec2 model
#         self.backbone_model = Wav2Vec2Model.from_pretrained(
#             "facebook/wav2vec2-base-960h", output_hidden_states=True
#         )
#         state_dict = self.backbone_model.state_dict()
#
#         # Read the model config and configure the fine-tuning parameters
#         self.model_config = self.backbone_model.config
#         self.model_config.finetune_method = args.finetune_method
#         self.model_config.adapter_hidden_dim = args.adapter_hidden_dim
#         self.model_config.embedding_prompt_dim = args.embedding_prompt_dim
#         self.model_config.lora_rank = args.lora_rank
#
#         # 1. Shallow network: Includes feature extraction and first half of the encoder
#         self.feature_extractor = self.backbone_model.feature_extractor
#         self.feature_projection = self.backbone_model.feature_projection
#         self.shallow_encoder = nn.ModuleList(
#             [Wav2Vec2EncoderLayer(self.model_config, i) for i in range(self.model_config.num_hidden_layers // 2)]
#         )
#
#         # 2. Deep network: Includes the remaining encoder layers and downstream layers
#         self.deep_encoder = nn.ModuleList(
#             [Wav2Vec2EncoderLayer(self.model_config, i) for i in range(self.model_config.num_hidden_layers // 2, self.model_config.num_hidden_layers)]
#         )
#
#         msg = self.backbone_model.load_state_dict(state_dict, strict=False)
#
#         if self.args.finetune_method == "adapter" or self.args.finetune_method == "adapter_l" or self.args.finetune_method == "embedding_prompt" or self.args.finetune_method == "finetune" or self.args.finetune_method == "lora" or self.args.finetune_method == "combined":
#             for name, p in self.backbone_model.named_parameters():
#                 if name in msg.missing_keys:
#                     p.requires_grad = True
#                 else:
#                     p.requires_grad = False
#
#         # Downstream classification layers (after encoder)
#         self.model_seq = nn.Sequential(
#             nn.Conv1d(self.model_config.hidden_size, hidden_dim, 1, padding=0),
#             nn.ReLU(),
#             nn.Dropout(p=0.1),
#             nn.Conv1d(hidden_dim, hidden_dim, 1, padding=0),
#             nn.ReLU(),
#             nn.Dropout(p=0.1),
#             nn.Conv1d(hidden_dim, hidden_dim, 1, padding=0)
#         )
#
#         self.out_layer = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_class_num),
#         )
#
#         self.weights = nn.Parameter(torch.zeros(self.model_config.num_hidden_layers))
#
#     def forward(self, x, length=None):
#         device = torch.device('cuda:2')
#
#         # Step 1: Shallow feature extraction (feature_extractor + feature_projection)
#         x = self.feature_extractor(x)
#         x = x.transpose(1, 2)  # New version of huggingface
#         x, _ = self.feature_projection(x)  # Feature projection
#
#         # Step 2: Shallow Encoder layers (only first half of the encoder)
#         for layer in self.shallow_encoder:
#             shallow_features, _ = layer(x)
#
#         # Step 3: Pass shallow features to deep network (fusion with current input)
#         # You can concatenate or add shallow features to the current input (x)
#         # Fusion approach: concatenate shallow features with the current input
#         x = torch.cat((x, shallow_features), dim=-1)  # Example of feature fusion, adjust as needed
#
#         # Step 4: Deep Encoder layers (second half of the encoder)
#         for layer in self.deep_encoder:
#             x, _ = layer(x)
#
#         # Step 5: Pass the features through the downstream layers (e.g., Conv1d layers)
#         features = x.transpose(1, 2)
#         features = self.model_seq(features)
#         features = features.transpose(1, 2)
#
#         # Step 6: Pooling
#         if length is not None:
#             length = length.to(device)
#             masks = torch.arange(features.size(1), device=device).expand(length.size(0), -1) < length.unsqueeze(1)
#             masks = masks.float()
#             features = (features * masks.unsqueeze(-1)).sum(1) / length.unsqueeze(1)
#         else:
#             features = torch.mean(features, dim=1)
#
#         # Step 7: Output predictions
#         predicted = self.out_layer(features)
#         return predicted, shallow_features

# class Wav2VecWrapper(nn.Module):
#     def __init__(
#         self,
#         args,
#         hidden_dim=256,
#         output_class_num=4,
#         num_shallow_layers=6
#     ):
#         super(Wav2VecWrapper, self).__init__()
#         # 1. We Load the model first with weights
#         self.args = args
#         #chenhj
#         #每次强制重新下载：force_download=True, resume_download=False
#         self.backbone_model = Wav2Vec2Model.from_pretrained(
#             "facebook/wav2vec2-base-960h",
#             output_hidden_states=True
#             # force_download=True,
#             # resume_download=False
#         )
#         state_dict = self.backbone_model.state_dict()
#         # 2. Read the model config
#         self.model_config = self.backbone_model.config
#         self.model_config.finetune_method        = args.finetune_method
#         self.model_config.adapter_hidden_dim     = args.adapter_hidden_dim
#         self.model_config.embedding_prompt_dim   = args.embedding_prompt_dim
#         self.model_config.lora_rank              = args.lora_rank
#
#         self.num_shallow_layers = num_shallow_layers
#         self.num_deep_layers = self.model_config.num_hidden_layers - num_shallow_layers
#
#         self.shallow_layers = nn.ModuleList([
#             Wav2Vec2EncoderLayer(self.model_config, i)
#             for i in range(self.num_shallow_layers)
#         ])
#
#         self.deep_layers = nn.ModuleList([
#             Wav2Vec2EncoderLayer(self.model_config, i)
#             for i in range(self.num_shallow_layers, self.model_config.num_hidden_layers)
#         ])
#
#         # # 3. Config encoder layers with adapter or embedding prompt
#         # # pdb.set_trace()
#         # self.backbone_model.encoder.layers = nn.ModuleList([Wav2Vec2EncoderLayer(self.model_config, i) for i in range(self.model_config.num_hidden_layers)])
#         # 4. Load the weights back
#         msg = self.backbone_model.load_state_dict(state_dict, strict=False)
#
#         if self.args.finetune_method == "adapter" or self.args.finetune_method == "adapter_l" or self.args.finetune_method == "embedding_prompt" or self.args.finetune_method == "finetune" or self.args.finetune_method == "lora" or self.args.finetune_method == "combined":
#             for name, p in self.backbone_model.named_parameters():
#                 if name in msg.missing_keys: p.requires_grad = True
#                 else: p.requires_grad = False
#         self.finetune_method = self.args.finetune_method
#
#         # # chenhj 冻结浅层，解冻深层或其他特定模块
#         # if self.args.finetune_method in ["lora", "adapter", "combined"]:
#         #     for name, p in self.backbone_model.named_parameters():
#         #         if "shallow_layers" in name:
#         #             p.requires_grad = False
#         #         elif "deep_layers" in name:
#         #             p.requires_grad = True
#         #
#         # self.finetune_method = self.args.finetune_method
#
#         # #chenhj 分层划分
#         # num_shallow_layers = args.num_shallow_layers
#         # self.shallow_layers = nn.ModuleList(self.backbone_model.encoder.layers[:num_shallow_layers])
#         # self.deep_layers = nn.ModuleList(self.backbone_model.encoder.layers[num_shallow_layers:])
#
#         # 6. Downstream models
#         self.model_seq = nn.Sequential(
#             nn.Conv1d(self.model_config.hidden_size + hidden_dim, hidden_dim, 1, padding=0),
#             nn.ReLU(),
#             nn.Dropout(p=0.1),
#             nn.Conv1d(hidden_dim, hidden_dim, 1, padding=0),
#             nn.ReLU(),
#             nn.Dropout(p=0.1),
#             nn.Conv1d(hidden_dim, hidden_dim, 1, padding=0)
#         )
#         self.weights = nn.Parameter(torch.zeros(self.model_config.num_hidden_layers))
#
#         self.out_layer = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_class_num),
#         )
#
#     def forward(self, x, length=None):
#
#         device = torch.device('cuda:2')
#
#         # 1. feature extraction and projections
#         with torch.no_grad():
#             x = self.backbone_model.feature_extractor(x)
#             x = x.transpose(1, 2) # New version of huggingface
#             x, _ = self.backbone_model.feature_projection(x) # New version of huggingface
#
#         shallow_input = x
#
#         # 浅层编码器处理
#         for layer in self.shallow_layers:
#             shallow_input, _ = layer(shallow_input)  # 更新浅层输入
#         shallow_features = shallow_input.clone()  # 保存浅层特征供后续使用
#
#         # 2. get length and mask
#         if length is not None:
#             length = self.get_feat_extract_output_lengths(length.detach().cpu())
#             length = length.cuda()
#
#         # 3. 构建深层输入：融合浅层特征和当前输入特征
#         # 当前输入特征通过深层处理，与浅层特征融合
#         deep_input = shallow_features  # 初始化深层输入
#         for layer in self.deep_layers:
#             # 当前深层输入与浅层特征融合
#             deep_input = torch.cat((deep_input, shallow_features), dim=-1)  # 特征拼接
#             deep_input, _ = layer(deep_input)  # 更新深层输入
#
#         # # 3. transformer encoding features
#         # x = self.backbone_model.encoder(
#         #     x, output_hidden_states=True
#         # ).hidden_states
#
#         # # 3. Transformer 编码器处理
#         # # 分层处理：浅层特征传递给深层
#         # for idx, layer in enumerate(self.backbone_model.encoder.layers):
#         #     if idx < len(self.shallow_layers):
#         #         shallow_features = layer(shallow_features)[0]  # 浅层处理
#         #     else:
#         #         # 在深层中融合浅层特征和当前输入特征
#         #         deep_input = torch.cat([shallow_features, x], dim=-1)  # 拼接
#         #         x = layer(deep_input)[0]
#
#         # # 4. stacked feature
#         # stacked_feature = torch.stack(x, dim=0)[1:]
#         #
#         # # 5. Weighted sum
#         # _, *origin_shape = stacked_feature.shape
#         # # Return transformer enc outputs [num_enc_layers, B, T, D]
#         # stacked_feature = stacked_feature.view(self.backbone_model.config.num_hidden_layers, -1)
#         # norm_weights = F.softmax(self.weights, dim=-1)
#         #
#         # # Perform weighted average
#         # weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
#         # features = weighted_feature.view(*origin_shape)
#
#         # 6. Pass the weighted average to point-wise 1D Conv
#         # B x T x D
#         features = features.transpose(1, 2)
#         features = self.model_seq(features)
#         features = features.transpose(1, 2)
#
#         # 7. Pooling
#         if length is not None:
#             #chenhj
#             features = features.to(device)
#             length = length.to(device)
#             # masks = torch.arange(features.size(1)).expand(length.size(0), -1).cuda() < length.unsqueeze(1)
#             masks = torch.arange(features.size(1), device=device).expand(length.size(0), -1) < length.unsqueeze(1)
#             masks = masks.float()
#             features = (features * masks.unsqueeze(-1)).sum(1) / length.unsqueeze(1)
#         else:
#             features = torch.mean(features, dim=1)
#
#         # 8. Output predictions
#         # B x D
#         predicted = self.out_layer(features)
#         return predicted, shallow_features


def prepare_mask(length, shape, dtype):
    # Modified from huggingface
    mask = torch.zeros(
        shape, dtype=dtype
    )
    # these two operations makes sure that all values
    # before the output lengths indices are attended to
    mask[(torch.arange(mask.shape[0]), length.cpu() - 1)] = 1
    mask = mask.flip([-1]).cumsum(-1).flip([-1]).bool()
    return mask
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='emo2vec finetune experiments')
    parser.add_argument(
        '--finetune_method',
        default='none',
        type=str,
        help='finetune method: adapter, embedding prompt, input prompt'
    )

    parser.add_argument(
        '--adapter_hidden_dim',
        default=128,
        type=int,
        help='adapter dimension'
    )

    parser.add_argument(
        '--embedding_prompt_dim',
        default=5,
        type=int,
        help='adapter dimension'
    )

    parser.add_argument(
        '--num_shallow_layers',
        default=6,  # 默认值为 6，可根据任务需要调整
        type=int,
        help='Number of shallow layers for AFL updates'
    )

    args = parser.parse_args()
    model = Wav2VecWrapper(args)
    data = torch.zeros([1, 16000])
    output = model(data)
    print(output.shape)