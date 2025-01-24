import random, copy
import torch, torch.nn as nn
from transformers import ASTFeatureExtractor, ASTForAudioClassification
import torchaudio.transforms as T

# 
class WrappedAST(nn.Module):
    def __init__(self, num_classes=527, time_mask=192, freq_mask=48, p_augment=0.5):
        super().__init__()
        self.model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.num_layers = len(self.model.audio_spectrogram_transformer.encoder.layer)

        self.p_augment = p_augment
        self.output_hidden_states = False

        # 修改分类头
        if num_classes != 527:
            self.model.classifier.dense = nn.Linear(768, num_classes)

        # 数据增强模块
        self.augmentations = nn.ModuleList()
        if time_mask > 0:
            self.augmentations.append(T.TimeMasking(time_mask_param=time_mask))
        if freq_mask > 0:
            self.augmentations.append(T.FrequencyMasking(freq_mask_param=freq_mask))

    def get_para_list(self):
        para_list = []
        para_num_classifier = sum(p.numel() for p in self.model.classifier.parameters())
        for i in range(self.num_layers):
            para_num = sum(p.numel() for p in self.model.audio_spectrogram_transformer.encoder.layer[:i+1].parameters())
            para_list.append((para_num+para_num_classifier)/1e6)
        return para_list

    def retain_layers(self, depth: int):
        """
        截取模型的前 depth 层。
        """
        if depth > self.num_layers:
            raise ValueError(f"depth must be <= {self.num_layers}, but got {depth}.")
        self.model.audio_spectrogram_transformer.encoder.layer = self.model.audio_spectrogram_transformer.encoder.layer[:depth]
        self.num_layers = depth
        return self  # 返回修改后的模型，便于链式调用

    def forward(self, S: torch.Tensor):
        # 数据增强（仅在训练模式下生效）
        if self.training and random.random() < self.p_augment:
            for aug in self.augmentations:
                S = aug(S)

        # 调用原始模型的 forward 方法
        return self.model(S, output_hidden_states=self.output_hidden_states)

if __name__ == '__main__':
    import torch
    x = torch.rand(4, 1024, 128)

    model = WrappedAST(num_classes=50)
    model.output_hidden_states = True
    print('--num_params:', sum(p.numel() for p in model.parameters()))

    pruned_model = model.retain_layers(7)
    print('--rest num_params:', sum(p.numel() for p in model.parameters()))
    print('--pruned num_params:', sum(p.numel() for p in pruned_model.parameters()))

    outputs = model(x)
    print('--model outputs.shape:', outputs['logits'].shape)
    print('--length hidden_states:', len(outputs['hidden_states']))

    outputs = pruned_model(x)
    print('--pruned outputs.shape:', outputs['logits'].shape)

    para_list = model.get_para_list()
    # inputs = feature_extractor(x, sampling_rate=16000, return_tensors="pt")['input_values']
    # inputs.shape
