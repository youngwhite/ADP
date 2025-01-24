import random, copy, numpy as np
import torch, torch.nn as nn
from transformers import ASTForAudioClassification
import torchaudio.transforms as T

# 
class WrappedAST(nn.Module):
    def __init__(self, num_classes=527, time_mask=192, freq_mask=48, p_augment=0.5):
        super().__init__()
        self.model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.num_layers = len(self.model.audio_spectrogram_transformer.encoder.layer)

        self.p_augment = p_augment
        self.output_hidden_states = False

        if num_classes != 527:
            self.model.classifier.dense = nn.Linear(768, num_classes)

        self.augmentations = nn.ModuleList()
        if time_mask > 0:
            self.augmentations.append(T.TimeMasking(time_mask_param=time_mask))
        if freq_mask > 0:
            self.augmentations.append(T.FrequencyMasking(freq_mask_param=freq_mask))

    def forward(self, S: torch.Tensor):
        if self.training and random.random() < self.p_augment:
            for aug in self.augmentations:
                S = aug(S)
        return self.model(S, output_hidden_states=self.output_hidden_states)

    @staticmethod
    def angular_distance(v1, v2):
        cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        similarity = cos_sim(v1, v2)
        similarity = torch.clamp(similarity, -1.0, 1.0)
        return (1 / np.pi) * torch.acos(similarity).mean().item()

    def analyse_layers(self, valset: Dataset, key: str='fbank', device: str = "cpu"):
        self.eval()
        self.to(torch.device(device))

        layer_count = self.layer_count

        result = np.zeros((layer_count, layer_count))
        with torch.no_grad():
            sample_count = 0
            for d in tqdm(valset, desc="Analyzing layers"):
                outputs = self(
                    input_values=d[key].unsqueeze(0).to(device),
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states

                for i in range(layer_count):
                    for j in range(i + 1, layer_count):
                        dist = self.angular_distance(
                            hidden_states[i][:, 0, :],  # [CLS] token
                            hidden_states[j][:, 0, :]
                        )
                        result[j - i - 1, i] += dist
                sample_count += 1

        result /= sample_count
        return result

