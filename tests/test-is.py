import os, sys, random, copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
import torch, torch.nn as nn, torch.nn.functional as F, torch_pruning as tp
import torchaudio.transforms as T
from transformers import ASTForAudioClassification
from transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer import ASTSelfOutput, ASTSdpaSelfAttention
from tqdm import tqdm

class ASTModel(nn.Module):
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

def prune(args: dict):
    # region --basemodel
    device = torch.device(f"cuda:{args.cuda_id}")
    example_inputs = torch.rand(1, 1024, 128).to(device)

    # model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(device)
    # model.classifier.dense = nn.Linear(768, 50)

    model = ASTModel(num_classes=50).to(device)  
    model.to(device)
    print(f"--output.shape: {model(example_inputs).logits.shape}")

    base_macs, base_param = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"--Base Params:{base_param/1e6:.2f}M, MACs:{base_macs/1e9:.2f}G")

    if args.test_before_prune:
        pass
    else:
        base_acc = 0
    # endregion

    # region --importance
    if args.pruning_type == 'random':
        imp = tp.importance.RandomImportance()
    elif args.pruning_type == 'taylor':
        imp = tp.importance.TaylorImportance()
    elif args.pruning_type == 'l1':
        imp = tp.importance.MagnitudeImportance(p=1)
    elif args.pruning_type == 'l2':
        imp = tp.importance.MagnitudeImportance(p=2)
    else: 
        raise NotImplementedError
    # endregion

    if args.pruning_type=='taylor':
        # train_loader, val_loader 
        pass

    # 构建依赖图
    # dependency_graph = tp.DependencyGraph()
    # dependency_graph.build_dependency(model, example_inputs)

    unwrapped_parameters = [
        (model.model.audio_spectrogram_transformer.embeddings.cls_token, 0),
        (model.model.audio_spectrogram_transformer.embeddings.distillation_token, 0),
        (model.model.audio_spectrogram_transformer.embeddings.position_embeddings, 0),
    ]

    ignored_layers = [
        # model.model.audio_spectrogram_transformer.embeddings,
        model.model.classifier
        ]
    
    num_heads = {}
    for m in model.modules():
        if isinstance(m, ASTSdpaSelfAttention):
            num_heads[m.query] = m.num_attention_heads
            num_heads[m.key] = m.num_attention_heads
            num_heads[m.value] = m.num_attention_heads

        if args.bottleneck and isinstance(m, ASTSelfOutput):
            ignored_layers.append(m.dense)

    pruner = tp.pruner.MetaPruner(
        model, 
        example_inputs, 
        ignored_layers=ignored_layers,        
        unwrapped_parameters=unwrapped_parameters, # if the model is wrapped by a custom class, the parameters should be unwrapped
        num_heads=num_heads, # number of heads in self attention        
        importance=imp,
        isomorphic=args.isomorphic, # If True, the pruner will prune the same amount of channels for each layer.
        global_pruning=args.global_pruning, # If False, a uniform pruning ratio will be assigned to different layers.
        pruning_ratio=args.pruning_ratio, # target pruning ratio
        prune_num_heads=args.prune_num_heads, # reduce num_heads by pruning entire heads (default: False)
        prune_head_dims=not args.prune_num_heads, # reduce head_dim by pruning featrues dims of each head (default: True)
        head_pruning_ratio=0.9, #args.head_pruning_ratio, # remove 50% heads, only works when prune_num_heads=True (default: 0.0)
        round_to=1
    )
    
    if isinstance(imp, tp.importance.TaylorImportance):
        model.zero_grad()
        print("Accumulating gradients for taylor pruning...")
        for k, (imgs, lbls) in enumerate(trainloader):
            if k>=args.taylor_batchs: break
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            output = model(imgs).logits
            loss = torch.nn.functional.cross_entropy(output, lbls)
            loss.backward()

    for g in pruner.step(interactive=True):
        g.prune()

    # Modify the attention head size and all head size aftering pruning
    for m in model.modules():
        if isinstance(m, ASTSdpaSelfAttention):
            print("num_heads:", m.num_attention_heads, 'head_dims:', m.attention_head_size, 'all_head_size:', m.all_head_size, '=>')
            m.num_attention_heads = pruner.num_heads[m.query]
            m.attention_head_size = m.query.out_features // m.num_attention_heads
            m.all_head_size = m.query.out_features
            print("num_heads:", m.num_attention_heads, 'head_dims:', m.attention_head_size, 'all_head_size:', m.all_head_size)

    print(f"--output.shape: {model(example_inputs).logits.shape}")

    print("----------------------------------------")
    pruned_macs, pruned_param = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"--Base Param: {base_param/1e6:.2f}M, Macs: {base_macs/1e9:.2f}G")
    print(f"--Pruned Param: {pruned_param/1e6:.2f}M, Macs: {pruned_macs/1e9:.2f}G")

if __name__ == '__main__':

    # Pruning
    from types import SimpleNamespace
    args = SimpleNamespace(
        cuda_id=0,
        test_before_prune=True,
        pruning_type='l2',
        bottleneck=True,
        isomorphic=False, #
        global_pruning=False, # 
        head_pruning_ratio=0.9, 
        prune_num_heads=False, 
        pruning_ratio=0.9
        )

    prune(args)