import os, sys, logging, numpy as np, random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Callable, Sequence, Tuple, Dict
import torch, torch.nn as nn, torch_pruning as tp
from torch_pruning import BasePruningFunc

from datas.esc_dataloaders import get_hf_dataset, split_hf_dataset, get_dataloaders
from models.model_getter import get_model
from src.traintest import train_epoch_amp, validate_epoch_amp, validate, fit

def set_random_seed(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ASTAttentionPruner(BasePruningFunc):
    def __init__(self):
        self.pruning_dim = 0  # 默认剪枝的维度

    def get_out_channels(self, layer: nn.Module) -> int:
        return layer.query.weight.size(0)

    def get_in_channels(self, layer: nn.Module) -> int:
        return layer.query.weight.size(1)

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        # print(f"[ASTAttentionPruner] Pruning layer: {layer} on {len(idxs)} out_channels")
        
        assert hasattr(layer, "query"), "Layer does not have 'query' attribute. Check the layer type."
        keep_idxs = list(set(range(layer.query.weight.size(0))) - set(idxs))
        keep_idxs.sort()

        layer.query.weight = torch.nn.Parameter(layer.query.weight[keep_idxs])
        layer.key.weight = torch.nn.Parameter(layer.key.weight[keep_idxs])
        layer.value.weight = torch.nn.Parameter(layer.value.weight[keep_idxs])
        layer.out_proj.weight = torch.nn.Parameter(layer.out_proj.weight[keep_idxs])

        return layer

    def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        assert hasattr(layer, "query"), "Layer does not have 'query' attribute. Check the layer type."
        keep_idxs = list(set(range(layer.query.weight.size(1))) - set(idxs))
        keep_idxs.sort()

        print(f"[ASTAttentionPruner] Pruning {layer.__class__.__name__} on {len(idxs)} in_channels.")

        layer.query.weight = torch.nn.Parameter(layer.query.weight[:, keep_idxs])
        layer.key.weight = torch.nn.Parameter(layer.key.weight[:, keep_idxs])
        layer.value.weight = torch.nn.Parameter(layer.value.weight[:, keep_idxs])

        return layer

ast_attention_pruner = ASTAttentionPruner()

class ASTLinearPruner(BasePruningFunc):
    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        assert len(idxs) < layer.out_features, "Cannot prune all out_channels."
        keep_idxs = list(set(range(layer.out_features)) - set(idxs))
        keep_idxs.sort()

        # print(f"[ASTLinearPruner] Pruning {layer.__class__.__name__} on {len(idxs)} out_channels.")

        layer.weight = nn.Parameter(layer.weight[keep_idxs])
        if layer.bias is not None:
            layer.bias = nn.Parameter(layer.bias[keep_idxs])
        layer.out_features = len(keep_idxs)

        return layer

    def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        assert len(idxs) < layer.in_features, "Cannot prune all in_channels."
        keep_idxs = list(set(range(layer.in_features)) - set(idxs))
        keep_idxs.sort()

        # print(f"[ASTLinearPruner] Pruning {layer.__class__.__name__} on {len(idxs)} in_channels.")

        layer.weight = nn.Parameter(layer.weight[:, keep_idxs])
        layer.in_features = len(keep_idxs)

        return layer

    def get_out_channels(self, layer: nn.Module) -> int:
        return layer.out_features

    def get_in_channels(self, layer: nn.Module) -> int:
        return layer.in_features

ast_linear_pruner = ASTLinearPruner()

#
class MagnitudeImportance(tp.importance.Importance):
    def __init__(self, p=2, group_reduction="mean", normalizer=None):
        self.p = p
        self.group_reduction = group_reduction
        self.normalizer = normalizer

    def _reduce(self, group_imp):
        if self.group_reduction == "sum":
            group_imp = group_imp.sum(dim=0)
        elif self.group_reduction == "mean":
            group_imp = group_imp.mean(dim=0)
        elif self.group_reduction == "max":
            group_imp = group_imp.max(dim=0)[0]
        elif self.group_reduction == "prod":
            group_imp = torch.prod(group_imp, dim=0)
        elif self.group_reduction=='first':
            group_imp = group_imp[0]
        elif self.group_reduction is None:
            group_imp = group_imp
        else: 
            raise NotImplementedError
        return group_imp

    @torch.no_grad()
    def __call__(self, group, ch_groups=1, consecutive_groups=1):
        group_imp = []
        for dep, idxs in group:
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler
            # Linear out_channels
            if prune_fn in [tp.prune_linear_out_channels, ast_linear_pruner.prune_out_channels]:
                #print(layer, idxs)
                w = layer.weight.data[idxs].flatten(1)
                local_norm = w.abs().pow(self.p).sum(1)
                group_imp.append(local_norm)
            # Linear in_channels
            elif prune_fn in [
                tp.prune_linear_in_channels, ast_linear_pruner.prune_in_channels
            ]:    
                w = layer.weight
                local_norm = w.abs().pow(self.p).sum(0)
                local_norm = local_norm[idxs]
                group_imp.append(local_norm)
            # RMSNorm
            elif prune_fn == ast_attention_pruner.prune_out_channels:
                # regularize BN
                w = layer.weight.data[idxs]
                local_norm = w.abs().pow(self.p)
                group_imp.append(local_norm)
            # Embedding
            elif prune_fn == tp.prune_embedding_out_channels:
                w = layer.weight.data[:, idxs]
                local_norm = w.abs().pow(self.p)
                group_imp.append(local_norm)
            # Attention
            elif prune_fn == ast_attention_pruner.prune_out_channels:
                local_norm = 0
                for sub_layer in [layer.o_proj]:
                    w_out = sub_layer.weight.data[idxs]
                    local_norm += w_out.abs().pow(self.p).sum(1)

                for sub_layer in [layer.q_proj, layer.k_proj, layer.v_proj]:
                    w_in = sub_layer.weight.data[:, idxs]
                    local_norm += w_in.abs().pow(self.p).sum(0)
                group_imp.append(local_norm)

        if len(group_imp)==0:
            return None
        min_imp_size = min([len(imp) for imp in group_imp])
        aligned_group_imp = []
        for imp in group_imp:
            if len(imp)>min_imp_size and len(imp)%min_imp_size==0:
                imp = imp.view(len(imp) // min_imp_size, min_imp_size).sum(0)
                aligned_group_imp.append(imp)
            elif len(imp)==min_imp_size:
                aligned_group_imp.append(imp)
        group_imp = torch.stack(aligned_group_imp, dim=0)
        group_imp = self._reduce(group_imp)
        if self.normalizer is not None:
            group_imp = self.normalizer(group, group_imp)
        return group_imp

class TaylorImportance(tp.importance.Importance):
    def __init__(self, group_reduction="sum", normalizer=None, taylor=None):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.taylor = taylor

    def _reduce(self, group_imp):
        if self.group_reduction == "sum":
            group_imp = group_imp.sum(dim=0)
        elif self.group_reduction == "mean":
            group_imp = group_imp.mean(dim=0)
        elif self.group_reduction == "max":
            group_imp = group_imp.max(dim=0)[0]
        elif self.group_reduction == "prod":
            group_imp = torch.prod(group_imp, dim=0)
        elif self.group_reduction=='first':
            group_imp = group_imp[0]
        elif self.group_reduction=='second':
            group_imp = group_imp[1]
        elif self.group_reduction is None:
            group_imp = group_imp
        else: 
            raise NotImplementedError
        return group_imp

    @torch.no_grad()
    def __call__(self, group, ch_groups=1, consecutive_groups=1):
        group_imp = []
        for dep, idxs in group:
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler

            if prune_fn not in [
                tp.prune_linear_out_channels, tp.prune_linear_in_channels, 
                hf_rmsnorm_pruner.prune_out_channels, tp.prune_embedding_out_channels, hf_attention_pruner.prune_out_channels,
                hf_linear_pruner.prune_out_channels, hf_linear_pruner.prune_in_channels
            ]:
                continue
            
            if prune_fn in [hf_attention_pruner.prune_out_channels]:
                salience = {}
                for sub_layer in [layer.o_proj, layer.q_proj, layer.k_proj, layer.v_proj]:
                    salience[sub_layer] = sub_layer.weight * sub_layer.weight.grad
                    
                    if self.taylor in ['param_second']:
                        salience[sub_layer] = sub_layer.weight * sub_layer.weight.acc_grad * sub_layer.weight
                    elif self.taylor in ['param_mix']: 
                        salience[sub_layer] = -salience + 0.5 * sub_layer.weight * sub_layer.weight.acc_grad * sub_layer.weight   
            else:
                salience = layer.weight * layer.weight.grad

                if self.taylor in ['param_second']:
                    salience = layer.weight * layer.weight.acc_grad * layer.weight
                elif self.taylor in ['param_mix']: 
                    salience = salience - 0.5 * layer.weight * layer.weight.acc_grad * layer.weight
                    
            # Linear out_channels
            if prune_fn in [tp.prune_linear_out_channels, hf_linear_pruner.prune_out_channels]:
                if self.taylor == 'vectorize':
                    local_norm = salience.sum(1).abs()
                elif 'param' in self.taylor:
                    local_norm = salience.abs().sum(1)
                else:
                    raise NotImplementedError
                group_imp.append(local_norm)

            # Linear in_channels
            elif prune_fn in [tp.prune_linear_in_channels, hf_linear_pruner.prune_in_channels]:
                if self.taylor == 'vectorize':
                    local_norm = salience.sum(0).abs()
                elif 'param' in self.taylor:
                    local_norm = salience.abs().sum(0)
                else:
                    raise NotImplementedError
                local_norm = local_norm[idxs]
                group_imp.append(local_norm)

            # RMSNorm
            elif prune_fn == hf_rmsnorm_pruner.prune_out_channels:
                local_norm = salience.abs()
                group_imp.append(local_norm)

            # Embedding
            elif prune_fn == tp.prune_embedding_out_channels:
                if self.taylor == 'vectorize':
                    local_norm = salience[:, idxs].sum(0).abs()
                elif 'param' in self.taylor:
                    local_norm = salience[:, idxs].abs().sum(0)
                else:
                    raise NotImplementedError
                group_imp.append(local_norm)

            # Attention
            elif prune_fn == hf_attention_pruner.prune_out_channels:
                local_norm = 0
                for sub_layer in [layer.o_proj]: #linear out channel, first dim in linear.weight
                    if self.taylor == 'vectorize':
                        local_norm += salience[sub_layer].sum(1).abs()
                    elif 'param' in self.taylor: 
                        local_norm += salience[sub_layer].abs().sum(1)   
                    else:
                        raise NotImplementedError                
                
                for sub_layer in [layer.q_proj, layer.k_proj, layer.v_proj]: # linear in channel, second dim in linear.weight
                    if self.taylor == 'vectorize':
                        local_norm += salience[sub_layer].sum(0).abs() 
                    elif 'param' in self.taylor == 'param':
                        local_norm += salience[sub_layer].abs().sum(0)
                    else:
                        raise NotImplementedError
                group_imp.append(local_norm)

        if len(group_imp)==0:
            return None

        min_imp_size = min([len(imp) for imp in group_imp])
        aligned_group_imp = []
        for imp in group_imp:
            if len(imp)>min_imp_size and len(imp)%min_imp_size==0:
                imp = imp.view(len(imp) // min_imp_size, min_imp_size).sum(0)
                aligned_group_imp.append(imp)
            elif len(imp)==min_imp_size:
                aligned_group_imp.append(imp)
        group_imp = torch.stack(aligned_group_imp, dim=0)
        group_imp = self._reduce(group_imp)
        if self.normalizer is not None:
            group_imp = self.normalizer(group, group_imp)
        return group_imp


# class Args:
#     seed = 0
#     pruner_type = 'random'
#     block_wise = True
#     global_pruning = True
#     iterative_steps = 10
#     pruning_ratio = 0.5
#     grouping_strategy = 'mean'
#     taylor = 'first'
#     block_attention_layer_start = 0
#     block_attention_layer_end = 11
#     block_mlp_layer_start = 0
#     block_mlp_layer_end = 11
# args = Args()


def main(args):
    # 设置日志
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 设置随机种子
    torch.manual_seed(args.seed)

    # 加载数据集
    dataset, num_classes, folds = get_hf_dataset('esc50')
    trainset, valset = split_hf_dataset(dataset, fold=args.fold)
    trainloader, valloader = get_dataloaders(trainset=trainset, valset=valset, num_workers=4, batch_size=16)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    example_inputs = torch.randn(1, 1024, 128).to(device)
    model = get_model(model_name='ast', num_classes=num_classes).to(device)
    print(f"--original params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # 设置重要性评估方法
    if args.pruner_type == 'random':
        imp = tp.importance.RandomImportance()
    elif args.pruner_type == 'l1':
        imp = MagnitudeImportance(p=1)
    elif args.pruner_type == 'l2':
        imp = MagnitudeImportance(p=2)
    elif args.pruner_type == 'taylor':
        imp = TaylorImportance(group_reduction=args.grouping_strategy, taylor=args.taylor)
    else:
        raise NotImplementedError("Unsupported pruner type.")

    # 设置剪枝参数
    kwargs = {
        "importance": imp,
        "global_pruning": args.global_pruning,
        "iterative_steps": args.iterative_steps,
        "pruning_ratio": args.pruning_ratio,
        "ignored_layers": [
            model.model.audio_spectrogram_transformer.embeddings,
            model.model.classifier.dense,
        ],
        "unwrapped_parameters": [
            (model.model.audio_spectrogram_transformer.embeddings.cls_token, 0),
            (model.model.audio_spectrogram_transformer.embeddings.distillation_token, 0),
            (model.model.audio_spectrogram_transformer.embeddings.position_embeddings, 0),
        ],
        # "round_to": 8,
    }

    # 设置自定义剪枝器
    customized_pruners = {}
    for i, layer in enumerate(model.model.audio_spectrogram_transformer.encoder.layer):
        customized_pruners[type(layer.attention.attention)] = ast_attention_pruner
        customized_pruners[type(layer.attention.output.dense)] = ast_linear_pruner
        customized_pruners[type(layer.intermediate.dense)] = ast_linear_pruner
        customized_pruners[type(layer.output.dense)] = ast_linear_pruner
    kwargs["customized_pruners"] = customized_pruners

    # 设置通道分组
    in_channel_groups, out_channel_groups = {}, {}
    for layer in model.model.audio_spectrogram_transformer.encoder.layer:
        attention_layer = layer.attention
        in_channel_groups[attention_layer.attention.query] = attention_layer.attention.query.weight.size(1) // 12
        out_channel_groups[attention_layer.attention.query] = attention_layer.attention.query.weight.size(0) // 12
        out_channel_groups[attention_layer.output.dense] = attention_layer.output.dense.weight.size(0) // 12

        intermediate_layer = layer.intermediate
        out_channel_groups[intermediate_layer.dense] = intermediate_layer.dense.weight.size(0) // 12
        in_channel_groups[intermediate_layer.dense] = intermediate_layer.dense.weight.size(1) // 12

        output_layer = layer.output
        out_channel_groups[output_layer.dense] = output_layer.dense.weight.size(0) // 12
        in_channel_groups[output_layer.dense] = output_layer.dense.weight.size(1) // 12

    kwargs["in_channel_groups"] = in_channel_groups
    kwargs["out_channel_groups"] = out_channel_groups

    # 初始化剪枝器
    pruner = tp.pruner.MetaPruner(
        model=model,
        example_inputs=example_inputs,
        **kwargs
    )

    # 开始剪枝
    logger.info("--Start Pruning")
    for i in range(args.iterative_steps):
        if args.pruner_type == 'taylor':
            model.zero_grad()
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast():
                    outputs = model(inputs)
                    loss = F.cross_entropy(outputs, labels)
                loss.backward()
        
        logger.info(f"Before pruning step {i+1}: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        pruner.step()
        logger.info(f"After pruning step {i+1}: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        model.zero_grad()

    print("--Pruning Completed")
    print(f"--output.shape: {model(example_inputs).shape}")
    print(f"--pruned params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    #----------------------------------------
    # Recovery accuracy
    if args.post_train:
        result = fit(device, trainloader, valloader, model, lr=args.lr, epochs=args.epochs)    
        pruned_acc = result['bacc']
        model.load_state_dict(result['bstate'])
        print(f"--pruned acc: {pruned_acc*100:.2f}%")
    else:
        _, pruned_acc = validate(device, valloader, model, criterion=nn.CrossEntropyLoss())
        print(f"--pruned acc: {pruned_acc*100:.2f}%")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='AST Pruning')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--pruner_type', type=str, default='l2')
    parser.add_argument('--block_wise', action='store_true')
    parser.add_argument('--global_pruning', action='store_true')
    parser.add_argument('--iterative_steps', type=int, default=10)
    parser.add_argument('--pruning_ratio', type=float, default=0.2)
    parser.add_argument('--grouping_strategy', type=str, default='mean')
    parser.add_argument('--taylor', type=str, default='first')
    parser.add_argument('--block_attention_layer_start', type=int, default=0)
    parser.add_argument('--block_attention_layer_end', type=int, default=11)
    parser.add_argument('--block_mlp_layer_start', type=int, default=0)
    parser.add_argument('--block_mlp_layer_end', type=int, default=11)
    parser.add_argument('--post_train', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    main(args)

    #
    # python run-llm.py --block_wise --global_pruning --pruning_type 'taylor' --pruning_ratio 0.2
