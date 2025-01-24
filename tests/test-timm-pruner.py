# https://github.com/YuanGongND/ast/blob/master/src/models/ast_models.py
import os, wget
import torch, torch.nn as nn, torch.nn.functional as F, torch_pruning as tp
import timm
from timm.layers import to_2tuple, trunc_normal_

class ASTModel(nn.Module):
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=True, audioset_pretrain=False, model_size='base384', verbose=True):

        super(ASTModel, self).__init__()
        if verbose == True:
            print('---------------AST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain),str(audioset_pretrain)))

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if audioset_pretrain == False:
            if model_size == 'tiny224':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'small224':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base224':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base384':
                self.v = timm.create_model('deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
                        
            else:
                raise Exception('Model size must be one of tiny224, small224, base224, base384.')
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

            # automatcially get the intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            # the linear projection layer
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            if imagenet_pretrain == True:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # the positional embedding
            if imagenet_pretrain == True:
                # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
                # cut (from middle) or interpolate the second dimension of the positional embedding
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
                # cut (from middle) or interpolate the first dimension of the positional embedding
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                # flatten the positional embedding
                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
                # concatenate the above positional embedding with the cls token and distillation token of the deit model.
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
                # TODO can use sinusoidal positional embedding instead
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)

        # now load a model that is pretrained on both ImageNet and AudioSet
        elif audioset_pretrain == True:
            if audioset_pretrain == True and imagenet_pretrain == False:
                raise ValueError('currently model pretrained on only audioset is not supported, please set imagenet_pretrain = True to use audioset pretrained model.')
            if model_size != 'base384':
                raise ValueError('currently only has base384 AudioSet pretrained model.')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if os.path.exists('pths/audioset_10_10_0.4593.pth') == False:
                # this model performs 0.4593 mAP on the audioset eval set
                audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
                wget.download(audioset_mdl_url, out='pths/audioset_10_10_0.4593.pth')
            sd = torch.load('pths/audioset_10_10_0.4593.pth', map_location=device, weights_only=True)
            audio_model = ASTModel(label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=False, audioset_pretrain=False, model_size='base384', verbose=False)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1212, 768).transpose(1, 2).reshape(1, 768, 12, 101)
            # if the input sequence length is larger than the original audioset (10s), then cut the positional embedding
            if t_dim < 101:
                new_pos_embed = new_pos_embed[:, :, :, 50 - int(t_dim/2): 50 - int(t_dim/2) + t_dim]
            # otherwise interpolate
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode='bilinear')
            if f_dim < 12:
                new_pos_embed = new_pos_embed[:, :, 6 - int(f_dim/2): 6 - int(f_dim/2) + f_dim, :]
            # otherwise interpolate
            elif f_dim > 12:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

        # 必须修改
        self.v.patch_embed.img_size = (input_fdim, input_tdim)

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def forward(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2

        x = self.mlp_head(x)
        return x

def prune(args: dict):
    # region--base
    device = torch.device(f"cuda:{args.cuda_id}")
    example_inputs = torch.rand(1, 1024, 128).to(device)
    model = ASTModel(
        label_dim=50, 
        fstride=10, tstride=10, 
        input_fdim=128, input_tdim=1024, 
        imagenet_pretrain=True, audioset_pretrain=True, 
        model_size='base384', verbose=True
        )

    model.eval()
    model.to(device)
    print(f"--output.shape:{model(example_inputs).shape}")

    base_macs, base_param = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"--Base Params:{base_param/1e6:.2f}M, MACs:{base_macs/1e9:.2f}G")
    base_param = sum(p.numel() for p in model.parameters())
    if args.test_before_prune:
        pass
    else:
        base_acc = 0
    # endregion

    # Pruning
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
    ## 配置！！！
    def forward(self, x):
        """https://github.com/huggingface/pytorch-image-models/blob/054c763fcaa7d241564439ae05fbe919ed85e614/timm/models/vision_transformer.py#L79"""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, -1) # original implementation: x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    unwrapped_parameters = [
        (model.v.cls_token, 0),
        (model.v.pos_embed, 0),
        (model.v.dist_token, 0)
    ]
    num_heads = {}
    ignored_layers = [model.mlp_head]
    from timm.models.vision_transformer import Attention, Mlp
    for m in model.modules():
        if isinstance(m, Attention):
            m.forward = forward.__get__(m, Attention)
            num_heads[m.qkv] = m.num_heads
        if args.bottleneck and isinstance(m, Mlp): 
            ignored_layers.append(m.fc2) # only prune the internal layers of FFN & Attention

    pruner = tp.pruner.MetaPruner(
        model, 
        example_inputs, 
        unwrapped_parameters=unwrapped_parameters, # if the model is wrapped by a custom class, the parameters should be unwrapped
        global_pruning=args.global_pruning, # If False, a uniform pruning ratio will be assigned to different layers.
        importance=imp, # importance criterion for parameter selection
        pruning_ratio=args.pruning_ratio, # target pruning ratio
        ignored_layers=ignored_layers,
        num_heads=num_heads, # number of heads in self attention
        prune_num_heads=args.prune_num_heads, # reduce num_heads by pruning entire heads (default: False)
        prune_head_dims=not args.prune_num_heads, # reduce head_dim by pruning featrues dims of each head (default: True)
        head_pruning_ratio=0.9, #args.head_pruning_ratio, # remove 50% heads, only works when prune_num_heads=True (default: 0.0)
        round_to=1
    )

    # if taylor or hessian importance is used, we need to accumulate gradients for the model
    if isinstance(imp, (tp.importance.GroupTaylorImportance, tp.importance.GroupHessianImportance)):
        model.zero_grad()
        if isinstance(imp, tp.importance.GroupHessianImportance):
            imp.zero_grad()
        print("Accumulating gradients for pruning...")
        for k, (imgs, lbls) in enumerate(train_loader):
            if k>=args.taylor_batchs: break
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            output = model(imgs)
            if isinstance(imp, tp.importance.GroupHessianImportance):
                loss = torch.nn.functional.cross_entropy(output, lbls, reduction='none')
                for l in loss:
                    model.zero_grad()
                    l.backward(retain_graph=True)
                    imp.accumulate_grad(model)
            elif isinstance(imp, tp.importance.GroupTaylorImportance):
                loss = torch.nn.functional.cross_entropy(output, lbls)
                loss.backward()

    # start pruning
    for i, g in enumerate(pruner.step(interactive=True)):
        g.prune()

    # Modify the attention head size and all head size aftering pruning
    head_id = 0
    for m in model.modules():
        if isinstance(m, Attention):
            print("Head #%d"%head_id)
            print("[Before Pruning] Num Heads: %d, Head Dim: %d =>"%(m.num_heads, m.head_dim))
            m.num_heads = pruner.num_heads[m.qkv]
            m.head_dim = m.qkv.out_features // (3 * m.num_heads)
            print("[After Pruning] Num Heads: %d, Head Dim: %d"%(m.num_heads, m.head_dim))
            head_id+=1

    print(f"--output.shape: {model(example_inputs).shape}")
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
        bottleneck=False, 
        global_pruning=False, 
        head_pruning_ratio=0.9, 
        prune_num_heads=False, 
        pruning_ratio=0.9
        )

    prune(args)
