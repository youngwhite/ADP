
# region 
import os, time, copy, tqdm, tempfile, argparse, pandas as pd, numpy as np
import torch, torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import optimizer
import matplotlib.pyplot as plt

import ray
from ray import train, tune
from ray.train import Checkpoint, CheckpointConfig

from datas.esc_dataloaders import get_hf_dataset, split_hf_dataset, get_dataloaders
from models.model_getter import get_model
from src.traintest import set_random_seed, fit
# endregion

def train_epoch_amp(device, train_dataloader, model, optimizer, criterion, scaler):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast(device_type='cuda', enabled=True):
            outputs = model(inputs)['logits']
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += len(labels)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def validate(device, val_dataloader, model, criterion):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)['logits']
            loss = criterion(outputs, labels)

            total_loss += loss.item() * len(labels)  # Adjust for batch-averaged loss
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += len(labels)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

class ADP:
    def __init__(self, dataset_name: str, model_name: str):
        self.task_name = f"ADP-{dataset_name}-{model_name}"
        self.dataset, self.num_classes, self.folds = get_hf_dataset(dataset_name)
        self.model = get_model(model_name, self.num_classes)
        self.num_layers = self.model.num_layers
        # x = self.dataset[0]['fbank'].unsqueeze(0)
        # outputs = self.model(x)
        # print('--outputs.shape:', outputs.shape)

    @staticmethod
    def train_ray(config: dict, common_space: dict):
        # data, model
        trainset, valset = common_space['trainset'], common_space['valset']
        trainloader, valloader = get_dataloaders(
            trainset=trainset, valset=valset,
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )

        # 独立创建
        model = get_model(config['model_name'], config['num_classes'])
        if common_space.get('initial_state'):
            model.load_state_dict(common_space['initial_state'])
        model.retain_layers(config['layers'])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Optimizer, Criterion, Scheduler
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, min_lr=1e-6)
        scaler = GradScaler()

        # Resume from checkpoint if available
        start = 1
        checkpoint = train.get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"), weights_only=True)
                start = checkpoint_dict["epoch"] + 1
                model.load_state_dict(checkpoint_dict["model_state"])
                print("--Resumed training from checkpoint--")
        # Training
        bepoch, bacc = 0, 0
        for epoch in range(start, config['max_epochs']+1):
            # Training and validation
            tloss, tacc = train_epoch_amp(device, trainloader, model, optimizer, criterion, scaler)
            vloss, vacc = validate(device, valloader, model, criterion)

            if vacc > bacc:
                bepoch, bacc = epoch, vacc
            metrics = {'tacc': tacc, 'vacc': vacc, 'bacc': bacc, 'bepoch': bepoch, 'tloss': tloss, 'vloss': vloss}

            if vacc > config.get('save_threshold', 0):
                with tempfile.TemporaryDirectory() as tempdir:
                    torch.save(
                        {"epoch": epoch, "fold": config['fold'], "vacc": vacc, "model_state": model.state_dict()},
                        os.path.join(tempdir, "checkpoint.pt"),
                    )
                    train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir))
            else:
                train.report(metrics)

            # Log training progress
            print(f"Epoch {epoch}/{config['max_epochs']}, Loss-t/v:{tloss:.4f}/{vloss:.4f}, Acc-t/v:{tacc:.4f}/{vacc:.4f}, "
                  f"Best Acc: {bacc:.4f} at Epoch {bepoch}")
            scheduler.step(vloss)
        print(f"--Finished Training: Best Epoch {bepoch}, Best Val Acc {bacc:.4f}--")

    @staticmethod
    def tune_ray(trainable: callable, common_space: dict, search_space: dict, resources: dict={'cpu': 2, 'gpu': 1}, num_samples: int=4):
        # ray.init(local_mode=True)
        os.environ["TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"] = "0"
        ray.init()
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(trainable, common_space=common_space),
                resources={'cpu': 2, 'gpu': 1}
            ),
            param_space=search_space,
            tune_config=tune.TuneConfig(
                metric='vacc',
                mode='max',
                num_samples=search_space.get('num_samples', torch.cuda.device_count())
                ),
            run_config=train.RunConfig(
                storage_path=os.path.abspath('ray_results'),
                checkpoint_config=CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_score_attribute="vacc",
                    checkpoint_score_order="max"
                    )
            )
        )
        result_grid = tuner.fit()
        if not result_grid:
            raise RuntimeError("No results found in tuning process")

        best_result = result_grid.get_best_result(metric="vacc", mode="max", scope='all')

        checkpoint_pth = None
        if best_result.checkpoint:
            best_checkpoint = best_result.get_best_checkpoint('vacc', mode='max')
            with best_checkpoint.as_directory() as checkpoint_dir:
                checkpoint_pth = os.path.join(checkpoint_dir, 'checkpoint.pt')
        else:
            print("--No checkpoint recorded--")

        ray.shutdown()
        return {
            'config': best_result.config,
            'metrics_df': best_result.metrics_dataframe,
            'hacc': best_result.metrics_dataframe['vacc'].max(),
            'hcheckpoint_pth': checkpoint_pth
        }

    def run_iter(self, config: dict):
        config['num_classes'] = self.num_classes
        common_space = {}
        # train
        all_results = []
        for fold in range(1, self.folds+1):
            config['fold'] = fold
            common_space['trainset'], common_space['valset'] = split_hf_dataset(self.dataset, fold=fold)

            ## train base model
            config['layers'] = self.num_layers
            results = self.tune_ray(
                trainable=self.train_ray, # 建立独立模型
                common_space=common_space, # 包含数据集、模型初始状态
                search_space=config
                )
            base_acc, checkpoint_pth = results['hacc'], results['hcheckpoint_pth']
            if checkpoint_pth:
                common_space['initial_state'] = torch.load(checkpoint_pth, weights_only=True)['model_state']
            all_results.append({
                'fold': fold, 
                'layers': self.model.num_layers, 
                'para': sum(p.numel() for p in self.model.parameters()), 
                'acc': base_acc, 
                'checkpoint_pth': checkpoint_pth
                })

            ### train shallow model
            for layers in range(self.model.num_layers-1, 0, -1):
                config['layers'] = layers

                results = self.tune_ray(
                    trainable=self.train_ray, # 建立独立模型
                    common_space=common_space, # 包含数据集、模型初始状态
                    search_space=config
                    )
                acc, checkpoint_pth = results['hacc'], results['hcheckpoint_pth']
                # if bacc - acc > 0.2:
                #     break

                if checkpoint_pth:
                    common_space['initial_state'] = torch.load(checkpoint_pth, weights_only=True)['model_state']

                all_results.append({
                    'fold': fold, 
                    'layers': layers, 
                    'para': sum(p.numel() for p in self.model.retain_layers(layers).parameters()), 
                    'acc': acc, 
                    'checkpoint_pth': checkpoint_pth
                    })
        pd.DataFrame(all_results).to_csv(f"{self.task_name}-iter.csv", index=False)

    @staticmethod
    def train_global_ray(config: dict, common_space: dict):
        # data
        trainset, valset = common_space['trainset'], common_space['valset']
        trainloader, valloader = get_dataloaders(
            trainset=trainset, valset=valset,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            seed=config['seed']
        )
        
        # model
        model = get_model(config['model_name'], config['num_classes'])
        model.output_hidden_states = True

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # middle classifiers & weights
        n_classifiers = model.num_layers
        middle_classifiers = nn.ModuleList([copy.deepcopy(model.model.classifier).to(device) for _ in range(n_classifiers - 1)])
        cweights = torch.nn.Parameter(torch.ones(n_classifiers, device=device, requires_grad=True))
        dweights = torch.nn.Parameter(torch.ones(n_classifiers - 1, device=device, requires_grad=True))

        # Optimizer, Criterion, Scheduler
        criterion = torch.nn.CrossEntropyLoss()
        optimizer_params = list(model.parameters()) + list(middle_classifiers.parameters()) + [cweights]
        if config.get('self_distill', True):
            dcriterion = torch.nn.KLDivLoss(reduction='batchmean')
            optimizer_params.append(dweights)
        optimizer = torch.optim.Adam(optimizer_params, lr=config['lr'])

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, min_lr=1e-6)
        T_max = len(trainloader) * config['max_epochs']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        scaler = GradScaler()

        # Resume from checkpoint if available
        start = 1
        checkpoint = train.get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"), weights_only=True)
                start = checkpoint_dict["epoch"] + 1
                model.load_state_dict(checkpoint_dict["model_state"])
                print("--Resumed training from checkpoint--")

        # Training
        t0 = time.time()
        bepoch, bacc = 0, 0
        for epoch in range(config['max_epochs']):
            model.train()
            tlosses, tcorrects, ttotal = [0] * n_classifiers, [0] * n_classifiers, 0

            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with autocast(device_type='cuda', enabled=True):
                    toutputs = model(inputs)
                    thidden_states, tlogits = toutputs['hidden_states'], toutputs['logits']

                    # 中间层分类器的输出
                    depth_toutputs = [middle_classifiers[i](hidden[:, 0, :]) for i, hidden in enumerate(thidden_states[1:-1])]
                    depth_toutputs.append(tlogits)

                    closses = [criterion(outputs, labels) for outputs in depth_toutputs]

                    if config.get('self_distill', False):
                        with torch.no_grad():
                            teacher_logits = tlogits.softmax(dim=-1)
                        dlosses = [
                            dcriterion(outputs.log_softmax(dim=-1), teacher_logits)
                            for outputs in depth_toutputs[:-1]]

                        cweights_normalized = cweights.softmax(dim=0)
                        dweights_normalized = dweights.softmax(dim=0)
                        dweights_full = torch.cat([dweights_normalized, torch.tensor([1.0], device=device)])

                        total_loss = sum(
                            w * (closs + dw * dloss)
                            for w, dw, closs, dloss in zip(cweights_normalized, dweights_full, closses, dlosses + [0])
                        )
                    else:
                        closs = sum(w * l for w, l in zip(cweights.softmax(dim=0), closses))
                        # cweights_reg = torch.sum((cweights.softmax(dim=0) - 1.0 / n_classifiers) ** 2)
                        total_loss = closs # + 0.01 * cweights_reg  # 权重正则化系数

                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    # metrics
                    for i in range(n_classifiers):
                        tcorrects[i] += (torch.argmax(depth_toutputs[i], dim=-1) == labels).sum().item()
                        tlosses[i] += closses[i].item()
                    ttotal += inputs.size(0)
            # Calculate average training loss and accuracy
            avg_tlosses = [tloss / ttotal for tloss in tlosses]
            avg_taccs = [tcorrect / ttotal for tcorrect in tcorrects]

            # Validation
            model.eval()
            vlosses, vcorrects, vtotal = [0] * n_classifiers, [0] * n_classifiers, 0
            with torch.no_grad():
                for inputs, labels in valloader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    voutputs = model(inputs)
                    vhidden_states, vlogits = voutputs['hidden_states'], voutputs['logits']

                    # hidden states
                    depth_voutputs = [middle_classifiers[i](hidden[:, 0, :]) for i, hidden in enumerate(vhidden_states[1:-1])]
                    depth_voutputs.append(vlogits)

                    closses = [criterion(outputs, labels) for outputs in depth_voutputs]

                    # metrics
                    vcorrects = [vcorrects[i] + (torch.argmax(depth_voutputs[i], dim=-1) == labels).sum().item() for i in range(n_classifiers)]
                    vlosses = [vlosses[i] + closses[i].item() for i in range(n_classifiers)]
                    vtotal += inputs.size(0)
            # Calculate average training loss and accuracy
            avg_vlosses = [vloss / vtotal for vloss in vlosses]
            avg_vaccs = [vcorrect / vtotal for vcorrect in vcorrects]

            # Logging
            bvacc = max(avg_vaccs)
            if bvacc > bacc:
                bepoch, bacc, save_pth = epoch, bvacc, True
            else:
                save_pth = False

            metrics = {
                'tacc': max(avg_taccs), 'vacc': max(avg_vaccs),
                'bepoch': bepoch,
                'bacc': bacc,
                'time': time.time() - t0,
                'tlosses': avg_tlosses, 'vlosses': avg_vlosses,
                'taccs': avg_taccs, 'vaccs': avg_vaccs,
            }
            scheduler.step(min(avg_vlosses))

            if bvacc > config.get('save_threshold', 0) and save_pth:
                with tempfile.TemporaryDirectory() as tempdir:
                    checkpoint_path = os.path.join(tempdir, "checkpoint.pt")
                    torch.save(
                        {"epoch": epoch, "fold": config['fold'], "vacc": bvacc, 
                        "model_state": model.state_dict(), "middle_states": middle_classifiers.state_dict()},
                        checkpoint_path,
                    )
                    train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir))
            else:
                train.report(metrics)

        # 

    def run_global(self, config):
        config['num_classes'] = self.num_classes
        common_space = {}
        # train
        paras, accs = self.model.get_para_list(), []
        for fold in range(1, self.folds+1):
            config['fold'] = fold
            common_space['trainset'], common_space['valset'] = split_hf_dataset(self.dataset, fold=fold)
            results = self.tune_ray(
                    trainable=self.train_global_ray, # 建立独立模型
                    common_space=common_space, # 包含数据集、模型初始状态
                    search_space=config
                    )
            acc, checkpoint_pth = results['hacc'], results['hcheckpoint_pth']

            savename = f"{self.task_name}-global-{fold}_{acc}"            
            metrics_df = results['metrics_df']

            best_row = metrics_df.loc[metrics_df['vacc'].idxmax()]
            print(f"--Final acc: {acc}, pth: {results['hcheckpoint_pth']}, accs: {best_row['vaccs']}--")            

            pd.DataFrame(
                {'param': paras, 'acc': best_row['vaccs']}
                ).to_csv(f"{savename}.csv", index=False)
            self.plot_curve(
                paras, 
                best_row['vaccs'], 
                f"{savename}.png"
                )
            accs.append(best_row['vaccs'])
        accs_array = np.array(accs).mean(axis=0)

        self.plot_curve(paras, accs_array, f"{self.task_name}-global-mean.png")
        print(f"--Depth paras: {paras}--")
        print(f"--Depth Mean accs: {accs_array}--")

    @staticmethod
    def plot_curve(para_list, vacc_list, savename):
        vaccs = [100*vacc for vacc in vacc_list]
        plt.figure()
        plt.plot(para_list, vaccs, marker='o')
        plt.xlabel("Parameter Number (M)")
        plt.ylabel("Validation Accuracy (%)")
        plt.title("Parameter Number vs. Average Accuracy")
        plt.xticks(range(0, 100, 10))
        plt.yticks(range(0, 101, 10))
        plt.grid(True)
        # 在坐标点旁边标记数值
        for x, y in zip(para_list, vaccs):
            # plt.text(x, y, f'{y:.1f}', fontsize=12, ha='left', va='top')
            offset_x, offset_y = -2.5, 3
            plt.text(x + offset_x, y + offset_y, f'{y:.2f}', fontsize=9, ha='center', va='center')

        plt.savefig(savename, dpi=300, bbox_inches='tight')  # PNG 格式，300 DPI
        plt.close()

def main(args): # 传入argparse对象
    set_random_seed(args.seed)

    if args.mode == 'iter':
        adp = ADP(args.dataset_name, args.model_name)
        adp.run_iter(vars(args))
    elif args.mode == 'global':
        adp = ADP(args.dataset_name, args.model_name)
        adp.run_global(vars(args))
    else:
        raise ValueError(f"--Invalid mode: {args.mode}")

if __name__ == '__main__':
    # region args
    parser = argparse.ArgumentParser(description="commands for ADP.")
    parser.add_argument('--dataset_name', type=str, default='esc50')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)    
    parser.add_argument('--model_name', type=str, default='ast')

    parser.add_argument('--mode', type=str, default='global', choices=['iter', 'global'])
    parser.add_argument('--self_distill', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--max_epochs', type=int, default=2)
    parser.add_argument('--save_threshold', type=float, default=0.94)
    parser.add_argument('--num_samples', type=int, default=4)
    args = parser.parse_args()
    # endregion

    main(args)
    # python run-adp.py --mode 'iter' --max_epochs 60 --save_threshold 0.94
    
    ## adp_global
    # python run-adp.py --mode 'global' --max_epochs 30 --save_threshold 0.94 --num_samples 36 | tee adp_global.txt

    ## +self-distill
    # python run-adp.py --mode 'global' --max_epochs 30 --save_threshold 0.94 --self_distill --num_samples 36 | tee adp_global_sd.txt

