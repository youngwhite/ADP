# # region: pytorch dataset ESC-50
# import os, tempfile, h5py, copy, pandas as pd
# import torch, torchaudio
# from torch.utils.data import Dataset, DataLoader
# from transformers import ASTFeatureExtractor
# from concurrent.futures import ProcessPoolExecutor

# class ESC50(Dataset):
#     def __init__(self, cache_dir="esc50_cache", num_workers=4):
#         self.root = self.download_and_extract_esc50_to_temp()
#         self.df = pd.read_csv(os.path.join(self.root, 'ESC-50-master/meta/esc50.csv'))

#         self.feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
#         self.df['fbank'] = self.df['filename'].apply(self._load_waveform).apply(self._compute_fbank)

#     def _load_waveform(self, filename):
#         filepath = os.path.join(self.root, 'ESC-50-master/audio', filename)
#         waveform, _ = torchaudio.load(filepath, normalize=True)
#         return waveform.mean(dim=0, keepdim=False)  # 转为单声道

#     def _compute_fbank(self, waveform):
#         fbank = self.feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")['input_values']
#         return fbank.squeeze(0)

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         return row['fbank'], row['target']

#     def split_dataset(self, fold: int):
#         folds_train, folds_val = [i for i in [1, 2, 3, 4, 5] if i != fold], [fold]
#         df_train, df_val = self.df[self.df['fold'].isin(folds_train)].reset_index(drop=True), self.df[self.df['fold'].isin(folds_val)].reset_index(drop=True)

#         trainset, valset = copy.copy(self), copy.copy(self)
#         trainset.df, valset.df = df_train, df_val
#         return trainset, valset

#     def download_and_extract_esc50_to_temp(self):
#         temp_dir = tempfile.gettempdir()
#         zip_file = os.path.join(temp_dir, "ESC-50-master.zip")
#         if not os.path.exists(zip_file):
#             dataset_url = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
#             response = requests.get(dataset_url, stream=True)
#             with open(zip_file, "wb") as file:
#                 for chunk in response.iter_content(chunk_size=1024):
#                     if chunk:
#                         file.write(chunk)
#             with zipfile.ZipFile(zip_file, 'r') as zip_ref:
#                 zip_ref.extractall(temp_dir)
#         return temp_dir
# # endregion

# hugging face dataset ESC-50
import os, tempfile, zipfile, tarfile, requests, numpy as np, pandas as pd
from filelock import FileLock
import torch, torchaudio
from torch.utils.data import DataLoader
from transformers import ASTFeatureExtractor
from datasets import Dataset, DatasetDict

def get_esc50_datasets(fold: int, num_proc=2):

    def download_and_extract_esc50(cache_dir: str):
        zip_file = os.path.join(cache_dir, "ESC-50-master.zip")
        extracted_dir = os.path.join(cache_dir, "ESC-50-master")
        
        if not os.path.exists(extracted_dir):
            os.makedirs(cache_dir, exist_ok=True)
            if not os.path.exists(zip_file):
                dataset_url = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
                response = requests.get(dataset_url, stream=True)
                with open(zip_file, "wb") as file:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            file.write(chunk)
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(cache_dir)
        return extracted_dir

    cache_dir = os.path.join(tempfile.gettempdir(), "esc50_cache")
    root = download_and_extract_esc50(cache_dir)

    # 加载元数据
    csv_path = os.path.join(root, "meta", "esc50.csv")
    dataset = Dataset.from_csv(csv_path)

    # 预处理音频
    feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    def preprocess(example):
        audio_path = os.path.join(root, "audio", example["filename"])
        waveform, _ = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0)  # 转为单声道
        example["fbank"] = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")["input_values"].squeeze(0)
        return example

    with FileLock(os.path.expanduser("~/.data.lock")):
        # 检查是否已缓存特征
        cached_features_path = os.path.join(cache_dir, "esc50_features.arrow")
        if os.path.exists(cached_features_path):
            dataset = Dataset.load_from_disk(cached_features_path)
        else:
            # 并行预处理
            dataset = dataset.map(preprocess, num_proc=num_proc)
            dataset.save_to_disk(cached_features_path)

        # 划分数据集
        folds = np.array(dataset["fold"])
        train_indices, val_indices = np.where(folds != fold)[0], np.where(folds == fold)[0]
        trainset, valset = dataset.select(train_indices), dataset.select(val_indices)

        trainset.set_format(type='torch', columns=['fbank', 'target'])
        valset.set_format(type='torch', columns=['fbank', 'target'])
    return trainset, valset

def get_urbansound8k_datasets(fold: int, num_proc=2):
    
    # https://goo.gl/8hY5ER
    cache_dir = os.path.join(tempfile.gettempdir(), "urbansound8k_cache")    
    def download_and_extract_urbansound8k(cache_dir: str):
        tar_file = os.path.join(cache_dir, "UrbanSound8K.tar.gz")
        extracted_dir = os.path.join(cache_dir, "UrbanSound8K")

        # 检查是否已解压
        if not os.path.exists(extracted_dir):
            os.makedirs(cache_dir, exist_ok=True)

            # 如果压缩包不存在，则下载
            if not os.path.exists(tar_file):
                dataset_url = "https://goo.gl/8hY5ER"  # 使用短链接
                print("Downloading UrbanSound8K dataset from goo.gl...")
                response = requests.get(dataset_url, stream=True, allow_redirects=True)  # 确保处理重定向
                with open(tar_file, "wb") as file:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            file.write(chunk)
                print("Download complete!")

            # 解压 .tar.gz 文件
            print("Extracting UrbanSound8K dataset...")
            with tarfile.open(tar_file, "r:gz") as tar:
                tar.extractall(path=cache_dir)
            print("Extraction complete!")

        return extracted_dir

    root = download_and_extract_urbansound8k(cache_dir)

    # 加载元数据
    csv_path = os.path.join(root, "metadata", "UrbanSound8K.csv")
    dataset = Dataset.from_csv(csv_path)

    # 预处理音频
    feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    def preprocess(example):
        audio_path = os.path.join(root, "audio", 'fold'+str(example['fold']), example["slice_file_name"])
        waveform, _ = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0)  # 转为单声道
        example["fbank"] = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")["input_values"].squeeze(0)
        example["target"] = example["classID"]
        del example["classID"]
        return example

    with FileLock(os.path.expanduser("~/.data.lock")):
        # 检查是否已缓存特征
        cached_features_path = os.path.join(cache_dir, "urbansound8k_features.arrow")
        if os.path.exists(cached_features_path):
            dataset = Dataset.load_from_disk(cached_features_path)
        else:
            # 并行预处理
            dataset = dataset.map(preprocess, num_proc=num_proc)
            dataset.save_to_disk(cached_features_path)

        # 划分数据集
        folds = np.array(dataset["fold"])
        train_indices, val_indices = np.where(folds != fold)[0], np.where(folds == fold)[0]
        trainset, valset = dataset.select(train_indices), dataset.select(val_indices)

        trainset.set_format(type='torch', columns=['fbank', 'target'])
        valset.set_format(type='torch', columns=['fbank', 'target'])
    return trainset, valset

def get_dataloaders(config: dict):
    if config['dataset'] == 'ESC-50':
        trainset, valset = get_esc50_datasets(config['fold'])
        num_classes = 50
    elif config['dataset'] == "UrbanSound8K":
        trainset, valset = get_urbansound8k_datasets(config['fold'])
        num_classes = 10
    else:
        raise ValueError(f"--Unknown dataset: {config['dataset']}")

    def augment_fn(batch, twidth=192, fwidth=48):
        fbanks = []
        for item in batch:
            X = item['fbank'].clone()  # 避免修改原始数据
            w, h = X.shape

            # 检查输入尺寸是否足够
            if w < twidth or h < fwidth:
                raise ValueError(f"Input size ({w}, {h}) is smaller than mask size ({twidth}, {fwidth}).")

            # 随机生成遮挡起始位置
            t_start = np.random.randint(0, w - twidth + 1)
            f_start = np.random.randint(0, h - fwidth + 1)

            # 应用遮挡
            X[t_start:t_start + twidth, f_start:f_start + fwidth] = 0
            fbanks.append(X)

        # 将所有 fbanks 和 targets 转为 Tensor
        fbanks = torch.stack(fbanks)
        targets = torch.tensor([item["target"] for item in batch], dtype=torch.long)

        return fbanks, targets

    def collate_fn(batch):
        fbanks = [item["fbank"] for item in batch]
        targets = [item["target"] for item in batch]
        return torch.stack(fbanks), torch.tensor(targets)

    trainloader = DataLoader(trainset, batch_size=config['batch_size'], num_workers=config['num_workers'], collate_fn=augment_fn, shuffle=True) 
    valloader = DataLoader(valset, batch_size=config['batch_size'], num_workers=config['num_workers'], collate_fn=collate_fn)
    return trainloader, valloader, num_classes

if __name__ == '__main__':
    # ##
    # trainset, valset = get_esc50_datasets(fold=1, num_proc=4)
    # print(f"--train/val: {len(trainset)}/{len(valset)}")

    # # 检查第一个样本
    # sample = trainset[0]
    # print(f"--fbank.shape: {sample['fbank'].shape}, target:{sample['target']}")

    ##
    config = {
        # 'dataset': 'ESC-50',
        'dataset': 'UrbanSound8K', 
        'fold': 1, 
        'num_workers': 4, 'batch_size': 3
        }
    trainloader, valloader, num_classes = get_dataloaders(config)
    X, y = next(iter(trainloader))

    import torch.nn as nn
    model = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, num_classes)),
        nn.Flatten(),
        nn.Linear(num_classes, num_classes)
    )
    print(f"--output.shape: {model(X).shape}, target:{y}")
