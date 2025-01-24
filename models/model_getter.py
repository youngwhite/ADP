# imported at root
import torch.nn as nn
from models.AST.ast_ import WrappedAST

def get_model(model_name: str, num_classes: int):
    if model_name == 'ast':
        model = WrappedAST(num_classes=num_classes)
    elif model_name == 'beats':
        pass
    elif model_name == 'toy':
        model = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, num_classes)),
            nn.Flatten(),
            nn.Linear(num_classes, num_classes)
        )
        model.num_layers = 1
    else:
        raise ValueError(f'Invalid model: {model_name}')
    return model

if __name__ == '__main__':
    import torch
    x = torch.rand(4, 1024, 128)

    model = get_model(model_name='ast', num_classes=50)
    print('--num_layers:', model.num_layers)

    outputs = model(x)
    print('outputs.shape:', outputs.shape)
