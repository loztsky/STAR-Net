import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts, StepLR
from torch.optim import SGD, Adam, AdamW
from starnet.utils.optimizer import MuonWithAuxAdam, SingleDeviceMuon, SingleDeviceMuonWithAuxAdam

from starnet.models import TMVANet,\
                         MVNet, \
                         MVANet,\
                         STARNet
                         

def build_scheduler(schedular_type : str, optimizer : torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    if schedular_type == "step":
        return StepLR(optimizer, step_size=1, gamma=0.1)
    
    elif schedular_type == "exp":
        return ExponentialLR(optimizer, gamma=0.9)
    
    elif schedular_type == "coswarm":
        return CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=2, eta_min=1e-7)

def build_optimizer(optimizer_type: str, net: nn.Module, lr: float, **kwargs) -> torch.optim.Optimizer:
    """
    构建优化器
    
    Args:
        optimizer_type: 优化器类型 ('Adam', 'SGD', 'AdamW', 'Muon')
        net: 神经网络模型
        lr: 学习率
        **kwargs: 其他优化器参数
    
    Returns:
        torch.optim.Optimizer: 优化器实例
    """
    
    if optimizer_type == 'Adam':
        return Adam(net.parameters(), lr=lr, **kwargs)
    
    elif optimizer_type == 'SGD':
        return SGD(net.parameters(), lr=lr, **kwargs)
    
    elif optimizer_type == 'AdamW':
        return AdamW(net.parameters(), lr=lr, **kwargs)
    
    elif optimizer_type == 'Muon':
        return build_muon_optimizer(net, lr, **kwargs)
    
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}") 

def build_model(cfg) -> nn.Module:
    # original mvnet、mvanet、tmvanet
    if cfg['model'] == 'mvnet':
        net = MVNet(n_classes=cfg['nb_classes'],
                    n_frames=cfg['nb_input_channels'])
        
    elif cfg['model'] == 'mvanet':
        net = MVANet(n_classes=cfg['nb_classes'],
                     n_frames=cfg['nb_input_channels'])
    elif cfg["model"] == "tmvanet":
        net = TMVANet(n_classes=cfg['nb_classes'],
                           n_frames=cfg['nb_input_channels'],
                           hidden_channels=cfg['hidden_channels'],
                           num_layers=cfg['num_layers'])
    elif cfg["model"] == "starnet":
        net = STARNet(n_classes=cfg['nb_classes'],
                           n_frames=cfg['nb_input_channels'],
                           hidden_channels=cfg['hidden_channels'],
                           num_layers=cfg['num_layers'])

    return net


def build_muon_optimizer(net: nn.Module, lr: float, **kwargs):
    """
    构建单设备Muon优化器,自动分类参数
    
    Args:
        net: 神经网络模型
        lr: 基础学习率
        **kwargs: 其他参数
            - muon_lr_scale: Muon学习率缩放因子 (默认: 1.0)
            - embed_lr_scale: 嵌入层学习率缩放因子 (默认: 10.0)
            - head_lr_scale: 输出层学习率缩放因子 (默认: 4.0)
            - scalar_lr_scale: 标量参数学习率缩放因子 (默认: 1.0)
            - momentum: Muon动量 (默认: 0.95)
            - betas: Adam的beta参数 (默认: (0.8, 0.95))
            - eps: Adam的epsilon (默认: 1e-10)
            - weight_decay: 权重衰减 (默认: 0.0)
    
    Returns:
        SingleDeviceMuonWithAuxAdam优化器实例
    """
    
    # 提取参数
    muon_lr_scale = kwargs.get('muon_lr_scale', 1.0)
    embed_lr_scale = kwargs.get('embed_lr_scale', 10.0)
    head_lr_scale = kwargs.get('head_lr_scale', 4.0)
    scalar_lr_scale = kwargs.get('scalar_lr_scale', 1.0)
    momentum = kwargs.get('momentum', 0.95)
    betas = kwargs.get('betas', (0.8, 0.95))
    eps = kwargs.get('eps', 1e-10)
    weight_decay = kwargs.get('weight_decay', 0.0)
    
    # 参数分类
    param_groups = classify_parameters(net, lr, muon_lr_scale, embed_lr_scale, 
                                     head_lr_scale, scalar_lr_scale, 
                                     momentum, betas, eps, weight_decay)
    
    # 使用单设备版本
    return SingleDeviceMuonWithAuxAdam(param_groups)


def classify_parameters(net: nn.Module, base_lr: float, muon_lr_scale: float,
                       embed_lr_scale: float, head_lr_scale: float, scalar_lr_scale: float,
                       momentum: float, betas: tuple, eps: float, weight_decay: float):
    """
    自动分类网络参数
    """
    
    # 参数容器
    hidden_matrix_params = []  # 隐藏层矩阵参数 -> Muon
    embed_params = []          # 嵌入层参数 -> Adam
    head_params = []           # 输出层参数 -> Adam
    scalar_params = []         # 标量参数 -> Adam
    
    # 遍历所有参数进行分类
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue
            
        # 嵌入层参数
        if is_embedding_param(name, param):
            embed_params.append(param)
        
        # 输出层参数（通常是最后的linear层）
        elif is_head_param(name, param, net):
            head_params.append(param)
        
        # 标量参数（偏置、LayerNorm等）
        elif param.ndim < 2:
            scalar_params.append(param)
        
        # 隐藏层矩阵参数
        elif param.ndim >= 2:
            # 排除嵌入层和输出层的矩阵参数
            if not is_embedding_param(name, param) and not is_head_param(name, param, net):
                hidden_matrix_params.append(param)
        
        else:
            # 默认归类为标量参数
            scalar_params.append(param)
    
    # 构建参数组
    param_groups = []
    
    # Adam组
    if embed_params:
        param_groups.append({
            'params': embed_params,
            'lr': base_lr * embed_lr_scale,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'use_muon': False
        })
    
    if head_params:
        param_groups.append({
            'params': head_params,
            'lr': base_lr * head_lr_scale,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'use_muon': False
        })
    
    if scalar_params:
        param_groups.append({
            'params': scalar_params,
            'lr': base_lr * scalar_lr_scale,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'use_muon': False
        })
    
    # Muon组
    if hidden_matrix_params:
        param_groups.append({
            'params': hidden_matrix_params,
            'lr': base_lr * muon_lr_scale,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'use_muon': True
        })
    
    return param_groups


def is_embedding_param(name: str, param: torch.Tensor) -> bool:
    """判断是否为嵌入层参数"""
    embed_keywords = ['embed', 'embedding', 'wte', 'word_embed']
    return any(keyword in name.lower() for keyword in embed_keywords)


def is_head_param(name: str, param: torch.Tensor, net: nn.Module) -> bool:
    """判断是否为输出层参数"""
    head_keywords = ['head', 'lm_head', 'classifier', 'fc', 'output']
    
    # 基于名称判断
    if any(keyword in name.lower() for keyword in head_keywords):
        return True
    
    # 基于模块结构判断（通常是最后几层）
    try:
        # 获取所有命名模块
        modules = list(net.named_modules())
        if len(modules) > 1:
            # 检查是否是最后几个线性层
            last_modules = modules[-3:]  # 检查最后3个模块
            for mod_name, module in last_modules:
                if isinstance(module, (nn.Linear, nn.Conv2d)) and mod_name in name:
                    return True
    except:
        pass
    
    return False



# 使用示例和测试
def test_build_optimizer():
    """测试优化器构建函数"""
    
    # 创建一个简单的测试模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(1000, 128)
            self.layers = nn.ModuleList([
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.LayerNorm(256),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.LayerNorm(128)
            ])
            self.head = nn.Linear(128, 10)
        
        def forward(self, x):
            x = self.embed(x)
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    x = layer(x)
                elif hasattr(layer, 'forward'):
                    x = layer(x)
            return self.head(x)
    
    model = TestModel()
    
    # 测试不同优化器
    optimizers = {
        'Adam': build_optimizer('Adam', model, 0.001),
        'SGD': build_optimizer('SGD', model, 0.01, momentum=0.9),
        'AdamW': build_optimizer('AdamW', model, 0.001, weight_decay=0.01),
        'Muon': build_optimizer('Muon', model, 0.02)
    }
    
    for name, opt in optimizers.items():
        print(f"{name} optimizer created successfully")
        if name == 'Muon':
            print(f"  - Parameter groups: {len(opt.param_groups)}")
            for i, group in enumerate(opt.param_groups):
                use_muon = group.get('use_muon', 'N/A')
                lr = group['lr']
                param_count = sum(p.numel() for p in group['params'])
                print(f"    Group {i}: use_muon={use_muon}, lr={lr}, params={param_count}")


if __name__ == "__main__":
    test_build_optimizer()