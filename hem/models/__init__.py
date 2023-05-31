from hem.models.trainer import Trainer


def get_model(name):
    if name == 'base':
        from .basic_embedding import BasicEmbeddingModel
        return BasicEmbeddingModel
    elif name == 'base rnn':
        from .basic_rnn import BaseRNN
        return BaseRNN
    elif name == 'conv1D':
        from .basic_rnn import Conv1D
        return Conv1D
    elif name == 'resnet':
        from .basic_embedding import ResNetFeats
        return ResNetFeats
    elif name == 'r3m':
        from .basic_embedding import R3M
        return R3M
    elif name == 'vgg':
        from .basic_embedding import VGGFeats
        return VGGFeats
    elif name == 'base traj':
        from .traj_embed import BaseTraj
        return BaseTraj
    elif name == 'attention goal state':
        from .traj_embed import AttentionGoalState
        return AttentionGoalState
    elif name == 'goal state':
        from .traj_embed import GoalState
        return GoalState
    elif name == 'simple':
        from .basic_embedding import SimpleSpatialSoftmax
        return SimpleSpatialSoftmax
    elif name == "basic":
        from .basic_embedding import BasicCNN
        return BasicCNN
    raise NotImplementedError
