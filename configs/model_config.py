import sys

sys.path.append('.')
sys.path.append('..')
from models.encoders.resnet152 import ResNet152_StyleEncoder

encoder_list = {
    'resnet152': ResNet152_StyleEncoder
}

pretrained_information = {
    'FFHQ256': {
        'path': '../pretrained_models/ffhq256.pkl',
        'size': 256
    },
    'Church256': {
        'path': '../pretrained_models/church_stylegan2.pkl',
        'size': 256
    },
}
