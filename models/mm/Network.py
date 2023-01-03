import argparse

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from models.resnet18_encoder_cifar import CIFAR_ResNet18, CIFAR_ResNet18_1, CIFAR_ResNet18_2, CIFAR_ResNet18_3
from models.supcon_resnet_cifar import supcon_resnet18
from vissl.utils.checkpoint import replace_module_prefix
from utils_s import *
from tqdm import tqdm


from clip.clip import load
from src.utils import torch_load, torch_save
#import clip.clip as clip
from clip.clip import tokenize
from src.datasets.templates import get_templates
from src.datasets.registry import get_dataset
from src.modeling import ClassificationHead, ImageEncoder



class MYNET(nn.Module):

    def __init__(self, args, fw_mode=None, textual_clf_weights=None):
        super().__init__()

        #self.new_mode = args.new_mode
        #self.temperature = args.temperature
        self.m = args.m
        self.s = args.s
        self.fw_mode = fw_mode
        self.base_class = args.base_class
        self.way = args.way
        self.use_head = args.use_head
        self.use_encmlp = args.use_encmlp
        if args.use_head:
            self.head_dim = args.head_dim
            self.head_type = args.head_type
        if args.use_encmlp:
            self.encmlp_dim = args.encmlp_dim
            self.encmlp_layers = args.encmlp_layers

        self.clip_img_encoder = CLIP_Model(args, keep_lang=False)
        #textual_classifier = get_classification_head(args, args.train_dataset)
        if self.clip_img_encoder is not None:
            self.train_preprocess = self.clip_img_encoder.train_preprocess
            self.val_preprocess = self.clip_img_encoder.val_preprocess
        self.textual_clf_weights = textual_clf_weights

        if args.model_type == 'ViT-L_14':
            self.num_features = 768
        else:
            raise NotImplementedError

        #self.textual_classifier = nn.Linear(self.num_features, args.num_classes, bias=False)


        self.textual_clf_weights = self.textual_clf_weights[args.task_class_order]
        textual_clf_head = ClassificationHead(normalize=True, weights=self.textual_clf_weights)
        self.textual_classifier = textual_clf_head





        """
        if not args.use_encmlp:
            self.fc = nn.Linear(self.num_features, args.num_classes, bias=False)
        else:
            self.fc = nn.Linear(self.encmlp_dim, args.num_classes, bias=False)
        """



        if args.use_head:
            if self.head_type == 'linear':
                self.head = nn.Linear(self.num_features, self.head_dim)
            elif self.head_type == 'mlp':
                self.head = nn.Sequential(
                    nn.Linear(self.num_features, self.num_features),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.num_features, self.head_dim)
                )
            else:
                raise NotImplementedError(
                    'head not supported: {}'.format(self.head_type))
        if args.use_encmlp:
            self.encmlp = projection_MLP(self.num_features, self.encmlp_dim, self.encmlp_layers)


        #self.fc.weight.data = abs(self.fc.weight.data)
        #nn.init.orthogonal_(self.fc.weight)

        #with torch.no_grad(): ### edit on 221009 for cosface debug mini
        #    self.fc.weight *= 2.321
        #nn.init.xavier_uniform_(self.fc.weight)

        if args.fw_mode == 'arcface':
            self.cos_m = math.cos(self.m)
            self.sin_m = math.sin(self.m)
            self.th = math.cos(math.pi - self.m)
            self.mm = math.sin(math.pi - self.m) * self.m

        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def freeze_head(self):
        self.textual_classifier.weight.requires_grad_(False)
        self.textual_classifier.bias.requires_grad_(False)



    #def __call__(self, inputs):
    #    return self.forward(inputs)

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return torch_load(filename)

    def set_mode(self, fw_mode):
        self.fw_mode = fw_mode

    def encode(self, x):
        x = self.encoder(x)
        if self.use_encmlp:
            x = self.encmlp(x)
        return x

    def calc_head(self, x, doenc=True):
        if doenc:
            x = self.encoder(x)
        x = F.normalize(self.head(x), dim=1)
        return x

    def forward(self, inputs, sess=None):
        features = self.clip_img_encoder(inputs)
        outputs = self.textual_classifier(features)
        n_cls = self.base_class if sess == 0 else self.base_class + self.way * (sess)
        outputs = outputs[:, :n_cls]
        return outputs
    """
    def forward(self, input, label=None, sess=None, doenc=True):
        if self.fw_mode == 'encoder':
            feat = self.encode(input)
            return feat
        elif self.fw_mode == 'fc_cos' or self.fw_mode == 'fc_dot':
            output = self.forward_fc(x=input, sess=sess, doenc=doenc)
            return output
        elif self.fw_mode == 'fc_cosface' or self.fw_mode == 'fc_arcface':
            logit, cos_logit = self.forward_fc(x=input, sess=sess, label=label, doenc=doenc)
            return logit, cos_logit
        elif self.fw_mode == 'head':
            output = self.calc_head(input, doenc=doenc)
            return output
        else:
            raise NotImplementedError
    """

    def forward_fc(self, x, sess, label=None, doenc=True):
        if doenc:
            x = self.encode(x)
        n_cls = self.base_class if sess==0 else self.base_class + self.way*(sess)
        #fc = self.fc[:n_cls]
        if self.fw_mode == 'fc_cos':
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            #x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))
            x = self.s * x
            x = x[:, :n_cls]
            return x
        elif self.fw_mode == 'fc_dot':
            x = self.fc(x)
            x = x[:, :n_cls]
            return x
        elif self.fw_mode == 'fc_cosface':
            cos = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            cos = cos[:, :n_cls]
            phi = cos - self.m
            # --------------------------- convert label to one-hot ---------------------------

            B = x.shape[0]
            one_hot = torch.arange(n_cls).expand(B, n_cls).cuda()
            label_ = label.unsqueeze(1).expand(B, n_cls)
            one_hot = torch.where(one_hot == label_, 1, 0)
            output = (one_hot * phi) + ((1.0 - one_hot) * cos)
        elif self.fw_mode == 'fc_arcface':
            cos = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            cos = cos[:, :n_cls]
            sine = torch.sqrt((1.0 - torch.pow(cos, 2)).clamp(0, 1))
            phi = cos * self.cos_m - sine * self.sin_m

            phi = torch.where(cos > self.th, phi, cos - self.mm)
            # --------------------------- convert label to one-hot ---------------------------
            # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
            one_hot = torch.zeros(cos.size(), device='cuda')
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
            output = (one_hot * phi) + (
                    (1.0 - one_hot) * cos)
        else:
            raise NotImplementedError

        output *= self.s
        return output, cos


class CLIP_Model(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()

        print(f'Loading {args.model_type} pre-trained weights.')
        self.model, self.train_preprocess, self.val_preprocess = load(
            args.model_type, args.device, jit=False)

        self.cache_dir = args.cache_dir

        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving image encoder to {filename}')
        torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image encoder from {filename}')
        return torch_load(filename)


class TextualClassifier(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading classification head from {filename}')
        return torch_load(filename)


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def freeze_head(self):
        self.classification_head.weight.requires_grad_(False)
        self.classification_head.bias.requires_grad_(False)

    def forward(self, inputs):
        features = self.image_encoder(inputs)
        outputs = self.classification_head(features)
        return outputs

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return torch_load(filename)


def build_zeroshot_weights(model, args, template, device):
    dataset_name = args.dataset
    template = get_templates(dataset_name)

    logit_scale = model.logit_scale
    model.eval()
    model.to(device)

    print('Building classification head.')
    with torch.no_grad():
        zeroshot_weights = []
        for i in range(args.num_classes):
        #for classname in tqdm(dataset.classnames):
            classname = args.dataset_label2txt[i]
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = tokenize(texts).to(device)  # tokenize
            embeddings = model.encode_text(texts)  # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()

        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)
    return zeroshot_weights



def get_zeroshot_weights(args):

    print(f'Did not find classification head for {args.model_type} on {args.dataset} at {args.text_clf_weight_fn}, building one from scratch.')

    model = CLIP_Model(args, keep_lang=True).model
    template = get_templates(args.dataset)
    zeroshot_weights = build_zeroshot_weights(model, args, template, args.device)
    return zeroshot_weights


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading classification head from {filename}')
        return torch_load(filename)


class projection_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2):
        super().__init__()
        hidden_dim = out_dim

        self.num_layers = num_layers

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=False)  # Page:5, Paragraph:2
        )

    def forward(self, x):
        if self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        elif self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        return x

