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
from english_words import english_words_lower_set



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
        self.num_classes = args.num_classes
        self.use_head = args.use_head
        self.use_encmlp = args.use_encmlp
        if args.use_head:
            self.head_dim = args.head_dim
            self.head_type = args.head_type
        if args.use_encmlp:
            self.encmlp_dim = args.encmlp_dim
            self.encmlp_layers = args.encmlp_layers
        self.use_randomtext = args.use_randomtext


        if not args.use_flyp_ft:
            self.clip_encoder = CLIP_Model(args, keep_lang=False)
        else:
            self.clip_encoder = CLIP_Model(args, keep_lang=True)
        #textual_classifier = get_classification_head(args, args.train_dataset)
        if self.clip_encoder is not None:
            self.train_preprocess = self.clip_encoder.train_preprocess
            self.val_preprocess = self.clip_encoder.val_preprocess
        self.textual_clf_weights = textual_clf_weights

        if args.model_type == 'ViT-L_14':
            self.num_features = 768
        elif args.model_type == 'ViT-B_16':
            self.num_features = 768
        elif args.model_type == 'RN50':
            self.num_features = 2048
        else:
            raise NotImplementedError

        #self.textual_classifier = nn.Linear(self.num_features, args.num_classes, bias=False)

        self.textual_clf_weights = self.textual_clf_weights[args.task_class_order]

        if not self.use_randomtext:
            textual_clf_head = ClassificationHead(normalize=True, weights=self.textual_clf_weights)
            self.textual_classifier = textual_clf_head
        else:
            temp_randomtext_embed = args.randomtext_embed
            rdtxt_idx = torch.randperm(len(temp_randomtext_embed))[:args.num_randomtext]
            #args.randomtext = [args.dic_randomtextp[i] for i in rdtxt_idx]
            self.randomtext_embed = temp_randomtext_embed[rdtxt_idx]

            temp_textual_clf_head = torch.cat([self.textual_clf_weights, self.randomtext_embed])
            self.textual_classifier = ClassificationHead(normalize=True, weights=temp_textual_clf_head)

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

    def encode(self, images, texts=None):
        if texts == None:
            return self.clip_encoder(images)
        else:
            return self.clip_encoder(images, texts)

    def update_text_clf(self, weights):
        self.textual_classifier = ClassificationHead(normalize=True, weights=weights)

    def calc_head(self, x, doenc=True):
        if doenc:
            x = self.encoder(x)
        x = F.normalize(self.head(x), dim=1)
        return x

    def forward(self, inputs, sess=None, train=False):
        # maybe only for CE-based loss calc
        # Because for FLYP, we use loss based on CLIP_loss,
        # which only use encode instead of calc to logits via classifiers for
        # training
        features = self.clip_encoder(inputs)
        outputs = self.textual_classifier(features)
        n_cls = self.base_class if sess == 0 else self.base_class + self.way * (sess)
        if train==False:
            outputs = outputs[:, :n_cls]
        else:
            if not self.use_randomtext:
                outputs = outputs[:, :n_cls]
            else:
                outputs = torch.cat([outputs[:,:n_cls], outputs[:, self.num_classes:]], dim=1)
        return outputs

class CLIP_Model(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()

        print(f'Loading {args.model_type} pre-trained weights.')
        self.model, self.train_preprocess, self.val_preprocess = load(
            args.model_type, args.device, jit=False)

        self.cache_dir = args.cache_dir
        self.keep_lang = keep_lang

        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images, texts=None):
        # usually use forward of self.model, which is already implemented
        # instead f forward here.
        if texts == None:
            return self.model.encode_image(images)
        else:
            return self.model.encode_image(images), self.model.encode_text(texts)


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




def word2textembed(word, args, model, template, device):
    texts = []
    for t in template:
        texts.append(t(word))
    texts = tokenize(texts).to(device)  # tokenize   #prompts x sentence -> #prompts x 77 (prompt embedding dimension)
    embeddings = model.encode_text(texts)  # embed with text encoder  #prompts x prompt_embed_dim -> #prompts x embed_dim
    embeddings /= embeddings.norm(dim=-1, keepdim=True)

    embeddings = embeddings.mean(dim=0, keepdim=True) #1 x embed_dim
    embeddings /= embeddings.norm()
    return embeddings

def temptokenize(args, template, word):
    texts = []
    for t in template:
        texts.append(t(word))
    texts = tokenize(texts).cuda()  # tokenize   #prompts x sentence -> #prompts x 77 (prompt embedding dimension)
    #texts = texts.mean(dim=0) #1 x embed_dim
    text = texts[0]
    return text

def lab_text_2weights(model, args, template, device, words):
    logit_scale = model.logit_scale

    model.eval()
    model.to(device)
    num_words = len(words)

    print('Building classification head.')
    with torch.no_grad():
        zeroshot_weights = []
        #for i in range(args.num_classes):
        for i in range(num_words):
        #for classname in tqdm(dataset.classnames):
            classname = words[i]
            embeddings = word2textembed(classname, args, model, template, device).squeeze() # (1,ch_dim) -> (ch_dim)
            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights *= logit_scale.exp()
    return zeroshot_weights

def get_words_weights(args, clip_model, words):

    print(f'Did not find classification head for '
          f'{args.model_type} on {args.dataset} at {args.text_clf_weight_fn}, building one from scratch.')

    template = get_templates(args.dataset)
    zeroshot_weights = lab_text_2weights(clip_model, args, template, args.device, words)
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

    def load_weight(self, weight):
        print(f'Loading weight for finetune model')
        self.weight = weight

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

