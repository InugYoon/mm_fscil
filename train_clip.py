import os

from src.eval import evaluate
from src.finetune import finetune
from src.modeling import ImageEncoder
from src.args import parse_arguments
from src.patch import *

if __name__ == '__main__':
    args = parse_arguments()
    patch(args)


