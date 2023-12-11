import argparse
import random

from transformer_model import TransformerModel
from test import *

parser = argparse.ArgumentParser()

parser.add_argument('--epoch_num', type=int, default=200, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=6e-4, help='initial learning rate')
parser.add_argument('--eta_min', type=float, default=1e-4, help='minimum learning rate for CosineAnnealingWarmRestarts')
parser.add_argument('--target_metric', type=str, default='GDI', help='target metric')
parser.add_argument('--sides', type=int, default=2, help='number of sides')

args = parser.parse_known_args()[0]

if args.target_metric.lower() == "kfme":
    args.target_metric = "KneeFlex_maxExtension"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(77655)

print(f'Target Metric = {args.target_metric}')

model = TransformerModel(args = args)
model.train()

test(dataset = model.dataset, num_epochs = args.epoch_num, model = model.model, criterion = model.criterion, optimizer = model.optimizer)