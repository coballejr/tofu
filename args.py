from argparse import ArgumentParser
from pathlib import Path
import random
import torch
import numpy as np

class Parser(ArgumentParser):

    def __init__(self):
        super(Parser, self).__init__(description='Read')
        self.add_argument('--experiment', type = str, default = 'mnist',
                          choices = ['mnist'], help = 'experiment name')
        self.add_argument('--model', type = str, default = 'vanilla',
                          choices = ['vanilla', 'tofu'], help = 'experiment name')

        # data
        self.add_argument('--train_batch', type=int, default= 50,
                          help='training batch size')
        self.add_argument('--test_batch', type=int, default= 1000,
                          help='testing batch size')
        # training
        self.add_argument('--epoch-start', type=int, default=0, help='epoch to start at, will load pre-trained network')
        self.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
        self.add_argument('--lr', type=float, default= 1e-3, help='ADAM learning rate')
        self.add_argument('--seed', type=int, default=12345, help='manual seed used in PyTorch and Numpy')


    def parse(self):
            args = self.parse_args()

            args.run_dir = Path('./outputs')/ f'{args.experiment}' \
                / f'{args.model}'

            args.ckpt_dir = args.run_dir / "checkpoints"
            args.pred_dir = args.run_dir / "predictions"
            for path in (args.run_dir, args.ckpt_dir, args.pred_dir):
                Path(path).mkdir(parents=True, exist_ok=True)

            # Set random seed
            if args.seed is None:
                args.seed = random.randint(1, 10000)
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            np.random.seed(seed=args.seed)

            return args

