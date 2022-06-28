from args import Parser
from data.mnist import create_training_loader as mnist_training_loader
from data.mnist import create_testing_loader as mnist_testing_loader
from models.vanilla.vae import VAE as vae_vanilla
from models.tofu.vae import ToFUVAE as vae_tofu
from loss.vae_loss import VAELoss
from viz.vae import plot_latent_space, color_latent_space
import torch
from torch.optim.lr_scheduler import ExponentialLR

if __name__ == '__main__':
    args = Parser().parse()

    # set experiment
    if args.experiment.lower() == 'mnist':
        training_loader = mnist_training_loader(args.train_batch)
        testing_loader = mnist_testing_loader(args.test_batch)
        criterion = VAELoss()
        if args.model.lower() == 'vanilla':
            model = vae_vanilla()
        elif args.model.lower() == 'tofu':
            model = vae_tofu()
        else:
            raise NotImplementedError('Invalid model')

    else:
        raise NotImplementedError('Invalid experiment.')

    # set optimizers
    opt = torch.optim.Adam(model.parameters(), lr = args.lr)
    scheduler = ExponentialLR(opt, gamma = 0.999)

    # train
    for ep in range(args.epochs):
        print('Epoch ', ep, ' start...')
        ep_loss = 0
        for mbi, (imgs, labels) in enumerate(training_loader):
            x = imgs.squeeze().flatten(start_dim = 1)
            opt.zero_grad()
            _, mu, logvar, x_recon = model(imgs)
            loss = criterion(mu, logvar, x, x_recon)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        scheduler.step()
        print('Epoch ', ep, 'finish!', ' Loss: ', ep_loss)

        # visualize decoder output
        plot_latent_space(model, plot_dir = args.pred_dir, epoch = ep)
        for mbi, (imgs, labels) in enumerate(testing_loader):
            imgs, labels = imgs, labels
        color_latent_space(model, imgs, labels, plot_dir = args.pred_dir, epoch = ep)
