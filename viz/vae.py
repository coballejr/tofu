# adapted from https://keras.io/examples/generative/vae/
import matplotlib.pyplot as plt
import numpy as np
import torch

# author: Jake VanderPlas
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

@torch.no_grad()
def plot_latent_space(vae, n=30, figsize=15, plot_dir = '.', epoch = 0):
    # display a n*n 2D manifold of digits
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder(torch.Tensor(z_sample))
            x_decoded = x_decoded.numpy()
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.savefig(plot_dir / f"decoded_{epoch}.png")
    plt.close()

@torch.no_grad()
def color_latent_space(vae, imgs, labels, nsamps = 500, plot_dir = '.', epoch = 0):
    imgs, labels = imgs[:nsamps,:], labels[:nsamps]
    labels = labels.cpu().numpy()
    enc = vae.encoder
    z, _, _ = enc(imgs)
    z = z.detach().cpu().numpy()

    scatter = plt.scatter(z[:,0], z[:,1], c=labels, cmap = discrete_cmap(10,'nipy_spectral'))
    plt.xlim([-4, 4])
    plt.ylim([-4,4])
    plt.title('Latent Space')
    plt.xlabel('$z_1$')
    plt.ylabel('$z_2$')
    plt.colorbar(scatter, ticks = [0,1,2,3,4,5,6,7,8,9])
    plt.gca().set_aspect('equal')
    plt.savefig(plot_dir / f"z_{epoch}.png")
    plt.close()

