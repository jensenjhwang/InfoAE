import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import random


from models.simple_model import Encoder, Decoder
from models.mi_net import MINet
from simple_dataloader import get_data_new
from utils import *
from config import params, baseline_suffix
from pytorch_msssim import ssim


# if(params['dataset'] == 'MNIST'):
#     from models.mnist_model import Generator, Discriminator, DHead, QHead
# elif(params['dataset'] == 'SVHN'):
#     from models.svhn_model import Generator, Discriminator, DHead, QHead
# elif(params['dataset'] == 'CelebA'):
#     from models.celeba_model import Generator, Discriminator, DHead, QHead
# elif(params['dataset'] == 'FashionMNIST'):
#     from models.mnist_model import Generator, Discriminator, DHead, QHead

# Set random seed for reproducibility.
seed = 1123
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

# Use GPU if available.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

train_load, val_load, test_load = get_data_new('MNIST', params['batch_size'])

# Plot the training images.
sample_batch = next(iter(train_load))
plt.figure(figsize=(10, 10))
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(
    sample_batch[0].to(device)[ : 100], nrow=10, padding=2, normalize=True).cpu(), (1, 2, 0)))
plt.savefig(f"Training Images {params['dataset']}{baseline_suffix}")
plt.close('all')

# Initialise the network.
encoder = Encoder().to(device)
encoder.apply(weights_init)
print(encoder)

decoder = Decoder().to(device)
decoder.apply(weights_init)
print(decoder)

if not params['is_baseline']:
    minet_orig = MINet(1, 16).to(device)
    minet_orig.apply(weights_init)
    print(minet_orig)

    minet_comp = MINet(4, 16).to(device)
    minet_comp.apply(weights_init)
    print(minet_comp)

mse_loss = nn.MSELoss()

# Adam optimiser is used.
optimE = optim.Adam([{'params': encoder.parameters()}], lr=params['learning_rate'], betas=(params['beta1'], params['beta2']))
optimD = optim.Adam([{'params': decoder.parameters()}], lr=params['learning_rate'], betas=(params['beta1'], params['beta2']))

if not params['is_baseline']:
    optimM = optim.Adam([{'params': minet_orig.parameters()}, {'params': minet_comp.parameters()}], lr=params['learning_rate'], betas=(params['beta1'], params['beta2']))

# List variables to store results pf training.
img_list = []
D_losses = []
M_losses = []

print("-"*25)
print("Starting Training Loop...\n")
print('Epochs: %d\nDataset: {}\nBatch Size: %d\nLength of Data Loader: %d'.format(params['dataset']) % (params['num_epochs'], params['batch_size'], len(train_load)))
print("-"*25)

start_time = time.time()
iters = 0


for epoch in range(params['num_epochs']):
    epoch_start_time = time.time()

    for i, (data, _) in enumerate(train_load, 0):
        # Get batch size
        b_size = data.size(0)
        # Transfer data tensor to GPU/CPU (device)
        data = data.to(device)

        # Updating encoder and MINets  
        optimE.zero_grad()
        optimM.zero_grad()
        
        compressed = encoder(data)
        if not params['is_baseline']:
            d_orig = minet_orig(data).view(b_size, 16, -1)
            d_comp = minet_comp(compressed).view(b_size, 16, -1)
            mi_loss = donsker_varadhan_loss(d_comp, d_orig)
            mi_loss.backward()
            optimE.step()
            optimM.step() # is this right? need to check whether EM are minimizing or maximizing MI Loss

        # Update decoder
        optimD.zero_grad()
        if not params['is_baseline']:
            reconstructed = decoder(compressed.detach())
        else:
            reconstructed = decoder(compressed)
        decoder_loss = mse_loss(reconstructed, data)
        decoder_loss.backward()
        optimD.step()
        if params['is_baseline']:
            optimE.step()

        # Check progress of training.
        if i != 0 and i%100 == 0:
            if params['is_baseline']:
                print('[%d/%d][%d/%d]\tMI_Loss: %.4f\tDecoder Loss: %.4f'
                    % (epoch+1, params['num_epochs'], i, len(train_load), 
                        mi_loss.item(), decoder_loss.item()))
            else:
                print('[%d/%d][%d/%d]\tDecoder Loss: %.4f'
                    % (epoch+1, params['num_epochs'], i, len(train_load), 
                        decoder_loss.item()))

        # Save the losses for plotting.
        if not params['is_baseline']:
            M_losses.append(mi_loss.item())

        D_losses.append(decoder_loss.item())

        iters += 1
        print("Finished one iter")

    epoch_time = time.time() - epoch_start_time
    print("Time taken for Epoch %d: %.2fs" %(epoch + 1, epoch_time))
    # # Generate image after each epoch to check performance of the generator. Used for creating animated gif later.
    # with torch.no_grad():
    #     gen_data = netG(fixed_noise).detach().cpu()
    # img_list.append(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True))

    # # Generate image to check performance of generator.
    # if((epoch+1) == 1 or (epoch+1) == params['num_epochs']/2):
    #     with torch.no_grad():
    #         gen_data = netG(fixed_noise).detach().cpu()
    #     plt.figure(figsize=(10, 10))
    #     plt.axis("off")
    #     plt.imshow(np.transpose(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True), (1,2,0)))
    #     plt.savefig("Epoch_%d {}".format(params['dataset']) %(epoch+1))
    #     plt.close('all')

    # Save network weights.
    # if (epoch+1) % params['save_epoch'] == 0:
    #     torch.save({
    #         'netG' : netG.state_dict(),
    #         'discriminator' : discriminator.state_dict(),
    #         'netD' : netD.state_dict(),
    #         'netQ' : netQ.state_dict(),
    #         'optimD' : optimD.state_dict(),
    #         'optimG' : optimG.state_dict(),
    #         'params' : params
    #         }, 'checkpoint/model_epoch_%d_{}'.format(params['dataset']) %(epoch+1))

training_time = time.time() - start_time
print("-"*50)
print('Training finished!\nTotal Time for Training: %.2fm' %(training_time / 60))
print("-"*50)

# Save network weights
if params['is_baseline']:
     torch.save({
        'encoder' : encoder.state_dict(),
        'decoder' : decoder.state_dict(),
        'optimE' : optimE.state_dict(),
        'optimD' : optimD.state_dict(),
        'params' : params
        }, 'checkpoint/model_baseline_final_{}'.format(params['dataset']))
else:
    torch.save({
        'encoder' : encoder.state_dict(),
        'decoder' : decoder.state_dict(),
        'mi_net_orig' : mi_net_orig.state_dict(),
        'mi_net_comp' : mi_net_comp.state_dict(),
        'optimE' : optimE.state_dict(),
        'optimD' : optimD.state_dict(),
        'optimM' : optimM.state_dict(),
        'params' : params
        }, 'checkpoint/model_final_{}'.format(params['dataset']))


# test evaluation
with torch.no_grad():
    for test_data, labels in test_load:
        compressed = encoder(test_data)
        uncompressed = decoder(compressed) # ?

        mse = mse_loss(test_data, uncompressed)

        normalized_test_data = normalize_to_zero_one(test_data)
        normalized_uncompressed = normalize_to_zero_one(uncompressed)
        
        ssim_score = ssim(normalized_test_data, normalized_uncompressed, data_range=1, size_average=True)
    
        print("-"*50)
        print('Testing finished!')
        print(f'MSE: {mse}   SSIM:{ssim_score}')
        print("-"*50)

        # Visualize 
        visual_indices = [i * 1000 for i in range(10)]
        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(np.transpose(vutils.make_grid(compressed[visual_indices], nrow=10, padding=2, normalize=True), (1,2,0)))

        plt.savefig(f"images/Test_Visualization_{params['dataset']}_Epoch_{params['num_epochs']}{baseline_suffix}")


# # Generate image to check performance of trained generator.
# with torch.no_grad():
#     gen_data = netG(fixed_noise).detach().cpu()
# plt.figure(figsize=(10, 10))
# plt.axis("off")
# plt.imshow(np.transpose(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True), (1,2,0)))
# plt.savefig("Epoch_%d_{}".format(params['dataset']) %(params['num_epochs']))

# Save network weights.
# torch.save({
#     'netG' : netG.state_dict(),
#     'discriminator' : discriminator.state_dict(),
#     'netD' : netD.state_dict(),
#     'netQ' : netQ.state_dict(),
#     'optimD' : optimD.state_dict(),
#     'optimG' : optimG.state_dict(),
#     'params' : params
#     }, 'checkpoint/model_final_{}'.format(params['dataset']))


# Plot the training losses.
plt.figure(figsize=(10,5))
plt.title("MI and Decoder Loss During Training")
plt.plot(M_losses,label="M")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f"Loss Curve {params['dataset']}{baseline_suffix}")

# # Animation showing the improvements of the generator.
# fig = plt.figure(figsize=(10,10))
# plt.axis("off")
# ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
# anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
# anim.save('infoGAN_{}.gif'.format(params['dataset']), dpi=80, writer='imagemagick')
# plt.show()