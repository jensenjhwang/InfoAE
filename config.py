# Dictionary storing network parameters.
params = {
    'batch_size': 2,# Batch size.
    'num_epochs': 0,# Number of epochs to train for.
    'learning_rate': 2e-4,# Learning rate.
    'beta1': 0.5,
    'beta2': 0.999,
    'save_epoch' : 25,# After how many epochs to save checkpoints and generate test output.
    'dataset' : 'MNIST',# Dataset to use. Choose from {MNIST, SVHN, CelebA, FashionMNIST}. CASE MUST MATCH EXACTLY!!!!!
    'is_baseline': False,
    'mi_size': 2,
    'mi_scaling': 1, # 1e-4
    'train_separately': True, # if True, this trains the encoder completely before training the decoder.
    'experiment_prefix': "exp" # prefix for experiment directory
}