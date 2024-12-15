tag = 'tmp'

is_vqvae = True
if is_vqvae:
    train_script_name = 'train'
else:
    train_script_name = 'train_vae'

###############Train#########################
enable_amp = True
dtype = "fp16"  # float32 fp16 bf16 fp8

num_gpus = 1
batch_size_per_gpu = 512
total_batch_size = batch_size_per_gpu * num_gpus

# world_size = num_gpus
print_freq = 500  # don't print too often
test_freq = print_freq * 50  # also the model-saving frequency
seed = 3507
compile = False
num_max_save_models = 5
test_distributed = True
test_before_train = False

# set to true if model is differ than original
use_gradient_ckpt = True  # unet_requires_grad
find_unused_parameters = False
# load_ckpt = '/'

###############Model EMA#####################
use_ema = False
ema_beta = 0.97  # ema_update_every=20
ema_update_every = 2 * print_freq  # every update cost about 7sec

###############Optimizer#####################
gradient_accumulation_steps = 1
warm_steps = 1000
total_steps = 1_000_000
max_epochs = 9999999
learning_rate = 1e-4
min_lr = 5e-6  # learning_rate / 10 usually
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95  # make a bit bigger when batch size per iter is small

#################Loss#########################
kl_weight = 0.00
vq_weight = 1.0


#################Net Structure#########################
#VQVAE
num_embeddings = 512
embedding_dim = 16
commitment_cost = 0.25

###############DataLoader#####################
num_dl_workers = 1  # number of dataloader processes per GPU(process)
train_dataset = []
test_dataset = []
