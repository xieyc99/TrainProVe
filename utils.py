from diffusers import DDPMPipeline
from diffusers import StableDiffusionPipeline, KandinskyV22PriorPipeline, KandinskyV22Pipeline, DiTPipeline, StableDiffusion3Pipeline, KandinskyV22CombinedPipeline
from diffusers import AutoPipelineForText2Image, LatentConsistencyModelPipeline, DiffusionPipeline, Kandinsky3Pipeline, UniDiffuserPipeline
import torch
import numpy as np
import os
import random
from torchvision.models import resnet50
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import yaml
import torch.nn as nn
from kornia import augmentation as aug
import torchvision
from kornia.augmentation.container import AugmentationSequential
# from glide_text2im.download import load_checkpoint
# from glide_text2im.model_creation import (
#     create_model_and_diffusion,
#     model_and_diffusion_defaults,
#     model_and_diffusion_defaults_upsampler
# )
import torch.nn.functional as F
# from models.vit_pytorch import ViT
from scipy.stats import t
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from diffusers import (
    StableCascadeDecoderPipeline,
    StableCascadePriorPipeline,
    StableCascadeUNet,
)

def get_class_names(dataset):
    if dataset == 'cifar10':
        classes = ['airplane',
                    'automobile',
                    'bird',
                    'cat',
                    'deer',
                    'dog',
                    'frog',
                    'horse',
                    'ship',
                    'truck']
        
    elif dataset == 'cifar100':
        classes = ['apple','aquarium_fish','baby','bear','beaver','bed','bee',
                        'beetle','bicycle','bottle','bowl','boy','bridge','bus',
                        'butterfly','camel','can','castle','caterpillar','cattle','chair',
                        'chimpanzee','clock','cloud','cockroach','couch','crab','crocodile',
                        'cup','dinosaur','dolphin','elephant','flatfish','forest','fox',
                        'girl','hamster','house','kangaroo','keyboard','lamp','lawn_mower',
                        'leopard','lion','lizard','lobster','man','maple_tree','motorcycle',
                        'mountain','mouse','mushroom','oak_tree','orange','orchid','otter',
                        'palm_tree','pear','pickup_truck','pine_tree','plain','plate','poppy',
                        'porcupine','possum','rabbit','raccoon','ray','road','rocket',
                        'rose','sea','seal','shark','shrew','skunk','skyscraper',
                        'snail','snake','spider','squirrel','streetcar','sunflower','sweet_pepper',
                        'table','tank','telephone','television','tiger','tractor','train',
                        'trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm']

    return classes

class Subset(torch.utils.data.Subset):
    """Overwrite subset class to provide class methods of main class."""

    def __getattr__(self, name):
        """Call this only if all attributes of Subset are exhausted."""
        return getattr(self.dataset, name)

def get_model(model_type):
    if model_type == "ddpm_cifar10":
        model_id = "google/ddpm-cifar10-32"
        cur_model = DDPMPipeline.from_pretrained(model_id).to("cuda")
        cur_model.unet.eval()
    
    elif model_type in ["sd1_4"]:
        model_id = "CompVis/stable-diffusion-v1-4"
        # model_id = r"C:\Users\1\.cache\huggingface\hub\models--CompVis--stable-diffusion-v1-4\snapshots\133a221b8aa7292a167afc5127cb63fb5005638b"
        cur_model = AutoPipelineForText2Image.from_pretrained(model_id,torch_dtype=torch.float32).to("cuda")
        cur_model.unet.eval()
        cur_model.vae.eval()

    elif model_type in ["sd1_5"]:
        model_id = "runwayml/stable-diffusion-v1-5"
        cur_model = AutoPipelineForText2Image.from_pretrained(model_id,torch_dtype=torch.float32).to("cuda")
        cur_model.unet.eval()
        cur_model.vae.eval()

    elif model_type in ["sd2"]:
        model_id = "stabilityai/stable-diffusion-2-base"
        cur_model = AutoPipelineForText2Image.from_pretrained(model_id,torch_dtype=torch.float32).to("cuda")
        cur_model.unet.eval()
        cur_model.vae.eval()

        for param in cur_model.unet.parameters():
            param.requires_grad = False
        
        for param in cur_model.vae.parameters():
            param.requires_grad = False

    elif model_type in ["sd3"]:
        model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
        cur_model = StableDiffusion3Pipeline.from_pretrained(model_id,torch_dtype=torch.float32).to("cuda")
        # cur_model = DiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float32).to("cuda")
        # cur_model.unet.eval()
        # cur_model.vae.eval()

    elif model_type in ["kds2_2"]:
        pipe_prior = KandinskyV22PriorPipeline.from_pretrained(r"C:\Users\1\.cache\huggingface\hub\models--kandinsky-community--kandinsky-2-2-prior\snapshots\9fc51ad5732afc5d031724219d22e6c42179c5a8").to("cuda")

        # model_id = "kandinsky-community/kandinsky-2-2-decoder"
        model_id = r"C:\Users\1\.cache\huggingface\hub\models--kandinsky-community--kandinsky-2-2-decoder\snapshots\9ae140d347fed8ce6e8bb3005dcc1f48543bb8e3"
        # cur_model = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float32).to("cuda")
        # cur_model = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16).to("cuda")
        cur_model = KandinskyV22Pipeline.from_pretrained(model_id,torch_dtype=torch.float32).to("cuda")
        cur_model.unet.eval()
    
    elif model_type in ["kds3"]:
        model_id = "kandinsky-community/kandinsky-3"
        # model_id = r"C:\Users\1\.cache\huggingface\hub\models--kandinsky-community--kandinsky-3\snapshots\bf79e6c219da8a94abb50235fdc4567eb8fb4632"
        # cur_model = AutoPipelineForText2Image.from_pretrained(model_id,torch_dtype=torch.float32).to("cuda")
        cur_model = Kandinsky3Pipeline.from_pretrained(model_id,torch_dtype=torch.float32).to("cuda")
        cur_model.unet.eval()
        cur_model.movq.eval()

    elif model_type in ["lcm"]:
        model_id = "SimianLuo/LCM_Dreamshaper_v7"
        cur_model = AutoPipelineForText2Image.from_pretrained(model_id,torch_dtype=torch.float32).to("cuda")
        cur_model.unet.eval()
        cur_model.vae.eval()

    elif model_type in ["dit"]:
        model_id = "facebook/DiT-XL-2-256"
        cur_model = DiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float32).to("cuda")
        # cur_model.unet.eval()
        # cur_model.vae.eval()

    elif model_type in ["pixart"]:
        model_id = "PixArt-alpha/PixArt-XL-2-512x512"
        cur_model = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float32).to("cuda")
    
    elif model_type in ["unidiff"]:
        model_id = "thu-ml/unidiffuser-v1"
        cur_model = UniDiffuserPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to("cuda")

    elif model_type in ["cascade"]:
        prior_unet = StableCascadeUNet.from_pretrained("stabilityai/stable-cascade-prior", subfolder="prior_lite").to("cuda")
        decoder_unet = StableCascadeUNet.from_pretrained("stabilityai/stable-cascade", subfolder="decoder_lite").to("cuda")

        pipe_prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", prior=prior_unet).to("cuda")
        cur_model = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", decoder=decoder_unet).to("cuda")

    cur_model.safety_checker = lambda images, clip_input: (images, None)
    
    if model_type == 'kds2_2' or model_type == 'cascade':
        return pipe_prior, cur_model
    else:
        return cur_model


def get_init_noise(model_type, model=None, bs=1):
    if model_type in ["ddpm_cifar10"]:
        init_noise = torch.randn([bs, model.unet.config.in_channels, model.unet.config.sample_size, model.unet.config.sample_size]).cuda()
    elif model_type in ["dcgan_cifar10"]:
        init_noise = torch.randn([bs, model.nz, 1, 1]).cuda()
    elif model_type in ["styleganv2ada_cifar10"]:
        init_noise = torch.randn([bs, model.z_dim]).cuda()
    elif model_type in ["vae_cifar10"]:
        init_noise = torch.randn([bs, model.latent_dim]).cuda()
    elif model_type in ["sd1_4", "sd1_5", "sd2", "sd3"]:
        height = model.unet.config.sample_size * model.vae_scale_factor
        width = model.unet.config.sample_size * model.vae_scale_factor
        init_noise = torch.randn([bs, model.unet.config.in_channels, height // model.vae_scale_factor, width // model.vae_scale_factor]).cuda()
    elif model_type in ["kds2_2", "kds3"]:
        # init_noise = torch.randn([bs, 1280]).cuda()  # for combine
        init_noise = torch.randn([bs, model.unet.config.in_channels, 64, 64]).cuda()
    elif model_type in ["lcm"]:
        height = model.unet.config.sample_size * model.vae_scale_factor
        width = model.unet.config.sample_size * model.vae_scale_factor
        init_noise = torch.randn([bs, model.unet.config.in_channels, height // model.vae_scale_factor, width // model.vae_scale_factor]).cuda()
    elif model_type in ["dit"]:
        latent_size = model.transformer.config.sample_size
        latent_channels = model.transformer.config.in_channels
        init_noise = torch.randn([1, latent_channels, latent_size, latent_size], dtype=model.transformer.dtype).cuda()
    elif model_type in ['cm']:
        init_noise = torch.randn(*(bs, 3, 64, 64)).cuda()
    elif model_type in ['glide']:
        init_noise = torch.randn([bs*2, 3, 64, 64]).cuda()
    elif model_type in ['pixart']:
        init_noise = torch.randn([bs, 4, 64, 64]).cuda()
    elif model_type in ['unidiff']:
        init_noise = torch.randn([bs, 21824]).cuda()

    return init_noise

def set_seed(seed):
    # for reproducibility. 
    # note that pytorch is not completely reproducible 
    # https://pytorch.org/docs/stable/notes/randomness.html  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.initial_seed() # dataloader multi processing 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_model(args, num_classes, img_size):
    if args.data_source == 'real' and args.dataset=='imagenet':
        if args.arch == 'RS50':
            net = resnet50(pretrained=True)
    else:
        ckpt = torch.load(args.model_path, "cuda")
        
        if args.arch == 'RS50':
            net = resnet50(num_classes=num_classes)
            # if args.dataset == 'imagenet' or 'imagenet100':
            #     net.fc = torch.nn.Linear(2048, num_classes, bias=False)  # change 1000 to 100 for "imagenet_100_sd.pth"
            # elif args.dataset == 'cifar10' or 'cifar100':
            #     net.fc = torch.nn.Linear(2048, num_classes, bias=True)
        elif args.arch == 'Vit-B':
            from models.vit_pytorch import ViT
            net = ViT(image_size=img_size, patch_size=16, num_classes=num_classes, dim=768 , depth=12, heads=12, mlp_dim=768*4)

        if 'state_dict' in ckpt.keys():
            ckpt['state_dict'] = {k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()}
            msg = net.load_state_dict(ckpt['state_dict'], strict=True)
        else:
            msg = net.load_state_dict(ckpt, strict=True)  
        print('msg:', msg)

    return net

def get_prompt(dataset, sample_size):
    prompts, names, all_classes, name2id = {}, [], [], {}

    if 'imagenet' in dataset:
        with open('data\ImageNet100\ImageNet-100.txt', "r") as file:
            class_id = [line.strip() for line in file]
        # class_id = os.listdir(rf'D:\Exp\datasets\{dataset}\val')
        # print(class_id)
        class_name, leaf_info = {}, {}

        with open(r'./imagenet-wordnet/class_dict.yaml', 'r') as file:
            class_dict = yaml.safe_load(file)
        
        # with open(r'D:\Exp\datasets\imagenet-wordnet\class_discribe.yaml', 'r') as file:
        #     class_des_dict = yaml.safe_load(file)

        with open('./imagenet-wordnet/class_name.txt', 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line != '':
                    id = line[line.find('n'):line.find('n')+9]
                    name = line.split(', ')[1:]
                    class_name[id] = name

        with open('./imagenet-wordnet/leaf_info.txt', 'r', encoding='utf-8') as file:
            tmp_meta_class = ''
            tmp_describe = ''
            for i, line in enumerate(file):
                line = line.strip()
                if i % 4 == 0:
                    tmp_meta_class = line.split(',')[0]
                elif (i-1) % 4 == 0:
                    tmp_describe = line.split(';')[0]
                    leaf_info[tmp_meta_class] = tmp_describe
        
        indices = np.random.permutation(len(class_id))
        selected_indices = indices[:sample_size]

        sample_classes = [class_id[index] for index in selected_indices]

        for i in range(sample_size):
            name = class_name[sample_classes[i]][0]
            name2id[name] = sample_classes[i]
            names.append(name)
            meta_class = class_dict[sample_classes[i]]['words'][0]
            describe = leaf_info[meta_class]
            prompts[name] = []
            # prompts[name].append(name)
            # prompts[name].append('a photo of a '+name)
            # prompts[name].append('a painting of '+name)
            # prompts[name].append('a drawing of '+name)
            # prompts[name].append('a poster of '+name)
            # prompts[name].append('a black and white sketch figure of '+name)
            # prompts[name].append(class_des_dict[class_id[selected_indices[i]]])
            # prompts[name].append('a cartoon of '+name)
            prompts[name].append(name+', '+describe)
            # prompts[name].append(name+', '+describe+', cartoon style')
            # prompts[name].append(name+', '+describe+', black and white sketch style')

        for i in range(len(class_id)):
            name = class_name[class_id[i]][0]
            all_classes.append(name)

        return prompts, sample_classes, names, selected_indices, all_classes, name2id, class_name

    elif dataset == 'cifar10':
        all_classes = ['airplane',
                    'automobile',
                    'bird',
                    'cat',
                    'deer',
                    'dog',
                    'frog',
                    'horse',
                    'ship',
                    'truck']
        indices = np.random.permutation(len(all_classes))
        selected_indices = indices[:sample_size]

        sample_classes = [all_classes[index] for index in selected_indices]
        # print(sample_classes)

        # Prompt to GPT4o: With each class in cifar10 as the central word, create a sentence that describes a surreal scene in detail and is not too complicated.
        d = {'airplane':"An airplane made of candy cane stripes flies gracefully through a sky filled with floating jellyfish.",
            'automobile':"An automobile with rainbow-colored wheels drives across a bridge made entirely of sparkling diamonds.",
            'bird':"A bird with luminous feathers sings a melody that causes flowers to bloom in mid-air.",
            'cat':"A cat with butterfly wings naps on a giant mushroom that glows softly in the moonlight.",
            'deer':"A deer with antlers made of crystal prances through a forest where the trees are made of liquid gold.",
            'dog':"A dog with a tail made of fireflies chases a glowing ball that changes color with every bounce.",
            'frog':"A frog with silver skin leaps from one floating lily pad to another in a pond of molten glass.",
            'horse':"A horse with a mane of starlight gallops across a meadow where the grass whispers secrets to the wind.",
            'ship':"A ship made of translucent ice sails smoothly through a sea of sparkling soda.",
            'truck':"A truck with tires made of clouds rolls down a highway that twists and turns through the skies."}

        for i in range(sample_size):
            name = sample_classes[i]
            prompts[name] = []
            # prompts[name].append('a photo of a '+name)
            prompts[name].append('A realistic, lifelike, and clear photo of a '+name)
            
            # prompts[name].append(d[name])

        return prompts, sample_classes, selected_indices, all_classes
    
    elif dataset == 'cifar100':
        all_classes = ['apple','aquarium_fish','baby','bear','beaver','bed','bee',
                        'beetle','bicycle','bottle','bowl','boy','bridge','bus',
                        'butterfly','camel','can','castle','caterpillar','cattle','chair',
                        'chimpanzee','clock','cloud','cockroach','couch','crab','crocodile',
                        'cup','dinosaur','dolphin','elephant','flatfish','forest','fox',
                        'girl','hamster','house','kangaroo','keyboard','lamp','lawn_mower',
                        'leopard','lion','lizard','lobster','man','maple_tree','motorcycle',
                        'mountain','mouse','mushroom','oak_tree','orange','orchid','otter',
                        'palm_tree','pear','pickup_truck','pine_tree','plain','plate','poppy',
                        'porcupine','possum','rabbit','raccoon','ray','road','rocket',
                        'rose','sea','seal','shark','shrew','skunk','skyscraper',
                        'snail','snake','spider','squirrel','streetcar','sunflower','sweet_pepper',
                        'table','tank','telephone','television','tiger','tractor','train',
                        'trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm']
        indices = np.random.permutation(len(all_classes))
        selected_indices = indices[:sample_size]

        sample_classes = [all_classes[index] for index in selected_indices]

        for i in range(sample_size):
            name = sample_classes[i]
            prompts[name] = []
            prompts[name].append('a photo of a '+name)

        return prompts, sample_classes, selected_indices, all_classes

    
def cal_Cosinesimilarity(tensor):  # 输入为(bs, feature_dim)
    norms = torch.norm(tensor, dim=1, keepdim=True)

    tensor_transposed = tensor.t()

    dot_products = torch.matmul(tensor, tensor_transposed)

    cosine_similarity = dot_products / (norms * norms.t())

    cosine_similarity = torch.clamp(cosine_similarity, 0, 1)

    k = tensor.size(0)
    n = k*(k-1)/2
    s = (torch.sum(cosine_similarity)-k)/2
    mean_cos = s/n

    return cosine_similarity, mean_cos

def get_aug(dataset):

    
    if dataset == 'imagenet':
        num_classes = 1000
        img_size = 256
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        normalize = aug.Normalize(mean=mean, std=std)
        transform = AugmentationSequential(
            aug.Resize(size = (256, 256)),
            aug.CenterCrop(size = (224, 224)),
            # aug.RandomResizedCrop(size = (224, 224), scale=(0.4, 1.0)),
            # aug.RandomHorizontalFlip(),
            normalize
        )

    elif dataset == 'imagenet100':
        num_classes = 100
        img_size = 256
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        normalize = aug.Normalize(mean=mean, std=std)
        transform = AugmentationSequential(
            aug.Resize(size = (256, 256)),
            aug.CenterCrop(size = (224, 224)),
            # aug.RandomResizedCrop(size = (224, 224), scale=(0.4, 1.0)),
            # aug.RandomHorizontalFlip(),
            normalize
        )

    elif dataset == 'cifar10':
        num_classes = 10
        img_size = 32
        # mean = (0.4914, 0.4822, 0.4465)
        # std = (0.2023, 0.1994, 0.2010)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        normalize = aug.Normalize(mean=mean, std=std)
        transform = AugmentationSequential(
            aug.Resize(size = (32, 32)),
            # aug.CenterCrop(size = (224, 224)),
            # aug.RandomResizedCrop(size = (32, 32), scale=(0.4, 1.0)),
            # aug.RandomHorizontalFlip(),
            normalize
        )

    elif dataset == 'cifar100':
        num_classes = 100
        img_size = 32
        # mean = (0.5071, 0.4867, 0.4408)
        # std = (0.2675, 0.2565, 0.2761)
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        # mean = (0.485, 0.456, 0.406)
        # std = (0.229, 0.224, 0.225)
        normalize = aug.Normalize(mean=mean, std=std)
        transform = AugmentationSequential(
            aug.Resize(size = (32, 32)),
            # aug.CenterCrop(size = (224, 224)),
            # aug.RandomResizedCrop(size = (224, 224), scale=(0.4, 1.0)),
            # aug.RandomHorizontalFlip(),
            normalize
        )
    
    return num_classes, img_size, transform

def get_clip_transform(n_px):
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    normalize = aug.Normalize(mean=mean, std=std)

    transform = AugmentationSequential(
            aug.Resize(size = (n_px, n_px)),
            aug.CenterCrop(size = (n_px, n_px)),
            # aug.RandomHorizontalFlip(),
            normalize
        )
    
    return transform

def transform4imgs(image, transform, params=None):
    img_tensor = None
    for i in range(len(image)):
        if params is None:
            img_ = transform(image[i])
        else:
            img_ = transform(image[i], params=params)
        # print('img_:', img_.size())
        if img_tensor is None:
            img_tensor = img_
        else:
            img_tensor = torch.cat((img_tensor, img_), dim=0)
    # print('img_tensor:', img_tensor.size())

    return img_tensor


def save_img_tensor(img, name):
    torchvision.utils.save_image(img, name)

def gram_schmidt(vectors):

    ortho_vectors = []
    for v in vectors:
        for u in ortho_vectors:
            proj = torch.dot(v, u) / torch.dot(u, u) * u
            v = v - proj
        ortho_vectors.append(v)
    return ortho_vectors

def normalize_vectors(vectors):

    normalized_vectors = []
    for v in vectors:
        mean = v.mean()
        std = v.std()
        normalized_v = (v - mean) / std  # 标准化为均值0，标准差1
        normalized_vectors.append(normalized_v)
    return normalized_vectors

def generate_orthogonal_tensors(shape, num_tensors=512):

    # 初始化张量列表
    tensors = []

    # 生成满足高斯分布的随机张量
    for _ in range(num_tensors):
        tensor = torch.randn(shape)
        tensors.append(tensor)

    flat_tensors = [tensor.view(-1) for tensor in tensors]

    # flat_ortho_tensors = gram_schmidt(flat_tensors)
    # flat_ortho_norm_tensors = normalize_vectors(flat_ortho_tensors)

    for i in range(1, num_tensors):
        for j in range(i):
            proj = torch.dot(flat_tensors[j], flat_tensors[i]) / torch.dot(flat_tensors[j], flat_tensors[j])
            flat_tensors[i] -= proj * flat_tensors[j]
        flat_tensors[i] /= torch.norm(flat_tensors[i])

    # for i in range(num_tensors):
    #     flat_ortho_tensors[i] = (flat_ortho_tensors[i]-torch.mean(flat_ortho_tensors[i]))/torch.std(flat_ortho_tensors[i])  # 标准化

    # print("\nOrthogonality check (should be close to zero):")
    # for i in range(len(flat_tensors)):
    #     for j in range(i + 1, len(flat_tensors)):
    #         dot_product = torch.dot(flat_tensors[i], flat_tensors[j])
    #         print(f"Dot product of vec{i+1} and vec{j+1}: {dot_product:.4f}")
    
    # print("\nCheck Gaussian properties:")
    # for i, vec in enumerate(flat_tensors):
    #     print(f"vec{i+1} mean: {vec.mean():.4f}, std: {vec.std():.4f}")

    tensors = None

    for tensor in flat_tensors:
        # tensors = [tensor.view(shape) for tensor in flat_tensors]
        tensor = torch.unsqueeze(tensor.view(shape), dim=0)
        if tensors is None:
            tensors = tensor
        else:
            tensors = torch.cat((tensors, tensor), dim=0)
    
    tensors = tensors[torch.randperm(len(tensors))].to('cuda')
    
    return tensors

def get_prompt_(c):
    d = {'moped':"The sleek moped, with its shiny chrome finish and efficient engine, zipped smoothly through the narrow city streets, effortlessly weaving through the morning traffic."}

    return d[c]

def grubbs_test(data, value, alpha=0.05, mode='high'):
    
    n = len(data)
    mean_y = np.mean(data)
    std_y = np.std(data, ddof=1)
    
    # 计算G统计量
    if mode == 'high':
        G = (value - mean_y) / std_y
    elif mode == 'low':
        G = (mean_y - value) / std_y
    
    # 计算临界值
    critical_value = ((n - 1) / np.sqrt(n)) * np.sqrt((t.ppf(1 - alpha / (2 * n), n - 2)**2) / (n - 2 + t.ppf(1 - alpha / (2 * n), n - 2)**2))
    
    return G > critical_value

def is_in_confidence_interval(data, value, confidence_level=0.95):
    mean_y = np.mean(data)
    std_y = np.std(data, ddof=1)
    n = len(data)
    
    h = std_y * t.ppf((1 + confidence_level) / 2, n - 1) / np.sqrt(n)
    lower_bound = mean_y - h
    upper_bound = mean_y + h
    print(lower_bound)
    print(upper_bound)
    # print(value)

    return lower_bound <= value

def create_one_hot_matrix(num_classes, repeat_interval):
    one_hot_matrix = np.eye(num_classes)
    
    repeated_block = np.repeat(one_hot_matrix, repeat_interval, axis=0)
    
    return repeated_block

def cal_cos_nd(matrix1, matrix2):
    cos = []
    # print(matrix1.shape)
    for i in range(matrix1.shape[0]):
        # print(matrix1[i].shape)
        # matrix1[i] = np.reshape(matrix1[i], (1,-1))
        # print(matrix1[i].shape)
        cos.append(cosine_similarity(np.expand_dims(matrix1[i], axis=0), np.expand_dims(matrix2[i], axis=0)))
    
    return np.squeeze(np.array(cos), axis=-1)

def get_random_prompt(c):
    # places = ['bedroom','living room','kitchen','bathroom','dining room','office','classroom','gym','library','hospital','beach','park','street','highway','mountain','forest','desert','farm','market','airport','supermarket','restaurant','hotel','mall','cinema','stadium','church','museum','zoo','amusement_park']
    places = ['office','classroom','gym','library','hospital','beach','park','street','mountain','desert']

    domains = ['cartoon', 'oil_painting', 'line_drawing', 'black_and_white', 'pixel_art']

    p = random.choice(places)
    d = random.choice(domains)

    # prompt = f'a {d} of a {c} in {p}'
    # prompt = f'a {c} in {p}'
    prompt = f'a photo of {c} in {p}'

    return prompt

def get_cos_subset(cos, N=20):
    ID = np.random.choice(1000, N, replace=False)
    cos_subset = cos[ID]

    return cos_subset

def one_tailed_ttest(res_sim_adv, res_sim_shadow):
    t_stat, p_val = stats.ttest_ind(res_sim_adv, res_sim_shadow)
    # t_stat, p_val = stats.ttest_rel(res_sim_adv, res_sim_shadow)

    if t_stat > 0:
        p_val_one_sided = p_val / 2
    else:
        p_val_one_sided = 1 - p_val / 2
    
    return t_stat, p_val_one_sided

def generalized_esd(data, alpha=0.05, max_outliers=None):
    data = np.array(data)
    n = len(data)
    if max_outliers is None:
        max_outliers = int(n * 0.1)  # 设置为样本量的 10%
    
    outliers = []
    
    for i in range(1, max_outliers + 1):
        mean_data = np.mean(data)
        std_data = np.std(data, ddof=1)
        
        G = np.abs(data - mean_data) / std_data
        max_G_idx = np.argmax(G)
        max_G = G[max_G_idx]
        
        t = stats.t.ppf(1 - alpha / (2 * (n - i + 1)), n - i - 1)
        G_crit = ((n - i) * t) / (np.sqrt((n - i - 1 + t**2) * (n - i + 1)))
        
        if max_G > G_crit:
            outliers.append(data[max_G_idx])
            data = np.delete(data, max_G_idx)
        else:
            break
    
    return outliers, data

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets) 
        pt = torch.exp(-ce_loss) 
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss 
        return focal_loss
    
def kmeans(X, k, max_iters=300, init_id=None):
    if init_id is None:
        centers = X[torch.randperm(X.size(0))[:k]]
    else:
        centers = X[init_id]
    
    for i in range(max_iters):
        distances = torch.cdist(X, centers)
        
        labels = distances.argmin(dim=1)
        
        new_centers = torch.stack([X[labels == j].mean(dim=0) for j in range(k)])
        
        if torch.all(centers == new_centers):
            break
        
        centers = new_centers
    
    return centers, labels

def cal_d(logits, num_classes, init=True, iter=50, input_init_ids = None):

    center_accs = []
    mean_coses = []
    init_ids = []
    mean_r_list = []

    for i in range(iter):
        if input_init_ids is None:
            init_id = []
            for j in range(num_classes):
                init_id.append(random.randint(j*5000/num_classes, (j+1)*5000/num_classes-1))
                # init_id.append(j*500+i)
            init_ids.append(init_id)
        else:
            init_id = input_init_ids[i]

        if init:
            # print(init_id)
            centers, labels_ = kmeans(logits, k=num_classes, init_id=init_id)
        else:
            centers, labels_ = kmeans(logits, k=num_classes)
        # center_labels = torch.argmax(centers, dim=-1)
        # right_num = 0
        # for m in range(num_classes):
        #     if center_labels[m] == m:
        #         right_num += 1
        # center_acc = right_num / num_classes

        # center_accs.append(center_acc)

        # Cos = []
        # for m in range(num_classes):
        #     # print(logits[m*500:(m+1)*500].size())
        #     # print(torch.unsqueeze(centers[m], dim=0).repeat(2,500).size())
        #     cos = torch.cosine_similarity(logits[m*500:(m+1)*500], torch.unsqueeze(centers[m], dim=0))
        #     cos = torch.mean(cos).item()
        #     Cos.append(cos)
        # mean_cos = np.mean(Cos)
        # mean_coses.append(mean_cos)

        # center_accs.append(center_acc*mean_cos)

        r_list = []
        for m in range(num_classes):
            # c = torch.argmax(centers, dim=-1)[i]
            # d_ = torch.cdist(logits[c*500:(c+1)*500], torch.unsqueeze(centers[i], dim=0))
            d_ = torch.cdist(logits[int(m*5000/num_classes):(m+1)*int(5000/num_classes)], torch.unsqueeze(centers[m], dim=0))
            d_ = d_.detach().cpu().numpy()
            r_list.append(np.mean(d_))
            
        mean_r_list.append(np.mean(r_list))

    # d = None
    # for i in range(num_classes):
    #     # c = torch.argmax(centers, dim=-1)[i]
    #     # d_ = torch.cdist(logits[c*500:(c+1)*500], torch.unsqueeze(centers[i], dim=0))
    #     d_ = torch.cdist(logits[i*500:(i+1)*500], torch.unsqueeze(centers[i], dim=0))
    #     d_ = d_.detach().cpu().numpy()
        
    #     if d is None:
    #         d = d_
    #     else:
    #         d = np.concatenate((d, d_), axis=0)

    # return d
    # return center_accs, init_ids
    return mean_r_list, init_ids

def hook(module, fea_in, fea_out):
    features_in_hook = []
    features_out_hook = []
    features_in_hook.append(fea_in.data)
    features_out_hook.append(fea_out.data)
    return None

def map_layer(m, n):
    MAX = max(m, n)
    MIN = min(m, n)
    r = MAX/MIN
    l_max, l_min = [], []

    for i in range(MAX):
        l_max.append(i)
        l_min.append(int(i/r))

    return l_max, l_min