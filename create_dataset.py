import subprocess
import argparse
import random
import os
from utils import get_model, set_seed, get_class_names
from PIL import Image
from tqdm import tqdm
from nltk.corpus import wordnet as wn

import torch

parser = argparse.ArgumentParser()

# General dataset parameters
# parser.add_argument("--classes", dest="classes", nargs='+', required=True)
# parser.add_argumnet("--trick", dest="trick", type=str, required=True)
parser.add_argument("--dataset", dest="dataset", type=str, default='cifar10', choices=['cifar10', 'cifar100'])
parser.add_argument("--dataset", dest="dataset", type=str, default='cifar10', choices=['cifar10', 'cifar100'])
parser.add_argument("--outdir", dest="outdir", type=str, default='D:\Exp\datasets')
parser.add_argument("--model_type", default="sd1_4", type=str, choices=['sd1_4', 'lcm', 'pixart', 'cascade'])
parser.add_argument("--num_imgs", default=4000, type=int)
parser.add_argument("--num_images_per_prompt", default=1, type=int)
parser.add_argument("--dataset_to_gen", default=str, choices=['suspect', 'shadow', 'val'], help='Generate the training set of suspicious model, shadow dataset or validation dataset')

# Multidomain parameters
parser.add_argument("--domains", dest="domains", default=['photo', 'drawing', 'painting', 'sketch', 'collage', 'poster', 'digital art image', 'rock drawing', 'stick figure', '3D rendering'])

# Random scale parameters
parser.add_argument("--min_scale", dest="min_scale", type=int, default=1)
parser.add_argument("--max_scale", dest="max_scale", type=int, default=5)

# Stable Diffusion parameters
parser.add_argument("--seed", dest="seed", type=int, default=99, choices=[999, 2024, 404, 99])
parser.add_argument('--size', default=[32,32], type=int, help='height and width of general images')

parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')

args = parser.parse_args()
set_seed(args.seed)
print(f'use gpu {args.gpu}')
torch.cuda.set_device(args.gpu)

# assert args.trick in ["class_prompt", "multidomain", "random_scale"]

root = os.path.join(args.outdir, f'{args.dataset}-gen')
folder_name = f'{args.dataset}_{args.model_type}_generated_merged_{args.size[0]}A'

if args.dataset_to_gen == 'shadow':
    folder_name = folder_name + '_sdw'
save_path = os.path.join(root, folder_name)
print('save_path:', save_path)

if args.model_type == 'cascade':
    pipe_prior, model = get_model(args.model_type)
else:
    model = get_model(args.model_type)

classes = get_class_names(args.dataset)
# print(classes)

num_per = int(args.num_imgs/len(classes)/args.num_images_per_prompt)


for id, c in enumerate(classes):
    if id >= 0:
        if args.dataset_to_gen == 'suspect':
            save_path_ = os.path.join(save_path, 'train', c)
            os.makedirs(save_path_, exist_ok=True)

            print(f'*******{id}:{c}*******')

            print('-------class_prompt-------')
            count = 0

            for i in tqdm(range(num_per-count), desc=f"{args.model_type} loop"):
                if args.model_type == 'cascade':
                    # pipe_prior.enable_model_cpu_offload()
                    prior_output = pipe_prior(
                        prompt=c,
                        height=1024,
                        width=1024,
                        negative_prompt='',
                        guidance_scale=4.0,
                        num_images_per_prompt=1,
                        num_inference_steps=20
                    )

                    # model.enable_model_cpu_offload()
                    imgs = model(
                        image_embeddings=prior_output.image_embeddings,
                        prompt=c,
                        negative_prompt='',
                        guidance_scale=0.0,
                        output_type="pil",
                        num_inference_steps=10
                    ).images
                else:
                    imgs = model(c, num_images_per_prompt=args.num_images_per_prompt).images

                for img in imgs:
                    img_path = os.path.join(save_path_, f'class_prompt_{c}_{str(count).zfill(5)}.png')
                    img = img.resize((args.size[0], args.size[1]), Image.Resampling.BILINEAR)
                    img.save(img_path)
                    count += 1

            print('-------multidomain-------')
            count = 0
            domains = args.domains
                    
            for domain in tqdm(domains, desc=f"{args.model_type} domains loop"):
                multi_domain_prompt = f"a {domain} of a {c}"

                for i in tqdm(range(int(num_per/len(domains))), desc="inner loop", leave=False):
                    if args.model_type == 'cascade':
                        # pipe_prior.enable_model_cpu_offload()
                        prior_output = pipe_prior(
                            prompt=multi_domain_prompt,
                            height=1024,
                            width=1024,
                            negative_prompt='',
                            guidance_scale=4.0,
                            num_images_per_prompt=1,
                            num_inference_steps=20
                        )

                        # model.enable_model_cpu_offload()
                        imgs = model(
                            image_embeddings=prior_output.image_embeddings,
                            prompt=multi_domain_prompt,
                            negative_prompt='',
                            guidance_scale=0.0,
                            output_type="pil",
                            num_inference_steps=10
                        ).images
                    else:
                        imgs = model(multi_domain_prompt, num_images_per_prompt=args.num_images_per_prompt).images

                    for img in imgs:
                        img_path = os.path.join(save_path_, f'multidomain_{c}_{str(count).zfill(5)}.png')
                        img = img.resize((args.size[0], args.size[1]), Image.Resampling.BILINEAR)
                        img.save(img_path)
                        count += 1

            print('-------random_scale-------')
            count = 0
            random_scale = random.randint(args.min_scale, args.max_scale)
            # random_scale = 1
            print('random_scale:', random_scale)

            if args.model_type == 'kds2_2':
                image_emb, negative_image_emb = pipe_prior(f'an image of a {c}').to_tuple()

            for i in tqdm(range(num_per-count), desc=f"{args.model_type} loop"):
                if args.model_type == 'cascade':
                    # pipe_prior.enable_model_cpu_offload()
                    prior_output = pipe_prior(
                        prompt=f'an image of a {c}',
                        height=1024,
                        width=1024,
                        negative_prompt='',
                        guidance_scale=random_scale,
                        num_images_per_prompt=1,
                        num_inference_steps=20
                    )

                    # model.enable_model_cpu_offload()
                    imgs = model(
                        image_embeddings=prior_output.image_embeddings,
                        prompt=f'an image of a {c}',
                        negative_prompt='',
                        guidance_scale=0.0,
                        output_type="pil",
                        num_inference_steps=10
                    ).images
                else:
                    imgs = model(f'an image of a {c}', num_images_per_prompt=args.num_images_per_prompt, guidance_scale=random_scale).images

                for img in imgs:
                    img_path = os.path.join(save_path_, f'random_scale_{c}_{str(count).zfill(5)}.png')
                    img = img.resize((args.size[0], args.size[1]), Image.Resampling.BILINEAR)
                    img.save(img_path)
                    count += 1

        elif args.dataset_to_gen == 'shadow':
            print('-------shadow-------')
            save_path_ = os.path.join(save_path, 'train', c)
            os.makedirs(save_path_, exist_ok=True)
            count = 0
            num_per = 300

            for i in tqdm(range(num_per), desc=f"{args.model_type} loop"):
                if args.model_type == 'cascade':
                    # pipe_prior.enable_model_cpu_offload()
                    prior_output = pipe_prior(
                        prompt=f'a {c}',
                        height=1024,
                        width=1024,
                        negative_prompt='',
                        guidance_scale=4.0,
                        num_images_per_prompt=1,
                        num_inference_steps=20
                    )

                    # model.enable_model_cpu_offload()
                    imgs = model(
                        image_embeddings=prior_output.image_embeddings,
                        prompt=f'a {c}',
                        negative_prompt='',
                        guidance_scale=0.0,
                        output_type="pil",
                        num_inference_steps=10
                    ).images
                else:
                    imgs = model(f'a {c}', num_images_per_prompt=args.num_images_per_prompt).images

                for img in imgs:
                    img_path = os.path.join(save_path_, f'{c}_{str(count).zfill(5)}.png')
                    img = img.resize((args.size[0], args.size[1]), Image.Resampling.BILINEAR)
                    img.save(img_path)
                    count += 1
        elif args.dataset_to_gen == 'val':
            print('-------gen_val-------')
            save_path_ = os.path.join(save_path, 'val', c)
            os.makedirs(save_path_, exist_ok=True)

            prompt = 'a photo of a '+c   # orig
            # prompt = 'a '+c   # inverse

            count = 0

            for i in tqdm(range(num_per), desc=f"{args.model_type} loop"):
                if args.model_type == 'cascade':
                    # pipe_prior.enable_model_cpu_offload()
                    prior_output = pipe_prior(
                        prompt=prompt,
                        height=1024,
                        width=1024,
                        negative_prompt='',
                        guidance_scale=4.0,
                        num_images_per_prompt=1,
                        num_inference_steps=20
                    )

                    # model.enable_model_cpu_offload()
                    imgs = model(
                        image_embeddings=prior_output.image_embeddings,
                        prompt=prompt,
                        negative_prompt='',
                        guidance_scale=0.0,
                        output_type="pil",
                        num_inference_steps=10
                    ).images
                else:
                    imgs = model(prompt, num_images_per_prompt=args.num_images_per_prompt).images

                for img in imgs:
                    img_path = os.path.join(save_path_, f'{c}_{str(count).zfill(5)}.png')
                    img = img.resize((args.size[0], args.size[1]), Image.Resampling.BILINEAR)
                    img.save(img_path)
                    count += 1