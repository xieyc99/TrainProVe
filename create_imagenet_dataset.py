import subprocess
import argparse
import random
import os
from utils import *
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser()

# General dataset parameters
# parser.add_argument("--classes", dest="classes", nargs='+', required=True)
# parser.add_argumnet("--trick", dest="trick", type=str, required=True)
parser.add_argument("--dataset", dest="dataset", type=str, default='imagenet100', choices=['imagenet100'])
parser.add_argument("--outdir", dest="outdir", type=str, default='D:\Exp\datasets')
parser.add_argument("--model_type", default="sd1_4", type=str, choices=['sd1_4', 'lcm', 'pixart', 'cascade'])
parser.add_argument("--num_imgs", default=5000, type=int)
parser.add_argument("--num_images_per_prompt", default=1, type=int)
parser.add_argument("--dataset_to_gen", default=str, choices=['suspect', 'shadow', 'val'], help='Generate the dataset of suspicious model, shadow dataset or validation dataset')

# Stable Diffusion parameters
parser.add_argument("--seed", dest="seed", type=int, default=999, choices=[999, 2024])
parser.add_argument('--size', default=[256,256], type=int, help='height and width of general images')

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

print('model_type:', args.model_type)
if args.model_type == 'cascade':
    pipe_prior, model = get_model(args.model_type)
else:
    model = get_model(args.model_type)

prompts, sample_classes, names, selected_indices, all_classes, name2id, class_name = get_prompt('imagenet100', 100)

if args.model_type == 'glide':
    num_per = int(args.num_imgs/len(all_classes))
else:
    num_per = int(args.num_imgs/len(all_classes)/args.num_images_per_prompt)


for id, c in enumerate(all_classes):
    if id >= 0:
        if args.dataset_to_gen == 'suspect':
            save_path_ = os.path.join(save_path, 'train', name2id[c])
            os.makedirs(save_path_, exist_ok=True)
            prompt = prompts[c][0]

            print(f'*******{id}:{c}*******')

            print(f'-------{prompt}-------')
            count = 0
            if args.model_type == 'kds2_2':
                image_emb, negative_image_emb = pipe_prior(prompt).to_tuple()
            for i in tqdm(range(num_per), desc=f"{args.model_type} loop"):
                # print(c)
                r = random.randint(0, 2**32-1)
                if args.model_type == 'cascade':
                    # pipe_prior.enable_model_cpu_offload()
                    prior_output = pipe_prior(
                        prompt=prompt,
                        height=1024,
                        width=1024,
                        negative_prompt='',
                        guidance_scale=2.0,
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
                    imgs = model(prompt, num_images_per_prompt=args.num_images_per_prompt, guidance_scale=2).images

                for img in imgs:
                    img_path = os.path.join(save_path_, f'class_prompt_{c}_{str(count).zfill(5)}.png')
                    img = img.resize((args.size[0], args.size[1]), Image.Resampling.BILINEAR)
                    img.save(img_path)
                    count += 1
        
        elif args.dataset_to_gen == 'shadow':
            print(f'*******shadow*******')
            print(f'*******{id}:{c}*******')
            prompt = f'a {c}'

            print(f'-------{prompt}-------')
            count = 0
            if args.model_type == 'kds2_2':
                image_emb, negative_image_emb = pipe_prior(prompt).to_tuple()
            for i in tqdm(range(num_per), desc=f"{args.model_type} loop"):
                # print(c)
                r = random.randint(0, 2**32-1)
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
                    img_path = os.path.join(save_path_, f'class_prompt_{c}_{str(count).zfill(5)}.png')
                    img = img.resize((args.size[0], args.size[1]), Image.Resampling.BILINEAR)
                    img.save(img_path)
                    count += 1

        elif args.dataset_to_gen == 'val':
            print('-------gen_val-------')
            save_path_ = os.path.join(save_path, 'val', name2id[c])
            os.makedirs(save_path_, exist_ok=True)
            prompt = f'a photo of a {c}'

            print(f'*******{id}:{c}*******')

            print(f'-------{prompt}-------')
            count = 0

            for i in tqdm(range(num_per), desc=f"{args.model_type} loop"):
                # print(c)
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
                    # imgs = model(c, num_images_per_prompt=args.num_images_per_prompt, generator=torch.manual_seed(r)).images
                    imgs = model(prompt, num_images_per_prompt=args.num_images_per_prompt).images

                for img in imgs:
                    img_path = os.path.join(save_path_, f'class_prompt_{c}_{str(count).zfill(5)}.png')
                    img = img.resize((args.size[0], args.size[1]), Image.Resampling.BILINEAR)
                    img.save(img_path)
                    count += 1
    