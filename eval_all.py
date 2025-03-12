from data.CIFAR10_generated.cifar10_gen_dataset import CIFAR10GeneratedDataset
from data.CIFAR100_generated.cifar100_gen_dataset import CIFAR100GeneratedDataset
from data.ImageNet100.imagenet100_datamodule import ImageNet
from utils import *
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet50, resnet101, convnext_base, convnext_small
from tqdm import tqdm
from data import CIFAR10Generated, CIFAR10DataModule, CIFAR100Generated, CIFAR100DataModule, EuroSATDatamodule, ImageNet100DataModule
from models.vit_pytorch import ViT
from scipy.ndimage import gaussian_filter1d
from models import SwinTransformer
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model_root", type=str, default="D:\Exp\AIGC\Diversity_is_Definitely_Needed-main\saved_models", help='The root directory for storing models')
parser.add_argument("--dataset", type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet100'])
parser.add_argument("--positive", type=bool, default=True, help='Whether to detect positive examples')
parser.add_argument("--criterion", type=str, default='acc', choices=['acc', 'entropy', 'binary_cos'], help='Use accuracy/entropy/binary cosine similarity as the detection metric (TrainProVe/TrainProVe-Ent/TrainProVe-Sim)')

parser.add_argument("--G_d", type=str, default='sd1_4', choices=['sd1_4', 'lcm', 'pixart', 'cascade'], help='The text-to-image model of defender')
parser.add_argument("--G_sus", type=str, default='sd1_4', choices=['sd1_4', 'lcm', 'pixart', 'cascade', 'real'], help='The data source of suspicious model')
parser.add_argument("--arch_shadow", type=str, default='RS18', choices=['RS18', 'swin', 'convnext'], help='The architecture of the shadow model')

parser.add_argument("--seed", type=int, default=999)


def cal_acc(arch, ckpt_path, criterion):

    if arch == 'RS50':
        model = resnet50(num_classes=num_classes).eval().cuda()
    elif arch == 'RS18':
        model = resnet18(num_classes=num_classes, return_feature=False).eval().cuda()
    elif arch == 'RS101':
        model = resnet101(num_classes=num_classes).eval().cuda()
    elif arch == 'Vit-B':
        model = ViT(image_size=img_size, patch_size=16, num_classes=num_classes, dim=768 , depth=12, heads=12, mlp_dim=768*4).cuda()
    elif arch == "Vit-S":
        model = ViT(image_size=img_size, patch_size=16, num_classes=num_classes, dim=448 , depth=12, heads=7, mlp_dim=448*3).cuda()
    elif arch == "swin":
        model = SwinTransformer(img_size=224 if img_size == 256 else 32, num_classes=num_classes, window_size=7 if img_size == 256 else 1).cuda()
    elif arch == "convnext":
        model = convnext_base(num_classes=num_classes).cuda()

    ckpt = torch.load(ckpt_path, map_location='cuda')
    msg = model.load_state_dict(ckpt, strict=True)  
    # print('msg:', msg)
   
    num_list = np.zeros((num_classes))
    right_list = np.zeros((num_classes))
    logits = None
    loss_list = []
    acc_list = []
    entropy_list = []

    with torch.no_grad():
        if criterion == 'binary_cos' or criterion == 'entropy':
            for idx, batch in enumerate(gen_val_loader_no_shuffle):
                if type(batch) == dict:
                    images, l = batch.values()
                    images = images.float()
                else:
                    images, l = batch
                images, l = images.cuda(), l.cuda()

                output = model(images)
                
                if logits is None:
                    logits = output
                else:
                    logits = torch.cat((logits, output), dim=0)

                if criterion == 'entropy':
                    prob = F.softmax(output, dim=1)
                    e = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)
                    # print(e.size())
                    e = torch.mean(e).item()
                    entropy_list.append(e)
        else:
            for idx, batch in enumerate(gen_val_loader_shuffle):
                right_num = 0

                if type(batch) == dict:
                    images, l = batch.values()
                    images = images.float()
                else:
                    images, l = batch
                images, l = images.cuda(), l.cuda()

                output = model(images)
                    
                loss = criterion(output, l)
                # loss.backward()

                loss_list.append(loss.item())

                for i in range(output.size()[0]):
                    label_id = l[i].item()
                    num_list[label_id] += 1

                    max_id = torch.argmax(output[i], dim=-1)
                    if max_id == label_id:
                        right_num += 1
                        right_list[max_id] += 1

                acc = right_num/output.size()[0]
                acc_list.append(acc)
                
    torch.cuda.empty_cache()

    return logits, np.array(acc_list), np.array(entropy_list)


if __name__ == '__main__':
    args = parser.parse_args()
    set_seed(args.seed)

    # generator_pool = ['sd1_4', 'lcm', 'pixart', 'cascade', 'real']
    generator_pool = [args.G_d, args.G_sus]
    arch_pool = ['RS50', 'RS101', 'Vit-S', 'Vit-B']
    bs_pool = [512, 256] if 'cifar' in args.dataset else [32, 64]
    lr_pool = [1e-4, 5e-5]
    wd_pool = [0.9, 0.09]
    loss_pool = ['CE', 'Focal']
    epoch = 100
    bs_shadow = 512 if 'cifar' in args.dataset else 64
    bs = 128 if 'cifar' in args.dataset else 16

    num_classes, img_size, transform = get_aug(args.dataset)

    dataset_size_shadow = 1
    ckpt_path_shadow = os.path.join(args.model_root, args.arch_shadow, f'x{dataset_size_shadow}', f'{args.dataset}_{args.G_d}_generated_merged_{img_size}A_sdw_true', f'e{epoch}_bs{bs_shadow}_lr0.0001_wd0.9_lossCE_optAdamW', f'{args.arch_shadow}_syn_trained_100.pt')

    print('*********shadow_model:', ckpt_path_shadow)
    print('dataset_size_shadow:', dataset_size_shadow)

    criterion = nn.CrossEntropyLoss()

    gen_data_root = f'D:\Exp\datasets\{args.dataset}-gen\{args.dataset}_{args.G_d}_generated_merged_{img_size}A'

    if args.dataset == 'cifar10':
        gen_val_set = CIFAR10GeneratedDataset(root=os.path.join(gen_data_root, 'val'), transforms=transform, dataset_size=5/30)
        print('gen_val_set:', len(gen_val_set))
        gen_val_loader_shuffle = DataLoader(gen_val_set, batch_size=bs, shuffle=True, pin_memory=True)
        gen_val_loader_no_shuffle = DataLoader(gen_val_set, batch_size=bs, shuffle=False, pin_memory=True)
    elif args.dataset == 'cifar100':
        gen_val_set = CIFAR100GeneratedDataset(root=os.path.join(gen_data_root, 'val'), transforms=transform, dataset_size=1/6)
        gen_val_loader_shuffle = DataLoader(gen_val_set, batch_size=bs, shuffle=True, pin_memory=True)
        gen_val_loader_no_shuffle = DataLoader(gen_val_set, batch_size=bs, shuffle=False, pin_memory=True)
    elif args.dataset == 'imagenet100':
        gen_val_set = ImageNet(root=gen_data_root, train=False)
        gen_val_loader_shuffle = DataLoader(gen_val_set, batch_size=bs, shuffle=True, pin_memory=True)
        gen_val_loader_no_shuffle = DataLoader(gen_val_set, batch_size=bs, shuffle=False, pin_memory=True)

        logits_shadow, accs_shadow, entropy_shadow = cal_acc(args.arch_shadow, ckpt_path_shadow, args.criterion)

    if args.criterion == 'binary_cos':
        cos_shadow_list = []
        num_per_class = logits_shadow.size(0) / num_classes
        for i in range(num_classes):
            _, cos_shadow = cal_Cosinesimilarity(logits_shadow[i*int(num_per_class):(i+1)*int(num_per_class)])
            cos_shadow_list.append(cos_shadow.item())
        binary_cos_shadow = np.mean(cos_shadow_list)

        print('binary_cos_shadow:', binary_cos_shadow)
    
    if args.positive is True:
        TP, FP = 0, 0
        ckpt_path_suspect_list = []

        with tqdm(desc="Positive Instance Processing", unit=" iteration", mininterval=0.1) as pbar:
            while True:
                if len(ckpt_path_suspect_list) == len(arch_pool)*len(bs_pool)*len(lr_pool)*len(wd_pool)*len(loss_pool):
                    break
                else:
                    data_source_suspect = args.G_d
                    arch_suspect = random.choice(arch_pool)
                    dataset_size_suspect = 1
                    bs_suspect = random.choice(bs_pool)
                    lr_suspect = random.choice(lr_pool)
                    wd_suspect = random.choice(wd_pool)
                    loss_suspect = random.choice(loss_pool)

                    ckpt_path_suspect = os.path.join(args.model_root, arch_suspect, f'x{dataset_size_suspect}', f'{args.dataset}_{data_source_suspect}', f'e{epoch}_bs{bs_suspect}_lr{lr_suspect}_wd{wd_suspect}_loss{loss_suspect}_optAdamW', f'{arch_suspect}_syn_trained_100.pt')

                    if ckpt_path_suspect in ckpt_path_suspect_list:
                        continue
                    else:
                        pbar.update(1)
                        ckpt_path_suspect_list.append(ckpt_path_suspect)

                        logits_suspect, accs_suspect, entropy_suspect = cal_acc(args.arch_shadow, ckpt_path_shadow, args.criterion)

                        print()
                        
                        if args.criterion == 'binary_cos':
                            cos_suspect_list = []
                            num_per_class = logits_suspect.size(0) / num_classes
                            # print('num_per_class:', num_per_class)
                            for i in range(num_classes):
                                _, cos_suspect = cal_Cosinesimilarity(logits_suspect[i*int(num_per_class):(i+1)*int(num_per_class)])
                                cos_suspect_list.append(cos_suspect.item())
                            binary_cos_suspect = np.mean(cos_suspect_list)
                            # losses_suspect = losses_suspect / binary_cos_suspect

                            print('binary_cos_suspect:', binary_cos_suspect)
                        
                        if args.criterion == 'acc':
                            print('acc_shadow:', np.max(accs_shadow), '|', np.min(accs_shadow), '|', np.mean(accs_shadow))
                            print('acc_suspect:', np.max(accs_suspect), '|', np.min(accs_suspect), '|', np.mean(accs_suspect))

                            if grubbs_test(accs_shadow, np.mean(accs_suspect), mode='low') == True:
                                FP += 1
                                print('FP:', ckpt_path_suspect)
                                print('*****FP*****')
                            else:
                                TP += 1
                                print('*****TP*****')

                        if args.criterion == 'entropy':
                            print('entropy_shadow:', np.max(entropy_shadow), '|', np.min(entropy_shadow), '|', np.mean(entropy_shadow))
                            print('entropy_suspect:', np.max(entropy_suspect), '|', np.min(entropy_suspect), '|', np.mean(entropy_suspect))

                            if grubbs_test(entropy_shadow, np.mean(entropy_suspect)) == True:
                                FP += 1
                                print('FP:', ckpt_path_suspect)
                                print('*****FP*****')
                            else:
                                TP += 1
                                print('*****TP*****')
                        
                        if args.criterion == 'binary_cos':
                            print('cos_shadow_list:', np.max(cos_shadow_list), '|', np.min(cos_shadow_list), '|', np.mean(cos_shadow_list))
                            print('cos_suspect_list:', np.max(cos_suspect_list), '|', np.min(cos_suspect_list), '|', np.mean(cos_suspect_list))

                            if grubbs_test(cos_shadow_list, np.mean(cos_suspect_list), mode='low') == True:
                                FP += 1
                                print('FP:', ckpt_path_suspect)
                                print('*****FP*****')
                            else:
                                TP += 1
                                print('*****TP*****')

                torch.cuda.empty_cache()
                            
        print('TP:', TP, 'FP:', FP)
    else:
        TN, FN = 0, 0
        generator_pool.remove(args.G_d)
        ckpt_path_suspect_list = []

        with tqdm(desc="Negative Instance Processing", unit=" iteration", mininterval=0.1) as pbar:
            while True:
                if len(ckpt_path_suspect_list) == len(arch_pool)*len(bs_pool)*len(lr_pool)*len(wd_pool)*len(loss_pool):
                    break
                else:
                    data_source_suspect = random.choice(generator_pool)
                    arch_suspect = random.choice(arch_pool)
                    dataset_size_suspect = 1
                    bs_suspect = random.choice(bs_pool)
                    lr_suspect = random.choice(lr_pool)
                    wd_suspect = random.choice(wd_pool)
                    loss_suspect = random.choice(loss_pool)

                    if data_source_suspect != 'real':
                        ckpt_path_suspect = os.path.join(args.model_root, arch_suspect, f'x{dataset_size_suspect}', f'{args.dataset}_{data_source_suspect}', f'e{epoch}_bs{bs_suspect}_lr{lr_suspect}_wd{wd_suspect}_loss{loss_suspect}_optAdamW', f'{arch_suspect}_syn_trained_100.pt')
                    else:
                        ckpt_path_suspect = os.path.join(args.model_root, arch_suspect, f'x{dataset_size_suspect}', f'{args.dataset}', f'e{epoch}_bs{bs_suspect}_lr{lr_suspect}_wd{wd_suspect}_loss{loss_suspect}_optAdamW', f'{arch_suspect}_syn_trained_100.pt')

                    if ckpt_path_suspect in ckpt_path_suspect_list:
                        # pbar.update(1)
                        continue
                    else:
                        pbar.update(1)
                        ckpt_path_suspect_list.append(ckpt_path_suspect)

                        logits_suspect, accs_suspect, entropy_suspect = cal_acc(args.arch_shadow, ckpt_path_shadow, args.criterion)

                        print()

                        if args.criterion == 'binary_cos':
                            cos_suspect_list = []
                            num_per_class = logits_suspect.size(0) / num_classes
                            # print('num_per_class:', num_per_class)
                            for i in range(num_classes):
                                _, cos_suspect = cal_Cosinesimilarity(logits_suspect[i*int(num_per_class):(i+1)*int(num_per_class)])
                                cos_suspect_list.append(cos_suspect.item())
                            binary_cos_suspect = np.mean(cos_suspect_list)
                            # losses_suspect = losses_suspect / binary_cos_suspect

                            print('binary_cos_suspect:', binary_cos_suspect)

                        if args.criterion == 'acc':
                            print('acc_shadow:', np.max(accs_shadow), '|', np.min(accs_shadow), '|', np.mean(accs_shadow))
                            print('acc_suspect:', np.max(accs_suspect), '|', np.min(accs_suspect), '|', np.mean(accs_suspect))

                            if grubbs_test(accs_shadow, np.mean(accs_suspect), mode='low') == True:
                                TN += 1
                                print('*****TN*****')
                            else:
                                FN += 1
                                print('FN:', ckpt_path_suspect)
                                print('*****FN*****')
                                  
                        if args.criterion == 'entropy':
                            print('entropy_shadow:', np.max(entropy_shadow), '|', np.min(entropy_shadow), '|', np.mean(entropy_shadow))
                            print('entropy_suspect:', np.max(entropy_suspect), '|', np.min(entropy_suspect), '|', np.mean(entropy_suspect))

                            if grubbs_test(entropy_shadow, np.mean(entropy_suspect)) == True:
                                TN += 1
                                print('*****TN*****')
                            else:
                                FN += 1
                                print('FN:', ckpt_path_suspect)
                                print('*****FN*****')
                        
                        if args.criterion == 'binary_cos':
                            print('cos_shadow_list:', np.max(cos_shadow_list), '|', np.min(cos_shadow_list), '|', np.mean(cos_shadow_list))
                            print('cos_suspect_list:', np.max(cos_suspect_list), '|', np.min(cos_suspect_list), '|', np.mean(cos_suspect_list))

                            if grubbs_test(cos_shadow_list, np.mean(cos_suspect_list), mode='low') == True:
                                TN += 1
                                print('*****TN*****')
                            else:
                                FN += 1
                                print('FN:', ckpt_path_suspect)
                                print('*****FN*****')

                torch.cuda.empty_cache()

        print('TN:', TN, 'FN:', FN)

