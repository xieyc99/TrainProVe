#########################################################################################################################
#   Code for Diversity is Definitely Needed: Improving Model-Agnostic Zero-shot Classification via Stable Diffusion     #
#   Jordan Shipard, Arnold Wiliem, Kien Nguyen Thanh, Wei Xiang, Clinton Fookes                                         #
#########################################################################################################################

import os
import torch
# import wandb
from data import CIFAR10Generated, CIFAR10DataModule, CIFAR100Generated, CIFAR100DataModule, EuroSATDatamodule, ImageNet100DataModule
from models import ViT
from models import SwinTransformer
# from models import mobilenet_v3_small
import torchmetrics
from tqdm import tqdm
from torchvision.models import resnet18, resnet50, resnet101, convnext_base, convnext_small
import argparse
from utils import *
from torchmetrics import ConfusionMatrix
from data.CIFAR10_generated.cifar10_gen_dataset import CIFAR10GeneratedDataset
from torch.utils.data import ConcatDataset, DataLoader

parser = argparse.ArgumentParser()

parser.add_argument("--model", dest="model", type=str, default="RS18", choices=['RS18', 'Vit-T', 'RS50', 'Vit-S', 'RS101', 'Vit-B', 'convnext'])
parser.add_argument("--batch_size", dest="batch_size", type=int, default=64, choices=[32, 64, 256, 512])
parser.add_argument("--lr", type=float, default=1e-4, choices=[1e-4, 5e-5])
parser.add_argument("--wd", type=float, default=0.9, choices=[0.9, 0.09])
parser.add_argument("--loss", default='CE', type=str, help='the loss for train', choices=['CE', 'Focal'])


parser.add_argument("--opt", default='AdamW', type=str, help='the loss for train', choices=['AdamW', 'SGD', 'RMS'])
parser.add_argument("--epoch", dest="epoch", type=int, default=100)
parser.add_argument('--save_freq', default=100, type=int)
parser.add_argument("--combine", default=False)
parser.add_argument("--shadow", default=True)

parser.add_argument("--syn_data_location", dest="syn_data_location", type=str, default='D:\Exp\datasets\imagenet100-gen')
parser.add_argument("--dataset", dest="dataset", type=str, default='imagenet100_sd1_4_generated_merged_256A')
parser.add_argument("--img_size", dest="img_size", type=int, default=256)

parser.add_argument("--model_path", type=str, default=None)
parser.add_argument("--wandb", dest="wandb", action="store_true", default=False)
parser.add_argument("--real_data_location", dest="real_data_location", type=str, default='D:\Exp\datasets')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
parser.add_argument('--seed', default=999, type=int, help='seed for initializing training')
parser.add_argument("--dataset_size", default=1, type=float, help='the size of dataset for train', choices=[1,2,3,4])

if __name__ == '__main__':
    args = parser.parse_args()
    set_seed(args.seed)
    torch.cuda.set_device(args.gpu)

    EPOCHS = args.epoch
    LR = args.lr
    BATCH_SIZE = args.batch_size

    if args.model_path is None:
        folder_name = args.dataset
        # folder_name = 'cifar10_sd1_4'
        if args.combine:
            # folder_name += '_combine'
            folder_name = 'imagenet100_sd1_4_combine'
        elif args.shadow:
            folder_name += '_sdw_true'

        save_path = os.path.join('saved_models', args.model, f'x{args.dataset_size}', folder_name, f'e{args.epoch}_bs{args.batch_size}_lr{args.lr}_wd{args.wd}_loss{args.loss}_opt{args.opt}')
        save_name = f"{args.model}_syn_trained.pt"
    else: 
        save_path = os.path.split(args.model_path)[-1]
        save_name = os.path.join(*os.path.split(args.model_path)[:-1])

    print('save_path:', save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Setup the generated and real datasets
    if "cifar10_" in args.dataset:
        if args.poison_dataset:
            gen_datamodule = CIFAR10Generated(batch_size=BATCH_SIZE, root_dir=os.path.join(args.syn_data_location, args.dataset, 'train'), dataset_size=args.dataset_size, poison_root=os.path.join(args.syn_data_location, args.poison_dataset, 'train'), poison_rate=args.poison_rate)
        else:
            gen_datamodule = CIFAR10Generated(batch_size=BATCH_SIZE, root_dir=os.path.join(args.syn_data_location, args.dataset, 'train'), dataset_size=args.dataset_size)
        # gen_datamodule = CIFAR10DataModule(batch_size=BATCH_SIZE, root_dir=args.real_data_location)
        real_datamodule = CIFAR10DataModule(batch_size=BATCH_SIZE, root_dir=args.real_data_location)
    elif "cifar100_" in args.dataset:
        gen_datamodule = CIFAR100Generated(batch_size=BATCH_SIZE, root_dir=os.path.join(args.syn_data_location, args.dataset, 'train'), dataset_size=args.dataset_size)
        # gen_datamodule = CIFAR100DataModule(batch_size=BATCH_SIZE, root_dir=args.real_data_location)
        real_datamodule = CIFAR100DataModule(batch_size=BATCH_SIZE, root_dir=args.real_data_location)
    elif "eurosat_" in args.dataset:
        gen_datamodule = EuroSATDatamodule(batch_size=BATCH_SIZE, root_dir=os.path.join(args.syn_data_location, args.dataset))
        real_datamodule = EuroSATDatamodule(batch_size=BATCH_SIZE, root_dir=args.real_data_location)
    elif 'imagenet100_' in args.dataset:
        gen_datamodule = ImageNet100DataModule(batch_size=BATCH_SIZE, root_dir=os.path.join(args.syn_data_location, args.dataset))
        # gen_datamodule = ImageNet100DataModule(batch_size=BATCH_SIZE, root_dir=args.real_data_location+'/imagenet100')
        real_datamodule = ImageNet100DataModule(batch_size=BATCH_SIZE, root_dir=args.real_data_location+'/imagenet100')

    if args.combine:
        gen_datamodule.setup(stage = "fit", img_size=args.img_size)
        gen_train_dataset = gen_datamodule.train_dataset()

        real_datamodule.setup(stage = "fit", img_size=args.img_size)
        real_train_dataset = real_datamodule.train_dataset()

        train_dataset = ConcatDataset([gen_train_dataset, real_train_dataset])
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True) 
    else:
        gen_datamodule.setup(stage = "fit", img_size=args.img_size)
        train_dataloader = gen_datamodule.train_dataloader()

    real_datamodule.setup("test", img_size=args.img_size)
    real_test_dataloader = real_datamodule.test_dataloader()

    ### MODELS ###
    if args.model == "Vit-B":
        model = ViT(image_size=args.img_size, patch_size=16, num_classes=gen_datamodule.n_classes, dim=768 , depth=12, heads=12, mlp_dim=768*4)
    elif args.model == "Vit-S":
        model = ViT(image_size=args.img_size, patch_size=16, num_classes=gen_datamodule.n_classes, dim=448 , depth=12, heads=7, mlp_dim=448*3)
    elif args.model == "Vit-T":
        model = ViT(image_size=args.img_size, patch_size=16, num_classes=gen_datamodule.n_classes, dim=192 , depth=12, heads=4, mlp_dim=192*3)
    elif args.model == "RS18":
        model = resnet18(num_classes=gen_datamodule.n_classes)
    elif args.model == "RS50":
        model = resnet50(num_classes=gen_datamodule.n_classes)
    elif args.model == "RS101":
        model = resnet101(num_classes=gen_datamodule.n_classes)
    elif args.model == "convnext":
        model = convnext_base(num_classes=gen_datamodule.n_classes)
    elif args.model == "convnext-s":
        model = convnext_small(num_classes=gen_datamodule.n_classes)
    elif args.model == "swin":
        model = SwinTransformer(img_size=224 if args.img_size == 256 else 32, num_classes=gen_datamodule.n_classes, window_size=7 if args.img_size == 256 else 1)

    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path))

    model.cuda()

    if args.opt == 'AdamW':
        optim = torch.optim.AdamW(params=model.parameters(), lr=LR, weight_decay=args.wd)
    elif args.opt == 'SGD':
        optim = torch.optim.SGD(params=model.parameters(), lr=LR, momentum=0.9, weight_decay=args.wd)
    elif args.opt == 'RMS':
        optim = torch.optim.RMSprop(params=model.parameters(), lr=LR, alpha=0.99, eps=1e-08, weight_decay=args.wd, momentum=0, centered=False)

    if args.loss == 'CE':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss == 'Focal':
        criterion = FocalLoss(gamma=2, weight=None)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=EPOCHS)
        
    train_metric = torchmetrics.Accuracy(task='multiclass', num_classes=gen_datamodule.n_classes).cuda()
    test_metric = torchmetrics.Accuracy(task='multiclass', num_classes=gen_datamodule.n_classes).cuda()

    name = "" # wandb name
    # logger = wandb.init(job_type="results", project="DDN", name=name,
    #                                                         config={
    #                                                         "total_epochs": EPOCHS,
    #                                                         "optimiser":type(optim).__name__,
    #                                                         "lr": LR,
    #                                                         "batch_size":BATCH_SIZE,
    #                                                         "model": args.model,
    #                                                         "syn_pretrain_amount": args.syn_amount,
    #                                                         "dataset": args.dataset,
    #                                                         "img_size": args.img_size,
    #                                                         "weight_decay": args.wd}) if args.wandb else None

    best_acc = 0 

    for epoch in range(EPOCHS):
        train_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        train_bar.set_description(f"Epoch: {epoch}")

        for idx, batch in train_bar:
            if type(batch) == dict:
                images, labels = batch.values()
                images = images.float()
            else:
                images, labels = batch
            
            images, labels = images.cuda(), labels.cuda()

            output = model(images)

            loss = criterion(output, labels)
            train_metric.update(output, labels)
            # logger.log({"train_loss":loss,
            #             "epoch":epoch}) if args.wandb else None

            optim.zero_grad()
            loss.backward()
            optim.step()

        scheduler.step()

        # if args.wandb:
        #     logger.log({"train_acc": train_metric.compute(),
        #                 "epoch": epoch,
        #                 "lr": optim.param_groups[0]['lr']})
        # else:
        #     print(f"Train acc: {train_metric.compute()}")
        print(f"Train acc: {train_metric.compute()}")

        train_metric.reset()

        # torch.save(model.state_dict(), os.path.join(save_path, save_name))

        if epoch % 20 == 0 or epoch == EPOCHS-1:
            with torch.no_grad():
                num_list = np.zeros((gen_datamodule.n_classes))
                right_list = np.zeros((gen_datamodule.n_classes))

                for idx, batch in enumerate(real_test_dataloader):
                    if type(batch) == dict:
                        images, labels = batch.values()
                        images = images.float()
                    else:
                        images, labels = batch
                    images, labels = images.cuda(), labels.cuda()

                    output = model(images)
                    # print('output:', output.size(), 'labels:', labels.size())
                    for i in range(output.size()[0]):
                        label_id = labels[i].item()
                        num_list[label_id] += 1

                        # if i == 0:
                        #     print('output:', output[i].size())
                        max_id = torch.argmax(output[i], dim=-1)
                        if max_id == label_id:
                            right_list[max_id] += 1

                    loss = criterion(output, labels)
                    acc = test_metric.update(output, labels)

                acc = test_metric.compute()

                print(f"Test acc: {acc}")
                print(f"Each Class Test acc: {right_list/num_list}")


                test_metric.reset()
                
        if (epoch+1) % args.save_freq == 0:
            
            torch.save(model.state_dict(), os.path.join(save_path, f'{args.model}_syn_trained_{epoch+1}.pt'))