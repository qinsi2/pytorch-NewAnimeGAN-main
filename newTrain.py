import torch
import torchvision
import argparse
import os
import cv2
import numpy as np
import torch.optim as optim
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from modeling.anime_gan import Generator
from modeling.anime_gan import Discriminator
from modeling.losses import AnimeGanLoss
from modeling.losses import LossSummary
from utils.common import load_checkpoint
from utils.common import save_checkpoint
from utils.common import set_lr
from utils.common import initialize_weights
from utils.common import seg_person
from utils.image_processing import denormalize_input
from dataset import AnimeDataSet
from tqdm import tqdm

gaussian_mean = torch.tensor(0.0)
gaussian_std = torch.tensor(0.1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Shinkai')
    parser.add_argument('--data-dir', type=str, default='/content/dataset')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--init-epochs', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--checkpoint-dir', type=str, default='/content/checkpoints')
    parser.add_argument('--save-image-dir', type=str, default='/content/images')
    parser.add_argument('--gan-loss', type=str, default='lsgan', help='lsgan / hinge / bce')
    parser.add_argument('--resume', type=str, default='G')
    parser.add_argument('--use_sn', action='store_true')
    parser.add_argument('--save-interval', type=int, default=1)
    parser.add_argument('--debug-samples', type=int, default=0)
    parser.add_argument('--lr-g', type=float, default=2e-4)
    parser.add_argument('--lr-d', type=float, default=4e-4)
    parser.add_argument('--init-lr', type=float, default=1e-3)
    parser.add_argument('--wadvg', type=float, default=10.0, help='Adversarial loss weight for G')
    parser.add_argument('--wadvd', type=float, default=10.0, help='Adversarial loss weight for D')
    parser.add_argument('--wcon', type=float, default=1.5, help='Content loss weight')
    parser.add_argument('--wgra', type=float, default=3.0, help='Gram loss weight')
    parser.add_argument('--wcol', type=float, default=30.0, help='Color loss weight')
    parser.add_argument('--d-layers', type=int, default=3, help='Discriminator conv layers')
    parser.add_argument('--d-noise', action='store_true')

    return parser.parse_args()


def collate_fn(batch):
    img, anime, anime_gray, anime_smt_gray = zip(*batch)
    return (
        torch.stack(img, 0),
        torch.stack(anime, 0),
        torch.stack(anime_gray, 0),
        torch.stack(anime_smt_gray, 0),
    )


def check_params(args):
    data_path = os.path.join(args.data_dir, args.dataset)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Dataset not found {data_path}')

    if not os.path.exists(args.save_image_dir):
        print(f'* {args.save_image_dir} does not exist, creating...')
        os.makedirs(args.save_image_dir)

    if not os.path.exists(args.checkpoint_dir):
        print(f'* {args.checkpoint_dir} does not exist, creating...')
        os.makedirs(args.checkpoint_dir)

    assert args.gan_loss in {'lsgan', 'hinge', 'bce'}, f'{args.gan_loss} is not supported'


def save_samples(generator, loader, args, max_imgs=2, subname='gen'):
    '''
    Generate and save images
    '''
    generator.eval()

    max_iter = (max_imgs // args.batch_size) + 1
    fake_imgs = []

    for i, (img, *_) in enumerate(loader):
        with torch.no_grad():
            fake_img = generator(img.cuda())
            fake_img = fake_img.detach().cpu().numpy()
            # Channel first -> channel last
            fake_img  = fake_img.transpose(0, 2, 3, 1)
            fake_imgs.append(denormalize_input(fake_img, dtype=np.int16))

        if i + 1 == max_iter:
            break

    fake_imgs = np.concatenate(fake_imgs, axis=0)

    for i, img in enumerate(fake_imgs):
        save_path = os.path.join(args.save_image_dir, f'{subname}_{i}.jpg')
        cv2.imwrite(save_path, img[..., ::-1])


def gaussian_noise():
    return torch.normal(gaussian_mean, gaussian_std)


def main(args):
    check_params(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Init models...")

    G = Generator(args.dataset).to(device)#.cuda()
    D = Discriminator(args).to(device)#.cuda()
    # ??????????????????mask-rcnn
    seg_model = torchvision.models.detection.maskrcnn_resnet50_fpn(True, True).to(device)
    seg_model.eval()

    loss_tracker = LossSummary()

    loss_fn = AnimeGanLoss(args)

    # Create DataLoader
    data_loader = DataLoader(
        AnimeDataSet(args),
        batch_size=args.batch_size,
        num_workers=cpu_count(),
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer_g = optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    start_e = 0
    if args.resume == 'GD':
        # Load G and D
        try:
            start_e = load_checkpoint(G, args.checkpoint_dir)
            print("G weight loaded")
            load_checkpoint(D, args.checkpoint_dir)
            print("D weight loaded")
        except Exception as e:
            print('Could not load checkpoint, train from scratch', e)
    elif args.resume == 'G':
        # Load G only
        try:
            start_e = load_checkpoint(G, args.checkpoint_dir, posfix='')
        except Exception as e:
            print('Could not load G init checkpoint, train from scratch', e)

    for e in range(start_e, args.epochs):
        print(f"Epoch {e}/{args.epochs}")
        bar = tqdm(data_loader)
        G.train()

        init_losses = []

        if e < args.init_epochs:
            # Train with content loss only
            set_lr(optimizer_g, args.init_lr)
            for img, *_ in bar:
                img = img.to(device)#.cuda()
                
                optimizer_g.zero_grad()

                fake_img = G(img)
                loss = loss_fn.content_loss_vgg(img, fake_img)
                loss.backward()
                optimizer_g.step()

                init_losses.append(loss.cpu().detach().numpy())
                avg_content_loss = sum(init_losses) / len(init_losses)
                bar.set_description(f'[Init Training G] content loss: {avg_content_loss:2f}')

            set_lr(optimizer_g, args.lr_g)
            save_checkpoint(G, optimizer_g, e, args, posfix='_init')
            save_samples(G, data_loader, args, subname='initg')
            continue

        loss_tracker.reset()
        for img, anime, anime_gray, anime_smt_gray in bar:
            # To cuda
            img = img.to(device)#.cuda()
            anime = anime.to(device)#.cuda()
            anime_gray = anime_gray.to(device)#.cuda()
            anime_smt_gray = anime_smt_gray.to(device)#.cuda()

            # ---------------- TRAIN D ---------------- #
            optimizer_d.zero_grad()
            fake_img = G(img).detach()

            # ????????????????????????
            # seg_fake_result = segmentation(seg_model, fake_img) 
            seg_fake_person, seg_fake_bg = seg_person(seg_model, fake_img)
            # ??????????????????????????????????????????         
            # seg_fake_d = D(seg_fake_result)
            seg_fake_person_d = torch.tensor([0])
            if len(seg_fake_person) > 0:
                seg_fake_person_d = D(seg_fake_person)
            seg_fake_bg_d = D(seg_fake_bg)


            # ????????????????????????
            # seg_anime_result = segmentation(seg_model, anime)
            seg_anime_person, seg_anime_bg = seg_person(seg_model, anime)
            # ??????????????????????????????????????????
            # seg_anime_d = D(seg_anime_result)
            seg_anime_person_d = torch.tensor([0])
            if len(seg_anime_person) > 0:
                seg_anime_person_d = D(seg_anime_person)
            seg_anime_bg_d = D(seg_anime_bg)


            # ?????????????????????????????????????????????
            # seg_ps = torch.cat([seg_anime_person, seg_fake_person])
            # ??????????????????????????????????????????
            # seg_bg = torch.cat([seg_anime_bg, seg_fake_bg])


            # Add some Gaussian noise to images before feeding to D
            if args.d_noise:
                fake_img += gaussian_noise()
                anime += gaussian_noise()
                anime_gray += gaussian_noise()
                anime_smt_gray += gaussian_noise()

            fake_d = D(fake_img)
            real_anime_d = D(anime)
            real_anime_gray_d = D(anime_gray)
            real_anime_smt_gray_d = D(anime_smt_gray)

            # ???????????????????????????????????????????????????????????????????????????????????????????????????
            loss_d = loss_fn.compute_loss_D(
                fake_d, real_anime_d, real_anime_gray_d, real_anime_smt_gray_d, seg_fake_person_d,seg_fake_bg_d, seg_anime_person_d, seg_anime_bg_d)

            loss_d.backward()
            optimizer_d.step()

            loss_tracker.update_loss_D(loss_d)

            # ---------------- TRAIN G ---------------- #
            optimizer_g.zero_grad()

            fake_img = G(img)
            fake_d = D(fake_img)

            adv_loss, con_loss, gra_loss, col_loss = loss_fn.compute_loss_G(
                fake_img, img, fake_d, anime_gray)

            loss_g = adv_loss + con_loss + gra_loss + col_loss

            loss_g.backward()
            optimizer_g.step()

            loss_tracker.update_loss_G(adv_loss, gra_loss, col_loss, con_loss)

            avg_adv, avg_gram, avg_color, avg_content = loss_tracker.avg_loss_G()
            avg_adv_d = loss_tracker.avg_loss_D()
            bar.set_description(f'loss G: adv {avg_adv:2f} con {avg_content:2f} gram {avg_gram:2f} color {avg_color:2f} / loss D: {avg_adv_d:2f}')

        if e % args.save_interval == 0:
            save_checkpoint(G, optimizer_g, e, args)
            save_checkpoint(D, optimizer_d, e, args)
            save_samples(G, data_loader, args)


if __name__ == '__main__':
    args = parse_args()

    print("# ==== Train Config ==== #")
    for arg in vars(args):
        print(arg, getattr(args, arg))
    print("==========================")

    main(args)
