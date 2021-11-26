import logging
import os
import shutil
import time

from torchvision.transforms import Normalize
from torchvision.transforms.functional import to_pil_image
import torch
from torch.optim.lr_scheduler import MultiStepLR

from modules.models.base import BaseModel
from modules.components import build_components
from utils.aux_tool import ImagePool

from tensorboardX import SummaryWriter
from utils.plot import plot_samples_per_epoch, plot_image, plot_val_samples, stitch
from utils.misc import AverageMeter
from tqdm import tqdm
from utils.metrics import calculate_batch_psnr, calculate_batch_ssim
from modules.initializer.weights_initializer import init_model_weights
from pathlib import Path


class CycleGAN(BaseModel):
    # TODO: completing all of the modules for this method
    def __init__(self, args, device):
        super(CycleGAN, self).__init__(args, device)

        components = build_components(self.args)
        self.net_G_A, self.net_G_B, self.net_D_A, self.net_D_B, self.criterion_l1,\
            self.criterion_cycle, self.criterion_idt, self.criterion_gan = components
        gen_params = list(self.net_G_A.parameters()) + list(self.net_G_B.parameters())
        disc_params = list(self.net_D_A.parameters()) + list(self.net_D_B.parameters())
        self.generator_optimizer = torch.optim.Adam(gen_params,
                                                    lr=self.args.learning_rate,
                                                    betas=(self.args.beta1, self.args.beta2))
        self.discriminator_optimizer = torch.optim.Adam(disc_params,
                                                        lr=self.args.learning_rate,
                                                        betas=(self.args.beta1, self.args.beta2))
        self.gen_scheduler = MultiStepLR(self.generator_optimizer, range(100, 200, 50), gamma=0.1)
        self.dis_scheduler = MultiStepLR(self.discriminator_optimizer, range(100, 200, 50), gamma=0.1)

        self.current_iteration = 0
        self.current_epoch = 0

        self.logger = logging.getLogger("CycleGAN")

        self.move_components_to_device(args.mode)

        self.fake_A_pool = ImagePool()
        self.fake_B_pool = ImagePool()

        self.unnormalizer = Normalize(-1, 2)
        self.lambda_idt = args.lambda_idt
        self.lambda_cycle = args.lambda_cycle
        self.lambda_gan = args.lambda_gan
        self.lambda_l1 = args.lambda_l1

    def load_checkpoint(self, file_path):
        """
        Load checkpoint
        """
        checkpoint = torch.load(file_path)

        self.current_epoch = checkpoint['epoch']
        self.current_iteration = checkpoint['iteration']
        self.net_G_A.load_state_dict(checkpoint['net_G_A_state_dict'])
        self.net_G_B.load_state_dict(checkpoint['net_G_B_state_dict'])
        self.net_D_A.load_state_dict(checkpoint['net_D_A_state_dict'])
        self.net_D_B.load_state_dict(checkpoint['net_D_B_state_dict'])
        self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer'])
        self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])
        self.logger.info('Chekpoint loaded successfully from {} at epoch: {} and iteration: {}'.format(
            file_path, checkpoint['epoch'], checkpoint['iteration']))
        return self.current_epoch

    def save_checkpoint(self, file_name, is_best=0):
        """
        Save checkpoint
        """
        state = {
            'epoch': self.current_epoch + 1,  # because epoch is used for loading then this must be added + 1
            'iteration': self.current_iteration,
            'net_G_A_state_dict': self.net_G_A.state_dict(),
            'net_G_B_state_dict': self.net_G_B.state_dict(),
            'net_D_A_state_dict': self.net_D_A.state_dict(),
            'net_D_B_state_dict': self.net_D_B.state_dict(),
            'generator_optimizer': self.generator_optimizer.state_dict(),
            'discriminator_optimizer': self.discriminator_optimizer.state_dict()
        }

        torch.save(state, os.path.join(self.args.checkpoint_dir, file_name))

        if is_best:
            shutil.copyfile(os.path.join(self.args.checkpoint_dir, file_name),
                            os.path.join(self.args.checkpoint_dir, 'model_best.pth'))

    def adjust_learning_rate(self, epoch):
        """
        Adjust learning rate every epoch
        """
        # TODO modify later after all of the architecture has been done
        self.gen_scheduler.step()
        self.dis_scheduler.step()

    def train_one_epoch(self, train_loader, epoch):
        """
        Training step for each mini-batch
        """
        self.current_epoch = epoch
        self._reset_metric()

        tqdm_batch = tqdm(train_loader, desc='epoch-{}'.format(epoch))

        self.net_G_A.train()
        self.net_G_B.train()
        self.net_D_A.train()
        self.net_D_B.train()

        end_time = time.time()
        for curr_it, data in enumerate(tqdm_batch):
            start_time = time.time()
            data_time = start_time - end_time
            img_A, img_B = data
            img_A = img_A.to(self.device)
            img_B = img_B.to(self.device)

            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()

            fake_B = self.net_G_A(img_A)
            fake_A = self.net_G_B(img_B)
            rec_A = self.net_G_B(fake_B)
            rec_B = self.net_G_A(fake_A)

            loss_l1 = self.lambda_l1 * (self.criterion_l1(fake_A, img_A) + self.criterion_l1(fake_B, img_B))
            loss_idt = self.lambda_idt * (self.criterion_idt(self.net_G_A(img_B), img_B) + self.criterion_idt(self.net_G_B(img_A), img_A)) / 2
            loss_cycle = self.lambda_cycle * (self.criterion_cycle(rec_A, img_A) + self.criterion_cycle(rec_B, img_B))
            loss_gan = self.lambda_gan * (self.criterion_gan(self.net_D_B(fake_B), True) + self.criterion_gan(self.net_D_A(fake_A), True))

            loss_G = loss_idt + loss_cycle + loss_gan + loss_l1
            loss_G.backward()

            loss_D_A_real = self.criterion_gan(self.net_D_A(img_A), True)
            loss_D_A_fake = self.criterion_gan(self.net_D_A(fake_A.detach()), False)
            loss_D_A = loss_D_A_real + loss_D_A_fake
            loss_D_A.backward()

            loss_D_B_real = self.criterion_gan(self.net_D_B(img_B), True)
            loss_D_B_fake = self.criterion_gan(self.net_D_B(fake_B.detach()), False)
            loss_D_B = loss_D_B_real + loss_D_B_fake
            loss_D_B.backward()
            self.generator_optimizer.step()
            self.discriminator_optimizer.step()

            self.loss_G_meter.update(loss_G.item())
            self.loss_D_A_meter.update(loss_D_A.item())
            self.loss_D_B_meter.update(loss_D_B.item())

            self.current_iteration += 1
            end_time = time.time()
            self.batch_time_meter.update(time.time() - end_time)

            self.summary_writer.add_scalar("epoch/loss_G", self.loss_G_meter.val, self.current_iteration)
            self.summary_writer.add_scalar("epoch/loss_D_A", self.loss_D_A_meter.val, self.current_iteration)
            self.summary_writer.add_scalar("epoch/loss_D_B", self.loss_D_B_meter.val, self.current_iteration)

            tqdm_batch.set_description(
                '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | LossG: {lossG:.4f} | LossD_A: {lossD_A:.4f} | LossD_B: {lossD_B:.4f}'.format(
                    batch=curr_it + 1,
                    size=len(train_loader),
                    data=data_time,
                    bt=end_time - start_time,
                    lossG=self.loss_G_meter.avg,
                    lossD_A=self.loss_D_A_meter.avg,
                    lossD_B=self.loss_D_B_meter.avg,
                ))


        tqdm_batch.close()
        self.logger.info(
            'Training at epoch-{} | LR: {} Loss_G: {} Loss_D_A: {} Loss_D_B: {}'.format(str(self.current_epoch), str(self.args.learning_rate),
                                                            str(self.loss_G_meter.val), str(self.loss_D_A_meter.val), str(self.loss_D_B_meter.val)))

    @torch.no_grad()
    def validate(self, val_loader):
        """
        Validation step for each mini-batch
        """
        self.net_G_A.eval()
        self.net_G_B.eval()
        tqdm_batch = tqdm(val_loader, desc='Validation at epoch-{}'.format(self.current_epoch))

        for curr_it, data in enumerate(tqdm_batch):
            img_A, _ = data
            img_A = img_A.to(self.device)

            if self.args.stitch:
                fake_B = stitch(self.net_G_A, img_A, n_patches=self.args.stitch_n_patches, size_constraints=4)
            else:
                fake_B = self.net_G_A(img_A)
            # DEBUG
            img_A = self.unnormalizer(img_A)
            fake_B = self.unnormalizer(fake_B)

            img_A = torch.clamp(img_A, 0., 1.)
            fake_B = torch.clamp(fake_B, 0., 1.)
            plot_val_samples(fake_B, output_dir=os.path.join(self.args.output_dir, "fake_B"), fname="batch_{}.png".format(curr_it))
            plot_val_samples(img_A, output_dir=os.path.join(self.args.output_dir, "real_A"), fname="batch_{}.png".format(curr_it))

            tqdm_batch.set_description(
                '({batch}/{size})'.format(
                    batch=curr_it + 1,
                    size=len(val_loader),
                ))

        # save last image
        fake_b_saved_img = plot_samples_per_epoch(gen_batch=fake_B.data,
                                                 output_dir=os.path.join(self.args.output_dir, "fake_B"),
                                                 epoch=self.current_epoch)
        fake_b_saved_img = fake_b_saved_img.transpose((2, 0, 1))

        self.summary_writer.add_image('validation/real_A', img_A.squeeze(0), self.current_epoch)
        self.summary_writer.add_image('validation/fake_B', fake_b_saved_img, self.current_epoch)

        tqdm_batch.close()

        return self.current_epoch

    def init_training_logger(self):
        """
        Initialize training logger specific for each model
        """
        self.summary_writer = SummaryWriter(log_dir=self.args.summary_dir, comment='CycleGAN')
        Path(os.path.join(self.args.output_dir, 'fake_A')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.args.output_dir, 'fake_B')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.args.output_dir, 'rec_A')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.args.output_dir, 'rec_B')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.args.output_dir, 'real_A')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.args.output_dir, 'real_B')).mkdir(parents=True, exist_ok=True)
        self._reset_metric()

    def init_validation_logger(self):
        """
        Initialize validation logger specific for each model
        """
        self.summary_writer = SummaryWriter(log_dir=self.args.summary_dir, comment='CycleGAN')
        Path(os.path.join(self.args.output_dir, 'fake_B')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.args.output_dir, 'real_A')).mkdir(parents=True, exist_ok=True)

    def init_testing_logger(self):
        """
        Initialize testing logger specific for each model
        """
        self.summary_writer = SummaryWriter(log_dir=self.args.summary_dir, comment='CycleGAN')
        Path(os.path.join(self.args.output_dir, 'fake_B')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.args.output_dir, 'real_A')).mkdir(parents=True, exist_ok=True)

    def finalize_training(self):
        """
        Finalize training
        """
        self.logger.info("Finalizing everything")
        self.save_checkpoint("final_checkpoint.pth")
        self.summary_writer.export_scalars_to_json(os.path.join(self.args.summary_dir, "all_scalars.json"))
        self.summary_writer.close()

    def move_components_to_device(self, mode):
        """
        Move components to device
        """
        self.net_G_A = self.net_G_A.to(self.device)
        self.net_G_B = self.net_G_B.to(self.device)
        self.net_D_A = self.net_D_A.to(self.device)
        self.net_D_B = self.net_D_B.to(self.device)
        self.logger.info('Model G_A: {}'.format(self.net_G_A))
        self.logger.info('Model G_B: {}'.format(self.net_G_B))
        self.logger.info('Model D_A: {}'.format(self.net_D_A))
        self.logger.info('Model D_B: {}'.format(self.net_D_B))

    def _reset_metric(self):
        """
        Metric related to average meter
        """
        self.batch_time_meter = AverageMeter()
        self.data_time_meter = AverageMeter()
        self.loss_G_meter = AverageMeter()
        self.loss_D_A_meter = AverageMeter()
        self.loss_D_B_meter = AverageMeter()

    def count_parameters(self):
        """
        Return the number of parameters for the model
        """
        net_G_A_params_number = sum(p.numel() for p in self.net_G_A.parameters() if p.requires_grad)
        net_G_B_params_number = sum(p.numel() for p in self.net_G_B.parameters() if p.requires_grad)
        net_D_A_params_number = sum(p.numel() for p in self.net_D_A.parameters() if p.requires_grad)
        net_D_B_params_number = sum(p.numel() for p in self.net_D_B.parameters() if p.requires_grad)
        print(net_D_B_params_number)
        return net_G_A_params_number + net_G_B_params_number + net_D_A_params_number + net_D_B_params_number
