import os
import math
from decimal import Decimal

import utility

import torch
from torch.autograd import Variable
from tqdm import tqdm


# Trainer(args, loader, model, loss, checkpoint)
class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)):
                self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        # self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        
        timer_data, timer_model = utility.timer(), utility.timer()
        scaler = torch.cuda.amp.GradScaler()
        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):
            lr, hr = self.prepare([lr, hr])
            timer_data.hold()  # hold：self.acc += self.toc()
            timer_model.tic()  # tic：是当前时间
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                sr = self.model(lr, idx_scale)
                loss = self.loss(sr, hr)
            
            # args.skip_threshold：skipping batch that has large error
            # self.error_last = 1e8
            if loss.item() < self.args.skip_threshold * self.error_last:
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))
            self.optimizer.zero_grad()
            self.scheduler.step()
            timer_model.hold()
            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()
        '''
        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):
            lr, hr = self.prepare([lr, hr])
            timer_data.hold()  # hold：self.acc += self.toc()
            timer_model.tic()  # tic：是当前时间

            self.optimizer.zero_grad()
            sr = self.model(lr, idx_scale)
            loss = self.loss(sr, hr)
        '''
        '''
        官方代码
        [Epoch 1]       Learning rate: 1.00e-4
        [1600/16000]    [L1: 21.6380]   93.3+1.1s
        [3200/16000]    [L1: 16.8315]   85.3+0.5s
        [4800/16000]    [L1: 14.7276]   85.7+0.5s
        '''
        '''
        修改后
        RuntimeError: Pin memory thread exited unexpectedly
        修改代码：kwargs['pin_memory'] = False，报错消失
        运行结果如下：
        [Epoch 1]       Learning rate: 1.00e-4
        [1600/16000]    [L1: 22.7962]   105.3+1.5s
        [3200/16000]    [L1: 17.4339]   91.3+0.4s
        '''
        '''
        warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
        '''
        
        
        self.loss.end_log(len(self.loader_train))# loss：self.log[-1].div_(n_batches)
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (lr, hr, filename, _) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare([lr, hr])
                    else:
                        lr = self.prepare([lr])[0]

                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)#***

                    save_list = [sr]
                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale)

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
