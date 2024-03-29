import os.path as osp
import torch.nn as nn
from copy import deepcopy
import seaborn as sns
import abc
import torch.nn.functional as F
import pdb
import torch.backends.cudnn as cudnn

from utils_s import *
from dataloader.data_utils_s import *
from tensorboardX import SummaryWriter
from .Network import *
from tqdm import tqdm
from SupConLoss import SupConLoss
#from src.datasets.common import get_dataloader, maybe_dictionarize
#from src.datasets.registry import get_dataset
from src.utils import *
from english_words import english_words_lower_set
from clip.loss import ClipLoss
from src.datasets.templates import get_templates


class FSCILTrainer(object, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()
        pass

    def init_vars(self, args):
        # Setting arguments
        # if args.start_session == 0:
        if args.load_ft_dir == None:
            args.secondphase = False
        else:
            args.secondphase = True

        if args.dataset == 'cifar100':
            import dataloader.cifar100.cifar_s as Dataset
            args.num_classes = 100
            args.width = 32
            args.height = 32

            if args.base_class == None and args.way == None:
                args.base_class = 60
                args.way = 5
                # args.shot = 5
                args.sessions = 9
            elif not args.base_class == None and not args.way == None:
                args.sessions = 1 + int((args.num_classes - args.base_class) / args.way)
            else:
                raise NotImplementedError

        if args.dataset == 'cub200':
            import dataloader.cub200.cub200_s as Dataset
            args.num_classes = 200
            args.width = 224
            args.height = 224

            if args.base_class == None and args.way == None:
                args.base_class = 100
                args.way = 10
                # args.shot = 5
                args.sessions = 11
            elif not args.base_class == None and not args.way == None:
                args.sessions = 1 + int((args.num_classes - args.base_class) / args.way)
            else:
                raise NotImplementedError

        if args.dataset == 'mini_imagenet':
            import dataloader.miniimagenet.miniimagenet_s as Dataset
            args.num_classes = 100
            args.width = 84
            args.height = 84

            if args.base_class == None and args.way == None:
                args.base_class = 60
                args.way = 5
                # args.shot = 5
                args.sessions = 9
            elif not args.base_class == None and not args.way == None:
                args.sessions = 1 + int((args.num_classes - args.base_class) / args.way)
            else:
                raise NotImplementedError

        args.Dataset = Dataset
        args.eps = 1e-7

        if args.base_dataloader_mode != 'episodic':
            args.shotpercls = False
        else:
            args.shotpercls = True


        # Setting save path
        # mode = args.base_mode + '-' + args.new_mode
        mode = args.fw_mode
        if not args.no_rbfc:
            mode = mode + '-' + 'rbfc'

        bfzb = 'T' if args.base_freeze_backbone else 'F'
        ifzb = 'T' if args.inc_freeze_backbone else 'F'

        bdg = 'T' if args.base_doubleaug else 'F'
        idg = 'T' if args.inc_doubleaug else 'F'

        rcnm = 'T' if args.rpclf_normmean else 'F'
        co = 'T' if args.use_custom_coslr else 'F'
        rdtxt = 'T%d'%(args.num_randomtext) if args.use_randomtext else 'F'
        if args.use_flyp_ft_v1:
            ft_type = 'flyp1'
        elif args.use_flyp_ft_v2:
            ft_type = 'flyp2'
        else:
            ft_type = 'ce'

        if args.schedule == 'Milestone':
            mile_stone = str(args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            schedule = 'MS_%s' % (mile_stone)
        elif args.schedule == 'Step':
            schedule = 'Step_%d' % (args.step)
        elif args.schedule == 'Cosine':
            schedule = 'schecos'
        elif args.schedule == 'Custom_Cosine':
            schedule = 'custcos'

        if args.schedule_new == 'Milestone':
            mile_stone_new = str(args.milestones_new).replace(" ", "").replace(',', '_')[1:-1]
            schedule_new = 'MS_%s' % (mile_stone_new)
        elif args.schedule_new == 'Step':
            raise NotImplementedError
            # schedule = 'Step_%d'%(args.step)
        elif args.schedule_new == 'Cosine':
            schedule_new = 'sch_c'
        elif args.schedule_new == 'Custom_Cosine':
            schedule_new = 'custcos'

        if args.batch_size_base > 256:
            args.warm = True
        else:
            args.warm = False
        if args.warm:
            # args.model_name = '{}_warm'.format(opt.model_name)
            args.warmup_from = 0.01
            args.warm_epochs = 10
            if args.schedule == 'Cosine':
                eta_min = args.lr_base * (args.gamma ** 3)
                args.warmup_to = eta_min + (args.lr_base - eta_min) * (
                        1 + math.cos(math.pi * args.warm_epochs / args.epochs_base)) / 2
            else:
                args.warmup_to = args.lr_base

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        # sch_c: shcedule cosine
        # bdm: base_dataloader_mode
        # l: lambda
        # sd : seed
        # SC: == supcon, use_supconloss T
        # BCF: base classifier fine-tune, == base_clf_ft T
        # Epf, Lrf: epoch, lr for fine-tune
        # RN == resnet, SCRN == Supconresnet
        # 2P == 2ndphase, st == start
        # hd == head_type, D == head_dim
        # SCA == supcon_angle
        # CK: cskd
        # CRN: CIFAR_RESNET
        # EM: encmlp
        # G: gauss, bt: tukey_beta, ns: num_sampled
        # CO: custom optimizer
        # MM: mapmlp
        # GG: GaussGenerate, b after GG is tukey_beta
        # MMB: use_mapmlp_in_base
        # ResMM: use_residual_mapmlp

        str_loss = '-'
        if args.use_celoss:
            str_loss += 'ce'
            if args.fw_mode == 'fc_cosface':
                str_loss += '_s%.1fm%.1f-' % (args.s, args.m)
        if args.use_supconloss:
            SCA_ = 'T' if args.supcon_angle else 'F'
            scstr_ = 'SC-SCA_%s-' % (SCA_)
            str_loss += scstr_
        # if args.use_cskdloss:
        #    str_loss += 'CK-l%.1f-'%(args.lamda)
        if args.use_cskdloss_1:
            str_loss += 'CK1-l%.1f-' % (args.lamda)
        if args.use_cskdloss_2:
            str_loss += 'CK2-l%.1f-' % (args.lamda)
        auglist1_ = ''.join(str(x) for x in args.aug_type1)
        auglist2_ = ''.join(str(x) for x in args.aug_type2)
        str_loss += 'aug_%s,%s-' % (auglist1_, auglist2_)

        if args.use_head:
            str_loss += 'hd%sD_%d-' % (args.head_type, args.head_dim)
        if args.use_encmlp:
            #str_loss += 'EM%d_D%d-' % (args.encmlp_layers, args.encmlp_dim)
            str_loss += 'EM%d-' % (args.encmlp_layers)
        if args.use_mapmlp:
            str_loss += 'MM%d-'%(args.mapmlp_layers)
        if args.use_gaussgen:
            str_loss += 'GG%d_%.1f-'%(args.num_gaussgen, args.tukey_beta)
        if args.use_mapmlp_in_base:
            str_loss += 'MMB-'
        if args.use_residual_mapmlp:
            str_loss += 'ResMM%.2f-'%(args.alpha_residual_mapmlp)

        cov_eyes = 'T' if args.use_cov_eyes else 'F'
        if args.use_cov_eyes:
            str_loss += 'covE_%s-'%(cov_eyes)


        hyper_name_list = 'Model_%s-Epob_%d-Epon_%d-Lrb_%.5f-Lrn_%.5f-%s-%s-Gam_%.2f-Dec_%.5f-wd_%.1f-ls_%.1f-Bs_%d-Mom_%.2f' \
                          'bsc_%d-way_%d-shot_%d-bfzb_%s-ifzb_%s-bdg_%s-idg_%s-rcnm_%s-bdm_%s-co_%s-rdtxt_%s' \
                          '-ft_type_%s-sd_%d' % (
                              args.model_type, args.epochs_base, args.epochs_new, args.lr_base, args.lr_new, schedule, schedule_new, \
                              args.gamma, args.decay, args.wd, args.ls,
                              args.batch_size_base, args.momentum, args.base_class, args.way, args.shot,
                              bfzb, ifzb, bdg, idg, rcnm, args.base_dataloader_mode, co, rdtxt, ft_type, args.seed)

        hyper_name_list += str_loss

        # if args.warm:
        #    hyper_name_list += '-warm'

        save_path = '%s/' % args.dataset
        save_path = save_path + '%s/' % args.project
        #save_path = save_path + '%s-st_%d/' % (mode, args.start_session)
        # save_path += 'mlp-ftenc'
        zsl_save_path = save_path + '/' + args.model_type
        zsl_save_path = os.path.join('checkpoint', zsl_save_path)
        args.zsl_save_path = zsl_save_path
        ensure_path(args.zsl_save_path)

        ft_save_path = save_path + hyper_name_list
        ft_save_path = os.path.join('checkpoint', ft_save_path)
        args.ft_save_path = ft_save_path
        ensure_path(ft_save_path)

        text_clf_weight_fn = 'head_%s_%s.pt' % (args.dataset, args.model_type)
        args.text_clf_weight_fn = os.path.join(args.zsl_save_path, text_clf_weight_fn)

        if args.use_randomtext:
            randomtext_embed_fn = 'rdtxt_%s_%s.pt' % (args.dataset, args.model_type)
            args.randomtext_embed_fn = os.path.join(args.zsl_save_path, randomtext_embed_fn)


        # Setting dictionaries
        # clsD initialize
        clsD = {}
        if args.use_gaussgen:
            gaussD = {}
            gaussD['mean'] = {}
            gaussD['cov'] = {}

        # clsD, procD
        if args.secondphase == False:
            args.task_class_order = np.random.permutation(np.arange(args.num_classes)).tolist()  #######
            # args.task_class_order = np.arange(args.num_classes).tolist()  #######
            clsD['tasks'], clsD['class_maps'] = init_maps(args)
            # self.args.task_class_order = (np.arange(self.args.num_classes)).tolist()

            procD = {}
            # varD['proc_book'] = {}

            # train statistics
            procD['trlog'] = {}
            procD['trlog']['train_loss'] = []
            procD['trlog']['val_loss'] = []
            procD['trlog']['test_loss'] = []
            procD['trlog']['train_acc'] = []
            procD['trlog']['val_acc'] = []
            procD['trlog']['test_acc'] = []
            procD['trlog']['max_acc_epoch'] = 0
            procD['trlog']['max_acc'] = [0.0] * args.sessions
            procD['trlog']['max_acc_base_rbfc'] = 0.0
            procD['trlog']['max_acc_base_after_ft'] = 0.0
            procD['trlog']['new_max_acc'] = [0.0] * args.sessions  # first will be left 0.0
            procD['trlog']['new_all_max_acc'] = [0.0] * args.sessions  # first will be left 0.0
            procD['trlog']['base_max_acc'] = [0.0] * args.sessions  # first will be left 0.0
            procD['trlog']['prev_max_acc'] = [0.0] * args.sessions  # first will be left 0.0
            procD['trlog']['prev_new_clf_ratio'] = [0.0] * args.sessions  # first will be left 0.0
            procD['trlog']['new_new_clf_ratio'] = [0.0] * args.sessions  # first will be left 0.0
            procD['session'] = -1

            bookD = book_val(args)

        else:
            assert args.load_ft_dir != None

            # load objs
            obj_dir = os.path.join(args.load_ft_dir, 'saved_dicts')
            with open(obj_dir, 'rb') as f:
                dict_ = pickle.load(f)
                procD = dict_['procD']
                bookD = dict_['bookD']
                if args.use_gaussgen:
                    gaussD = dict_['gaussD']

            procD['session'] = args.start_session - 1
            # epoch, step is init to -1 for every sessions so no worry.
            if args.start_session > 0:
                procD['trlog']['max_acc'][args.start_session:] = [0.0] * (args.sessions - args.start_session)
                procD['trlog']['new_max_acc'][args.start_session:] = [0.0] * (args.sessions - args.start_session)
                procD['trlog']['new_all_max_acc'][args.start_session:] = [0.0] * (args.sessions - args.start_session)
                procD['trlog']['base_max_acc'][args.start_session:] = [0.0] * (args.sessions - args.start_session)
                procD['trlog']['prev_max_acc'][args.start_session:] = [0.0] * (args.sessions - args.start_session)
                procD['trlog']['prev_new_clf_ratio'][args.start_session:] = [0.0] * (args.sessions - args.start_session)
                procD['trlog']['new_new_clf_ratio'][args.start_session:] = [0.0] * (args.sessions - args.start_session)
            else:
                procD['trlog']['new_max_acc'][args.start_session:] = [0.0] * (args.sessions - args.start_session)
                procD['trlog']['new_all_max_acc'][args.start_session:] = [0.0] * (args.sessions - args.start_session)
                procD['trlog']['base_max_acc'][args.start_session:] = [0.0] * (args.sessions - args.start_session)
                procD['trlog']['prev_max_acc'][args.start_session:] = [0.0] * (args.sessions - args.start_session)

            tasks = bookD[0]['tasks']  # Note that bookD[i]['tasks'] is same for each i
            class_order_ = []
            for i in range(len(tasks)):
                class_order_ += tasks[i].tolist()
            # clsD to up-to-date
            args.task_class_order = class_order_
            clsD['tasks'], clsD['class_maps'] = init_maps(args)
            # for i in range(args.start_session):

            if args.start_session == 0 or args.start_session == 1:
                pass
            else:
                clsD = inc_maps(args, clsD, procD, args.start_session - 1)
        if not args.use_gaussgen:
            return args, procD, clsD, bookD
        else:
            return args, procD, clsD, bookD, gaussD

    def get_optimizer_base(self, args, model, num_batches):

        if args.base_freeze_backbone:
            assert not args.use_flyp_ft_v1 and not args.use_flyp_ft_v2

            for param in model.module.clip_encoder.parameters():
                param.requires_grad = False
        # for param in self.model.module.fc.parameters():
        #    param.requires_grad = False

        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.module.parameters()),
        #                             lr=0.0003, weight_decay=0.0008)

        # optimizer = torch.optim.SGD(model.parameters(), #####
        """
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    # optimizer = torch.optim.SGD(model.module.parameters(),
                                    # args.lr_base, momentum=0.9, nesterov=True, weight_decay=args.decay)
                                    args.lr_base, momentum=0.9, weight_decay=args.decay)
        # optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr_base, momentum=0.9, nesterov=True,
        #                            weight_decay=self.args.decay)
        if args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)
        elif args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones,
                                                             gamma=args.gamma)
        elif args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_base)
        """
        params = [p for p in model.module.parameters() if p.requires_grad]
        if args.base_freeze_backbone:
            assert args.use_encmlp
            optimizer = torch.optim.AdamW(params, lr=args.lr_encmlp, weight_decay=args.wd)
        else:
            if not args.use_encmlp:
                optimizer = torch.optim.AdamW(params, lr=args.lr_base, weight_decay=args.wd)
            else:
                assert not args.use_flyp_ft_v1 and not args.use_flyp_ft_v2
                optimizer = torch.optim.AdamW([{'params': model.module.clip_encoder.parameters(),
                                                'lr': args.lr_base},
                                         {'params': model.module.encmlp.parameters(), 'lr': args.lr_encmlp}],
                                         weight_decay=args.wd)

        if args.schedule == 'Custom_Cosine':
            scheduler = cosine_lr(optimizer, args.lr_base, args.warmup_length, args.epochs_base * num_batches)
        elif args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)
        elif args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones,
                                                             gamma=args.gamma)
        elif args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_base)
        else:
            raise NotImplementedError

        return model, optimizer, scheduler



    def get_optimizer_new(self, args, model, num_batches):
        # assert self.args.angle_mode is not None

        if args.inc_freeze_backbone:
            assert not args.use_flyp_ft_v1 and not args.use_flyp_ft_v2

            for param in model.module.clip_encoder.parameters():
                # param.requires_grad = False
                param.requires_grad = False

        if args.use_head:
            for param in model.head.parameters():
                param.requires_grad = False

        params = [p for p in model.module.parameters() if p.requires_grad]
        if args.inc_freeze_backbone:
            if args.use_encmlp:
            #assert args.use_encmlp
                optimizer = torch.optim.AdamW(params, lr=args.lr_encmlp, weight_decay=args.wd)
            elif args.use_mapmlp:
                optimizer = torch.optim.AdamW(params, lr=args.lr_new, weight_decay=args.wd)
            else:
                raise NotImplementedError
        else:
            if not args.use_encmlp:
                optimizer = torch.optim.AdamW(params, lr=args.lr_new, weight_decay=args.wd)
            else:
                assert not args.use_flyp_ft_v1 and not args.use_flyp_ft_v2
                optimizer = torch.optim.AdamW([{'params': model.module.clip_encoder.parameters(),
                                                'lr': args.lr_new},
                                         {'params': model.module.encmlp.parameters(), 'lr': args.lr_encmlp}],
                                         weight_decay=args.wd)

        if args.schedule_new == 'Custom_Cosine':
            scheduler = cosine_lr(optimizer, args.lr_new, args.warmup_length, args.epochs_new * num_batches)
        elif args.schedule_new == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)
        elif args.schedule_new == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones,
                                                             gamma=args.gamma)
        elif args.schedule_new == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_new)

        return model, optimizer, scheduler



    def base_train(self, args, model, procD, clsD, trainloader, optimizer, kdloss, scheduler, epoch,
                   loss_fn, supcon_criterion=None):

        tl = Averager()
        ta = Averager()
        # self.model = self.model.train()
        model.train()
        num_batches = len(trainloader)

        # standard classification for pretrain
        tqdm_gen = tqdm(trainloader)
        for i, batch in enumerate(tqdm_gen, 1):

            step = i + epoch * num_batches
            scheduler(step)
            procD['step'] += 1

            if args.base_doubleaug is False:
                data, train_label = [_.cuda() for _ in batch]
                target_cls = clsD['class_maps'][0][train_label]
            else:
                # data = batch[0][0].cuda()
                # train_label = batch[1].cuda()
                data = torch.cat((batch[0][0], batch[0][1]), dim=0).cuda()
                # train_label = batch[1].repeat(2).cuda()
                train_label = batch[1].cuda()
                train_label = train_label.repeat(2)
                target_cls = clsD['class_maps'][0][train_label]

            inputs = data
            labels = target_cls
            """
            batch = maybe_dictionarize(batch)
            inputs = batch['images'].cuda()
            labels = batch['labels'].cuda()
            data_time = time.time() - start_time
            """
            if not args.use_flyp_ft_v1 and not args.use_flyp_ft_v2:
                if not args.use_mapmlp_in_base:
                    logits = model(inputs, sess=procD['session'], train=True)
                else:
                    feats_ = model(inputs, sess=procD['session'], encode=True, mapmlp=True, res_mapmlp=args.use_residual_mapmlp)
                    logits = model(feats_, sess=procD['session'], encode=False, mapmlp=False)
                loss = loss_fn(logits, labels)
            else:
                torch.autograd.set_detect_anomaly(True)
                batch_words = [args.dataset_label2txt[int(i)] for i in train_label] # -> prompt -> text encoder
                template = get_templates(args.dataset)
                if args.use_flyp_ft_v1:
                # Version follwing flyp git.
                # But without modifying the whole csv-dataloader part, can't use all templates
                # So moved to second version
                    tokentxts = [temptokenize(args, template, batch_words[i]) for i in range(len(batch_words))]
                    tokentxts = torch.stack(tokentxts, dim=0).detach()
                    #tokentxts = torch.stack(tokentxts, dim=0)
                    embed_imgs, embed_words, logit_scale2 =  model.module.clip_encoder.model(inputs, tokentxts)
                    #embed_words = lab_text_2weights_v2(model.module.clip_encoder.model, args, template, args.device, batch_words)
                    loss = loss_fn(embed_imgs, embed_words, logit_scale2)
                # Modified (united) version for flyp, using all templates.
                #embed_words = lab_text_2weights(model.module.clip_encoder.model, args, template, args.device, batch_words)
                elif args.use_flyp_ft_v2:
                    embed_words = lab_text_2weights_v2(model.module.clip_encoder.model, args, template, args.device, batch_words)
                    embed_imgs = model.module.clip_encoder.model.encode_image(inputs)
                    loss = loss_fn(embed_imgs, embed_words, model.module.clip_encoder.model.logit_scale.exp())
                # model.clip_encoder.logit_scale.exp()[0] is replacement of logit_scale2[0] from flyp git
                # since lgit_scale2 is output of clip forward and it outputs self.logit_scale.exp()
                # https://github.com/locuslab/FLYP/blob/main/clip/model.py


            params = [p for p in model.module.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(params, 1.0)

            if i % args.print_every == 0:
                percent_complete = 100 * i / len(trainloader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(trainloader)}]\t"
                    f"Loss: {loss.item():.6f}", flush=True
                )

            total_loss = loss

            if not args.use_custom_coslr:
                lrc = scheduler.get_last_lr()[0]
            tl.add(total_loss.item())

            optimizer.zero_grad()
            # loss.backward()
            total_loss.backward()
            optimizer.step()

            if not args.use_flyp_ft_v1 and not args.use_flyp_ft_v2:
                acc = count_acc(logits, labels)
                tqdm_gen.set_description(
                    'Session 0, epo {}, total loss={:.4f} acc={:.4f}'.format(epoch, total_loss.item(), acc))
                ta.add(acc)
            else:
                tqdm_gen.set_description(
                    'Session 0, epo {}, total loss={:.4f}'.format(epoch, total_loss.item()))
        tl = tl.item()
        if not args.use_flyp_ft_v1 and not args.use_flyp_ft_v2:
            ta = ta.item()
            return tl, ta
        else:
            return tl


    def new_train(self, args, model, procD, clsD, trainloader, optimizer, kdloss, scheduler, epoch,
                   loss_fn, supcon_criterion=None):

        tl = Averager()
        ta = Averager()
        # self.model = self.model.train()
        model.train()
        num_batches = len(trainloader)

        # standard classification for pretrain
        tqdm_gen = tqdm(trainloader)
        for i, batch in enumerate(tqdm_gen, 1):
            step = i + epoch * num_batches
            scheduler(step)
            if args.inc_doubleaug is False:
                data, train_label = [_.cuda() for _ in batch]
                target_cls = clsD['class_maps'][procD['session']][train_label]
            else:
                # data = batch[0][0].cuda()
                # train_label = batch[1].cuda()
                data = torch.cat((batch[0][0], batch[0][1]), dim=0).cuda()
                # train_label = batch[1].repeat(2).cuda()
                train_label = batch[1].cuda()
                train_label = train_label.repeat(2)
                target_cls = clsD['class_maps'][procD['session']][train_label]

            inputs = data
            labels = target_cls

            if not args.use_flyp_ft_inc:
                logits = model(inputs, sess=procD['session'], train=True)
                loss = loss_fn(logits, labels)
            else:
                torch.autograd.set_detect_anomaly(True)
                batch_words = [args.dataset_label2txt[int(i)] for i in train_label] # -> prompt -> text encoder
                template = get_templates(args.dataset)

                tokentxts = [temptokenize(args, template, batch_words[i]) for i in range(len(batch_words))]
                tokentxts = torch.stack(tokentxts, dim=0).detach()
                #tokentxts = torch.stack(tokentxts, dim=0)
                embed_imgs, embed_words, logit_scale2 =  model.module.clip_encoder.model(inputs, tokentxts)
                #embed_words = lab_text_2weights_v2(model.module.clip_encoder.model, args, template, args.device, batch_words)
                loss = loss_fn(embed_imgs, embed_words, logit_scale2)


            params = [p for p in model.module.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(params, 1.0)

            total_loss = loss

            if not args.use_custom_coslr:
                lrc = scheduler.get_last_lr()[0]
            tl.add(total_loss.item())

            optimizer.zero_grad()
            # loss.backward()
            total_loss.backward()
            optimizer.step()

            if not args.use_flyp_ft_inc:
                acc = count_acc(logits, labels)
                tqdm_gen.set_description(
                    'Session {}, epo {}, total loss={:.4f} acc={:.4f}'.format(procD['session'], epoch, total_loss.item(), acc))
                ta.add(acc)
            else:
                tqdm_gen.set_description(
                    'Session {}, epo {}, total loss={:.4f}'.format(procD['session'], epoch, total_loss.item()))
        tl = tl.item()
        if not args.use_flyp_ft_inc:
            ta = ta.item()
            return tl, ta
        else:
            return tl

    def new_train_featgen(self, args, model, procD, clsD, trainloader, optimizer, kdloss, scheduler, epoch,
                   loss_fn, supcon_criterion=None, gaussD=None):
        assert not args.use_encmlp
        assert args.inc_freeze_backbone
        assert args.use_mapmlp

        tl = Averager()
        ta = Averager()
        # self.model = self.model.train()
        model.train()
        num_batches = len(trainloader)

        #inputs, labels = tot_datalist_train(args, trainloader, model, args.inc_doubleaug, map=clsD['class_maps'][procD['session']])
        inputs, labels = tot_datalist_train(args, trainloader, model, args.inc_doubleaug,
                                            map=clsD['seen_unsort_map'])

        inputs = inputs.detach().cuda()
        labels == labels.detach().cuda()

        gen_inputs = []
        gen_labels = []

        n_cls_prev = args.base_class + args.way * (procD['session']-1)
        prev_feats = model.module.textual_classifier.weight[:n_cls_prev].detach()
        # for matching num_gpu remainder 0
        remain_ = ((args.way*args.shot + n_cls_prev*args.num_gaussgen) % args.batch_size_base) % args.num_gpu
        feat_dim = prev_feats.shape[1]
        for i in range(n_cls_prev):
            if i<remain_:
                gen_num = args.num_gaussgen-1
            else:
                gen_num = args.num_gaussgen
            if not args.use_gaussgen:
                tmp_gen_inputs = prev_feats[i].expand(gen_num, feat_dim) + torch.rand(gen_num, feat_dim)
            else:
                if args.use_cov_eyes:
                    # mvn = torch.distributions.MultivariateNormal(gaussD['mean'][i], gaussD['cov'][i])
                    n_feats = model.module.num_features
                    mvn = torch.distributions.MultivariateNormal(gaussD['mean'][i], 0.01*torch.eye(n_feats, n_feats))
                    sampled = mvn.sample((gen_num,))
                else:
                    sample = np.random.multivariate_normal(gaussD['mean'][i], gaussD['cov'][i], gen_num)
                    sampled = torch.tensor(sample).float()
                # random_sample = torch.randn().cuda()
                tmp_gen_inputs = torch.pow(sampled, 1 / args.tukey_beta).cuda()
            gen_inputs.append(tmp_gen_inputs)
            gen_labels.append(i * torch.ones(gen_num))
        gen_inputs = torch.cat(gen_inputs, dim=0).detach().cuda()
        gen_labels = torch.cat(gen_labels, dim=0).detach().cuda()


        inputs = torch.cat((inputs, gen_inputs),dim=0)
        labels = torch.cat((labels, gen_labels),dim=0)
        inputs = inputs[torch.randperm(len(inputs))]
        randperm_ = torch.randperm(len(labels)).cuda()
        #inputs = inputs[randperm_].cuda()
        labels = labels[randperm_].type(torch.LongTensor).cuda()

        bsz_ = args.batch_size_base
        for i in range(0, len(inputs), bsz_):
            batch_ = inputs[i:i+bsz_]
            label_ = labels[i:i+bsz_]
            #label_ = label_.type(torch.LongTensor).cuda()
            #feats = model.module.forw_mapmlp(batch_)
            model.eval() # Since BatchNorm1d does not allow single lasted elements,
            # Using error occurs when splitting into multiple gpus.
            # req_grad and updpate params aren't effected and only batchnorm layer-like things are affected,
            # which I desire. so just do this and forward mapmlp and do model.train() right away
            feats = model(batch_, encode=False, mapmlp=True, res_mapmlp=args.use_residual_mapmlp)
            feats = F.normalize(feats, dim=1)
            # Need to normalize since no batchnorm due to eval()
            # (Batchnorm is differ to norm but just did)
            model.train()
            logits_ = model(feats, sess=procD['session'], encode=False)
            loss = loss_fn(feats, label_)

            total_loss = loss

            tl.add(total_loss.item())

            optimizer.zero_grad()
            # loss.backward()
            total_loss.backward()
            optimizer.step()

            """
            if not args.use_flyp_ft_inc:
                acc = count_acc(feats, labels)
                print('Session %d, epo%d, total loss=%.4f acc=%.4f'%(procD['sesion'],epoch, total_loss.item(), acc))
                ta.add(acc)
            else:
                print('Session %d, epo%d, total loss=%.4f' % (procD['sesion'],epoch, total_loss.item()))
            """
        tl = tl.item()
        """
        if not args.use_flyp_ft_inc:
            ta = ta.item()
            return tl, ta
        else:
            return tl
        """
        return tl



    def test(self, args, model, procD, clsD, testloader):
        epoch = procD['epoch']
        session = procD['session']

        model = model.cuda()
        model.eval()
        vl = Averager()
        va = Averager()

        with torch.no_grad():
            tqdm_gen = tqdm(testloader)
            for i, batch in enumerate(tqdm_gen, 1):
                data, test_label = [_.cuda() for _ in batch]
                #logits = get_logits(data, model)
                if not args.use_gaussgen:
                    logits = model(data, procD['session'])
                else:
                    feats = model(data, procD['session'], encode=True, mapmlp=True, res_mapmlp=args.use_residual_mapmlp)
                    logits = model(feats, procD['session'], encode=False, mapmlp=False)

                if session == 0:
                    # logits_cls = logits[:, clsD['tasks'][0]]
                    logits_cls = logits
                    target_cls = clsD['class_maps'][0][test_label]
                    # tasks 대신 seen_unsort / seen_unsort_map으로 바꾼 후 아래랑 합쳐도 무방.
                    loss = F.cross_entropy(logits_cls, target_cls)
                    acc = count_acc(logits_cls, target_cls)
                else:
                    # logits = logits[:, clsD['seen_unsort']]
                    target_cls = clsD['seen_unsort_map'][test_label]
                    # seen, seen_map results same.
                    loss = F.cross_entropy(logits, target_cls)
                    acc = count_acc(logits, target_cls)

                vl.add(loss.item())
                va.add(acc)

            vl = vl.item()
            va = va.item()
        if session == 0:
            print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
        else:
            print('session {}, test, loss={:.4f} acc={:.4f}'.format(session, vl, va))

        return vl, va



    def main(self, args):
        timer = Timer()
        if not args.use_gaussgen:
            args, procD, clsD, bookD = self.init_vars(args)
        else:
            args, procD, clsD, bookD, gaussD = self.init_vars(args)

        if args.use_supconloss:
            assert args.use_head == True
        if args.rbfc_opt2:
            assert args.no_rbfc == False
        if args.use_mapmlp_in_base:
            assert args.use_mapmlp
        if args.use_residual_mapmlp:
            assert args.use_mapmlp

        # model = MYNET(args, mode=args.base_mode)
        # model = SupConResNet(name=opt.model)

        if args.dataset == 'cifar100':
            tmp_trainset = args.Dataset.CIFAR100(args, train=True, shotpercls=args.shotpercls, base_sess=True,
                                                 root=args.dataroot, doubleaug=args.base_doubleaug, download=True,
                                                 index=np.array(args.task_class_order))
            args.dataset_label2txt = {v: k for k, v in tmp_trainset.class_to_idx.items()}
        elif args.dataset == 'mini_imagenet':
            tmp_trainset = args.Dataset.MiniImageNet(args, train=True, shotpercls=args.shotpercls, base_sess=True,
                                                     root=args.dataroot, doubleaug=args.base_doubleaug,
                                                     index=np.array(args.task_class_order))
            mini_dic_loc = 'dataloader/miniimagenet/imagenet_label_textdic'
            with open(mini_dic_loc, 'rb') as f:
                mini_dic = pickle.load(f)
            args.dataset_label2txt = {}
            for i in range(args.num_classes):
                args.dataset_label2txt[i] = mini_dic[tmp_trainset.wnids[i]]
        elif args.dataset == 'cub200':
            tmp_trainset = args.Dataset.CUB200(args, train=True, shotpercls=args.shotpercls, base_sess=True,
                                               root=args.dataroot, doubleaug=args.base_doubleaug,
                                               index=np.array(args.task_class_order))
            mini_dic = {}
            for key, value in tmp_trainset.id2image.items():
                _str = value.split('/')[0]
                strs = _str.split('.')
                mini_dic[int(strs[0])-1] = strs[1]
            args.dataset_label2txt = mini_dic
        else:
            raise NotImplementedError

        zeroshot_clipmodel = CLIP_Model(args, keep_lang=True)
        print(f'Classification head for {args.model_type} on {args.dataset} exists at {args.text_clf_weight_fn}')
        if os.path.exists(args.text_clf_weight_fn):
            print('Loading %s'%(args.text_clf_weight_fn))
            textual_clf_weights = torch.load(args.text_clf_weight_fn)
        else:
            print('Creating head for %s on %s at %s' %(args.model_type, args.dataset, args.text_clf_weight_fn))
            #textual_clf_weights = get_zeroshot_weights(args)
            _words = [args.dataset_label2txt[i] for i in range(args.num_classes)]
            textual_clf_weights = get_words_weights(args, zeroshot_clipmodel.model, words=_words)
            torch.save(textual_clf_weights, args.text_clf_weight_fn)
            #textual_clf_weights.save(args.text_clf_head_path)

        if args.use_randomtext:
            if os.path.exists(args.randomtext_embed_fn):
                print(f'Random text embedding for {args.model_type} on {args.dataset} exists at {args.randomtext_embed_fn}')
                print('Loading %s'%(args.randomtext_embed_fn))
                randomtext_embed = torch.load(args.randomtext_embed_fn)
            else:
                print('Creating randomtext embeddings for %s on %s at %s' %(args.model_type, args.dataset, args.randomtext_embed_fn))
                #randomtext_embed = get_zeroshot_weights(args, randtxt=True)
                randomtext_embed = get_words_weights(args, zeroshot_clipmodel.model, words=list(english_words_lower_set))
                torch.save(randomtext_embed, args.randomtext_embed_fn)
            args.randomtext_embed = randomtext_embed

        model = MYNET(args, fw_mode=args.fw_mode, textual_clf_weights=textual_clf_weights)

        # model = MYNET(args, fw_mode=args.fw_mode)
        # model = SupConResNet(name=opt.model, head='linear')
        criterion = SupConLoss(temperature=args.supcontemp, supcon_angle=args.supcon_angle)

        model = nn.DataParallel(model, list(range(args.num_gpu)))
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        #if args.load_dir is not None:
        if args.load_ft_dir is not None:
            print('Loading init parameters from: %s_%s' % (args.load_ft_dir,args.load_ft_model_name))
            args.load_ft_model = os.path.join(args.load_ft_dir, args.load_ft_model_name)
            load_ft_model_weight = torch.load(args.load_ft_model)['params']
            model.load_state_dict(load_ft_model_weight, strict=True)




        supcon_criterion = SupConLoss(temperature=args.supcontemp, supcon_angle=args.supcon_angle)
        kdloss = KDLoss(args.s)

        writer = SummaryWriter(os.path.join(args.log_path, args.project))
        # for tsne plotting
        if args.plot_tsne:
            n_components = 2
            perplexity = 30
            sns.set_style('darkgrid')
            sns.set_palette('muted')
            sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
            # draw_base_cls_num = 60
            draw_base_cls_num = 15
            draw_n_per_basecls = 100

        t_start_time = time.time()




        # init train statistics
        result_list = [args]
        # args.seesions: total session num. == len(self.tasks)
        natsa_ = []

        if args.angle_exp:
            l_inter_angles_base = [];
            l_angle_intra_mean_base = [];
            l_angle_intra_std_base = []
            l_angle_feat_fc_base = [];
            l_angle_feat_fc_base_std = []
            l_inter_angles_inc = []
            l_angle_intra_mean_inc = [];
            l_angle_intra_std_inc = []
            l_angle_feat_fc_inc = [];
            l_angle_feat_fc_inc_std = []
            l_angle_base_feats_new_clf = []
            l_angle_base_clfs_new_feat = []
            l_base_inc_fc_angle = []
            l_inc_inter_fc_angle = []
            l_angle_featmean_fc_base = [];
            l_angle_featmean_fc_inc = []

        # init train statistics
        result_list = [args]

        # args.seesions: total session num. == len(self.tasks)
        for session in range(args.start_session, args.sessions):
            procD['step'] = -1
            procD['epoch'] = -1
            procD['session'] += 1
            clsD = inc_maps(args, clsD, procD, procD['session'])

            if session == 0:
                train_set, trainloader, testloader = get_dataloader(args, procD, clsD, bookD)
                trainloader.dataset.transform = model.module.train_preprocess
                trainloader.dataset.transform2 = model.module.train_preprocess
                testloader.dataset.transform = model.module.val_preprocess
            else:
                train_set, trainloader, testloader, new_testloader, prev_testloader, new_all_testloader, base_testloader \
                    = get_dataloader(args, procD, clsD, bookD)
                trainloader.dataset.transform = model.module.train_preprocess
                trainloader.dataset.transform2 = model.module.train_preprocess
                testloader.dataset.transform = model.module.val_preprocess
                new_testloader.dataset.transform = model.module.val_preprocess
                prev_testloader.dataset.transform = model.module.val_preprocess
                base_testloader.dataset.transform = model.module.val_preprocess
                new_all_testloader.dataset.transform = model.module.val_preprocess


            model.module.freeze_head()
            args.print_every = 100

            # Should erase this part. but how to use preprocess_fn on existing dataloader? train & test.
            num_batches = len(trainloader)

            if not args.use_flyp_ft_v1 and not args.use_flyp_ft_v2:
                loss_fn = torch.nn.CrossEntropyLoss()
            else:
                loss_fn = ClipLoss(local_loss=False,
                                   gather_with_grad=False,
                                   cache_labels=True,
                                   rank=0,
                                   world_size=1,
                                   use_horovod=False)



            if session == 0:  # load base class train img label
                if not args.secondphase:
                    if args.angle_exp:
                        init_inter_angles, init_intra_angle_mean, init_intra_angle_std, init_angle_feat_fc, \
                        init_angle_feat_fc_std, init_angle_featmean_fc = \
                            base_angle_exp(args, trainloader, args.base_doubleaug, procD, clsD, model,
                                           testloader.dataset.transform)
                        te_init_inter_angles, te_init_intra_angle_mean, te_init_intra_angle_std, te_init_angle_feat_fc, \
                        te_init_angle_feat_fc_std, te_init_angle_featmean_fc = \
                            base_angle_exp(args, testloader, False, procD, clsD, model)
                        print('sess 0 bef train')
                        print(init_inter_angles, init_intra_angle_mean, init_intra_angle_std, init_angle_feat_fc, \
                              init_angle_feat_fc_std, init_angle_featmean_fc)
                        print(te_init_inter_angles, te_init_intra_angle_mean, te_init_intra_angle_std,
                              te_init_angle_feat_fc, \
                              te_init_angle_feat_fc_std, te_init_angle_featmean_fc)
                        print('text classifiers inter angle: %d'%(get_inter_angle(model.module.textual_classifier.weight)))

                    print('new classes for this session:\n', np.unique(train_set.targets))
                    model, optimizer, scheduler = self.get_optimizer_base(args, model, num_batches)

                    angles = []
                    for epoch in range(args.epochs_base):
                        procD['epoch'] += 1
                        start_time = time.time()
                        # train base sess

                        if args.angle_exp:
                            angle = get_intra_avg_angle_from_loader(args, trainloader, args.base_doubleaug, procD, clsD,
                                                                    model)
                            angles.append(angle)

                        #tl, ta = self.base_train(args, model, procD, clsD, trainloader, optimizer, kdloss, scheduler,
                        #                         epoch, supcon_criterion)
                        if not args.use_flyp_ft_v1 and not args.use_flyp_ft_v2:
                            tl, ta = self.base_train(args, model, procD, clsD, trainloader, optimizer, kdloss, scheduler,
                                                 epoch, loss_fn, supcon_criterion)
                            procD['trlog']['train_loss'].append(tl)
                            procD['trlog']['train_acc'].append(ta)
                            result_list.append(
                                'epoch:%03d,training_loss:%.5f,training_acc:%.5f' % (epoch, tl, ta))
                            writer.add_scalar('Session {0} - Loss/train'.format(session), tl, epoch)

                        else:
                            tl = self.base_train(args, model, procD, clsD, trainloader, optimizer, kdloss, scheduler,
                                                     epoch, loss_fn, supcon_criterion)
                            procD['trlog']['train_loss'].append(tl)
                            result_list.append(
                                'epoch:%03d,training_loss:%.5f' % (epoch, tl))
                            writer.add_scalar('Session {0} - Loss/train'.format(session), tl, epoch)

                        """
                        # test model with all seen class
                        tsl, tsa = self.test(args, model, procD, clsD, testloader)  ####
                        procD['trlog']['test_loss'].append(tsl)
                        procD['trlog']['test_acc'].append(tsa)
                        result_list.append(
                            'epoch:%03d,test_loss:%.5f,test_acc:%.5f' % (
                                epoch, tsl, tsa))
                        writer.add_scalar('Session {0} - Acc/val_ncm'.format(session), tsa, epoch)
                        """
                        #scheduler.step()

                        if epoch == args.epochs_base-1:
                            save_model_dir = os.path.join(args.ft_save_path, 'session' + str(session) \
                                                          + '_epo' + str(epoch) + '_acc.pth')
                            torch.save(dict(params=model.state_dict()), save_model_dir)

                            if args.use_gaussgen:
                                gaussD = learn_gauss(args, trainloader, model, clsD, procD, gaussD)
                            # gaussD is used from sess>0 so no matter here or at last of sess 0.
                            if not args.use_gaussgen:
                                save_obj(args.ft_save_path, procD, clsD, bookD)
                            else:
                                save_obj(args.ft_save_path, procD, clsD, bookD, gaussD)
                            print('save path is %s'%(args.ft_save_path))

                if args.use_flyp_ft_v1 or args.use_flyp_ft_v2:
                    _words = [args.dataset_label2txt[i] for i in range(args.num_classes)]
                    ft_textual_clf_weights = get_words_weights(args, model.module.clip_encoder.model,
                                                            words=_words)
                    ft_textual_clf_weights = ft_textual_clf_weights[args.task_class_order].cuda().detach()
                    #model.module.textual_classifier.load_weight(ft_textual_clf_weights)
                    model.module.update_text_clf(ft_textual_clf_weights)

                # test model with all seen class
                tsl, tsa = self.test(args, model, procD, clsD, testloader)  ####
                # tsl, tsa = self.test2(args, model, procD, clsD, trainloader, testloader)  ####
                procD['trlog']['max_acc'][session] = float('%.3f' % (tsa * 100))
                procD['trlog']['test_loss'].append(tsl)
                procD['trlog']['test_acc'].append(tsa)
                result_list.append(
                    'test_loss:%.5f,test_acc:%.5f' % (tsl, tsa))
                writer.add_scalar('Session {0} - Acc/val_ncm'.format(session), tsa)



                if args.epochs_base == 0:
                    epoch = -1
                    if args.angle_exp:
                        init_inter_angles, init_intra_angle_mean, init_intra_angle_std, init_angle_feat_fc, \
                        init_angle_feat_fc_std, init_angle_featmean_fc = \
                            base_angle_exp(args, trainloader, args.base_doubleaug, procD, clsD, model,
                                           testloader.dataset.transform)
                        te_init_inter_angles, te_init_intra_angle_mean, te_init_intra_angle_std, te_init_angle_feat_fc, \
                        te_init_angle_feat_fc_std, te_init_angle_featmean_fc = \
                            base_angle_exp(args, testloader, False, procD, clsD, model)
                        print('sess 0 epoc_base 0')
                        print(init_inter_angles, init_intra_angle_mean, init_intra_angle_std, init_angle_feat_fc, \
                              init_angle_feat_fc_std, init_angle_featmean_fc )
                        print(te_init_inter_angles, te_init_intra_angle_mean, te_init_intra_angle_std, te_init_angle_feat_fc, \
                              te_init_angle_feat_fc_std, te_init_angle_featmean_fc)
                    """
                    # Duplicated so remove
                    tsl, tsa = self.test(args, model, procD, clsD, testloader)  ####
                    procD['trlog']['max_acc'][session] = float('%.3f' % (tsa * 100))
                    procD['trlog']['test_loss'].append(tsl)
                    procD['trlog']['test_acc'].append(tsa)
                    result_list.append(
                        'epoch:%03d, test_loss:%.5f,test_acc:%.5f' % (
                            epoch, tsl, tsa))
                    writer.add_scalar('Session {0} - Acc/val_ncm'.format(session), tsa, epoch)
                    writer.add_scalar('Session {0} - Learning rate/train'.format(session), epoch)
                    
                    """
                    # scheduler.step()

                    if args.use_gaussgen:
                        gaussD = learn_gauss(args, trainloader, model, clsD, procD, gaussD)
                    # gaussD is used from sess>0 so no matter here or at last of sess 0.
                    if not args.use_gaussgen:
                        save_obj(args.ft_save_path, procD, clsD, bookD)
                    else:
                        save_obj(args.ft_save_path, procD, clsD, bookD, gaussD)
                    save_model_dir = os.path.join(args.ft_save_path, 'session' + str(session) \
                                                  + '_epob0_model' + str(epoch) + '_acc.pth')
                    torch.save(dict(params=model.state_dict()), save_model_dir)

                if args.plot_tsne:
                    base_tsne_idx = torch.arange(args.base_class)[
                        torch.randperm(args.base_class)[:draw_base_cls_num]]
                    palette = np.array(sns.color_palette("hls", args.base_class))
                    data_, label_ = tot_datalist(args, trainloader, model, args.base_doubleaug, \
                                                 clsD['seen_unsort_map'], gpu=False)
                    data_, label_ = selec_datalist(args, data_, label_, base_tsne_idx, draw_n_per_basecls)
                    draw_tsne(data_, label_, n_components, perplexity, palette, base_tsne_idx, 'base train')

                    data_, label_ = tot_datalist(args, testloader, model, False, clsD['seen_unsort_map'], gpu=False)
                    data_, label_ = selec_datalist(args, data_, label_, base_tsne_idx, draw_n_per_basecls)
                    draw_tsne(data_, label_, n_components, perplexity, palette, base_tsne_idx, 'base test')

                if args.angle_exp:
                    afterbase_inter_angles, afterbase_intra_angle_mean, afterbase_intra_angle_std, afterbase_angle_feat_fc, \
                    afterbase_angle_feat_fc_std, afterbase_angle_featmean_fc = \
                        base_angle_exp(args, trainloader, args.base_doubleaug, procD, clsD, model,
                                       testloader.dataset.transform)
                    te_afterbase_inter_angles, te_afterbase_intra_angle_mean, te_afterbase_intra_angle_std, \
                    te_afterbase_angle_feat_fc, te_afterbase_angle_feat_fc_std, te_afterbase_angle_featmean_fc = \
                        base_angle_exp(args, testloader, False, procD, clsD, model)
                    print('sess 0 after train')
                    print(afterbase_inter_angles, afterbase_intra_angle_mean, afterbase_intra_angle_std, afterbase_angle_feat_fc, \
                          afterbase_angle_feat_fc_std, afterbase_angle_featmean_fc)
                    print(te_afterbase_inter_angles, te_afterbase_intra_angle_mean, te_afterbase_intra_angle_std, \
                          te_afterbase_angle_feat_fc, te_afterbase_angle_feat_fc_std, te_afterbase_angle_featmean_fc)
                # fig, (ax1,ax2,ax3) = plt.subplots(1,3)
                # ax1.plot(range(args.epochs_base), angles)
                # ax2.plot(range(args.epochs_base), tas)
                # ax3.plot(range(args.epochs_base), tsas)
                # plt.show()
                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, procD['trlog']['max_acc_epoch'], procD['trlog']['max_acc'][session], ))



            else:  # incremental learning sessions
                print("training session: [%d]" % session)
                num_batches = len(trainloader)
                model, optimizer, scheduler = self.get_optimizer_new(args, model, num_batches)
                transform_ = trainloader.dataset.transform
                trainloader.dataset.transform = testloader.dataset.transform
                trainloader.dataset.transform = transform_

                for epoch in range(args.epochs_new):
                    procD['epoch'] += 1
                    start_time = time.time()
                    if not args.use_gaussgen:
                        if not args.use_flyp_ft_inc:
                            tl, ta = self.new_train(args, model, procD, clsD, trainloader, optimizer, kdloss, scheduler,
                                                     epoch, loss_fn, supcon_criterion)
                        else:
                            tl = self.new_train(args, model, procD, clsD, trainloader, optimizer, kdloss, scheduler,
                                                 epoch, loss_fn, supcon_criterion)
                    else:
                        tl = self.new_train_featgen(args, model, procD, clsD, trainloader, optimizer, kdloss, scheduler,
                                                    epoch, loss_fn, supcon_criterion, gaussD=gaussD)
                        #if not args.use_flyp_ft_inc:
                        #    tl, ta = self.new_train_featgen(args, model, procD, clsD, trainloader, optimizer, kdloss, scheduler,
                        #                             epoch, loss_fn, supcon_criterion)
                        #else:
                        #    tl = self.new_train_featgen(args, model, procD, clsD, trainloader, optimizer, kdloss, scheduler,
                        #                         epoch, loss_fn, supcon_criterion)
                print('Incremental session, test')

                model.eval()
                tsl, tsa = self.test(args, model, procD, clsD, testloader)
                ntsl, ntsa = self.test(args, model, procD, clsD, new_testloader)
                ptsl, ptsa = self.test(args, model, procD, clsD, prev_testloader)

                _, btsa = self.test(args, model, procD, clsD, base_testloader)
                _, natsa = self.test(args, model, procD, clsD, new_all_testloader)

                # save model
                procD['trlog']['max_acc'][session] = float('%.3f' % (tsa * 100))
                procD['trlog']['new_max_acc'][session] = float('%.3f' % (ntsa * 100))
                procD['trlog']['new_all_max_acc'][session] = float('%.3f' % (natsa * 100))
                procD['trlog']['base_max_acc'][session] = float('%.3f' % (btsa * 100))
                procD['trlog']['prev_max_acc'][session] = float('%.3f' % (ptsa * 100))

                result_list.append(
                    'Session {}, test Acc {:.3f}\n'.format(session, procD['trlog']['max_acc'][session]))

                if args.use_gaussgen:
                    gaussD = learn_gauss(args, trainloader, model, clsD, procD, gaussD)

                if args.plot_tsne:
                    # if session !=  args.sessions -1:
                    #    continue
                    base_tsne_idx = torch.arange(args.base_class)[torch.randperm(args.base_class)[:draw_base_cls_num]]
                    palette = np.array(sns.color_palette("hls", args.base_class + args.way * session))

                    data_base_, label_base_ = tot_datalist(args, base_testloader, model, False, clsD['seen_unsort_map'],
                                                           gpu=False)
                    data_base_, label_base_ = selec_datalist(args, data_base_, label_base_, base_tsne_idx,
                                                             draw_n_per_basecls)
                    # draw_tsne(data_, label_, n_components, perplexity, palette, base_tsne_idx, 'base test')

                    new_tsne_idx = torch.arange(args.way * session) + args.base_class
                    new_tsne_idx = new_tsne_idx[-20:]
                    data_new_, label_new_ = tot_datalist(args, new_all_testloader, model, False,
                                                         clsD['seen_unsort_map'], gpu=False)
                    data_new_, label_new_ = selec_datalist(args, data_new_, label_new_, new_tsne_idx,
                                                           draw_n_per_basecls)

                    data_ = torch.cat((data_base_, data_new_), dim=0)
                    label_ = torch.cat((label_base_, label_new_), dim=0)
                    combine_tsne_idx = torch.cat((base_tsne_idx, new_tsne_idx), dim=0)
                    # draw_tsne(data_, label_, n_components, perplexity, palette, combine_tsne_idx, 'new test')
                    palette2 = np.array(sns.color_palette("hls", args.way * session))
                    lll = label_new_ - args.base_class
                    iii = new_tsne_idx - args.base_class
                    # draw_tsne(data_new_, label_new_, n_components, perplexity, palette2, new_tsne_idx, 'new test')
                    draw_tsne(data_new_, lll, n_components, perplexity, palette2, iii, 'new test')


                # if session == 1:
                if args.angle_exp:
                    if session == args.sessions-1:
                        inter_angles_base, angle_intra_mean_base, angle_intra_std_base, angle_feat_fc_base, angle_feat_fc_base_std, \
                        inter_angles_inc, angle_intra_mean_inc, angle_intra_std_inc, angle_feat_fc_inc, angle_feat_fc_inc_std, \
                        angle_base_feats_new_clf, angle_base_clfs_new_feat, base_inc_fc_angle, inc_inter_fc_angle, \
                        angle_featmean_fc_base, angle_featmean_fc_inc \
                            = inc_angle_exp(args, base_testloader, new_testloader, args.base_doubleaug, procD, clsD, model)

                        l_inter_angles_base.append(inter_angles_base)
                        l_angle_intra_mean_base.append(angle_intra_mean_base)
                        l_angle_intra_std_base.append(angle_intra_std_base)
                        l_angle_feat_fc_base.append(angle_feat_fc_base)
                        l_angle_feat_fc_base_std.append(angle_feat_fc_base_std)
                        l_inter_angles_inc.append(inter_angles_inc)
                        l_angle_intra_mean_inc.append(angle_intra_mean_inc)
                        l_angle_intra_std_inc.append(angle_intra_std_inc)
                        l_angle_feat_fc_inc.append(angle_feat_fc_inc)
                        l_angle_feat_fc_inc_std.append(angle_feat_fc_inc_std)
                        l_angle_base_feats_new_clf.append(angle_base_feats_new_clf)
                        l_angle_base_clfs_new_feat.append(angle_base_clfs_new_feat)
                        l_base_inc_fc_angle.append(base_inc_fc_angle)
                        l_inc_inter_fc_angle.append(inc_inter_fc_angle)
                        l_angle_featmean_fc_base.append(angle_featmean_fc_base)
                        l_angle_featmean_fc_inc.append(angle_featmean_fc_inc)

                    # print(inter_angles_base, angle_intra_std_base, angle_feat_fc_base, inter_angles_inc, angle_intra_std_inc, \
                    # angle_feat_fc_inc, angle_base_feats_new_clf, angle_base_clfs_new_feat, base_inc_fc_angle, inc_inter_fc_angle)

        result_list.append('Base Session Best Epoch {}\n'.format(procD['trlog']['max_acc_epoch']))
        result_list.append(procD['trlog']['max_acc'])
        print('max_acc:', procD['trlog']['max_acc'])
        print('max_acc_base_rbfc:', procD['trlog']['max_acc_base_rbfc'])
        print('max_acc_base_after_ft:', procD['trlog']['max_acc_base_after_ft'])
        print('new_max_acc:', procD['trlog']['new_max_acc'])
        print('new_all_max_acc:', procD['trlog']['new_all_max_acc'])
        print('base_max_acc:', procD['trlog']['base_max_acc'])
        print('prev_max_acc:', procD['trlog']['prev_max_acc'])
        save_list_to_txt(os.path.join(args.ft_save_path, 'results.txt'), result_list)
        #save_obj(args.ft_save_path, procD, clsD, bookD)

        print('Base Session Best epoch:', procD['trlog']['max_acc_epoch'])

        if args.angle_exp:
            print('base exp result')
            print(init_inter_angles, init_intra_angle_mean, init_intra_angle_std, init_angle_feat_fc,
                  init_angle_feat_fc_std, \
                  init_angle_featmean_fc)
            print(te_init_inter_angles, te_init_intra_angle_mean, te_init_intra_angle_std, te_init_angle_feat_fc, \
                  te_init_angle_feat_fc_std, te_init_angle_featmean_fc)
            print(afterbase_inter_angles, afterbase_intra_angle_mean, afterbase_intra_angle_std, \
                  afterbase_angle_feat_fc, afterbase_angle_feat_fc_std, afterbase_angle_featmean_fc)
            print(te_afterbase_inter_angles, te_afterbase_intra_angle_mean, te_afterbase_intra_angle_std, \
                  te_afterbase_angle_feat_fc, te_afterbase_angle_feat_fc_std, te_afterbase_angle_featmean_fc)

            print('inc exp result')
            print(l_inter_angles_base)
            print(l_angle_intra_mean_base)
            print(l_angle_intra_std_base)
            print(l_angle_feat_fc_base)
            print(l_angle_feat_fc_base_std)
            print(l_inter_angles_inc)
            print(l_angle_intra_mean_inc)
            print(l_angle_intra_std_inc)
            print(l_angle_feat_fc_inc)
            print(l_angle_feat_fc_inc_std)
            print(l_angle_base_feats_new_clf)
            print(l_angle_base_clfs_new_feat)
            print(l_base_inc_fc_angle)
            print(l_inc_inter_fc_angle)
            print(l_angle_featmean_fc_base)
            print(l_angle_featmean_fc_inc)
            print('angles')
            print(angles)
