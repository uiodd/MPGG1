import os

from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import numpy as np, argparse, time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset
from model import MaskedNLLLoss, MaskedKLDivLoss, Transformer_Based_Model
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import pickle as pk
import datetime

from emtion_constraint_loss import EmotionConstraintLoss

# Comprehensive random seed control should include
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_random_seed(seed=42):
    """
    Set random seeds for all relevant libraries to make results reproducible.
    """
    # Python random
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic CUDA behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for better determinism
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"随机种子已设置为: {seed}")


def get_train_valid_sampler(trainset, valid=0.1, dataset='MELD'):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = MELDDataset('/media/asus/SATA2/lyz/data/meld_multimodal_features.pkl')
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, 'MELD')
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset('/media/asus/SATA2/lyz/data/meld_multimodal_features.pkl', train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset('/media/asus/SATA2/lyz/data/iemocap_multimodal_features.pkl')
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset('/media/asus/SATA2/lyz/data/iemocap_multimodal_features.pkl', train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader



def train_or_eval_model(model, loss_function, kl_loss, emotion_loss, dataloader, epoch, optimizer=None, train=False,
                        gamma_1=1.0, gamma_2=1.0, gamma_3=1.0, gamma_4=0.1, gamma_5=0.1):
    losses, preds, labels, masks = [], [], [], []

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if torch.cuda.is_available() else data[:-1]
        qmask = qmask.permute(1, 0, 2)
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        emotion_labels = label if train else None
        log_prob1, log_prob2, log_prob3, all_log_prob, all_prob, \
            kl_log_prob1, kl_log_prob2, kl_log_prob3, kl_all_prob, \
            t_features, a_features, v_features, all_features, cluster_contrastive_loss, proto_loss, contrastive_features = model(textf, visuf, acouf,
                                                                                               umask, qmask, lengths,
                                                                                               emotion_labels=emotion_labels)

        lp_1 = log_prob1.view(-1, log_prob1.size()[2])
        lp_2 = log_prob2.view(-1, log_prob2.size()[2])
        lp_3 = log_prob3.view(-1, log_prob3.size()[2])
        lp_all = all_log_prob.view(-1, all_log_prob.size()[2])
        labels_ = label.view(-1)

        kl_lp_1 = kl_log_prob1.view(-1, kl_log_prob1.size()[2])
        kl_lp_2 = kl_log_prob2.view(-1, kl_log_prob2.size()[2])
        kl_lp_3 = kl_log_prob3.view(-1, kl_log_prob3.size()[2])
        kl_p_all = kl_all_prob.view(-1, kl_all_prob.size()[2])

        # Original loss computation
        loss = gamma_1 * loss_function(lp_all, labels_, umask) + \
               gamma_2 * (loss_function(lp_1, labels_, umask) + loss_function(lp_2, labels_, umask) + loss_function(
            lp_3, labels_, umask)) + \
               gamma_3 * (kl_loss(kl_lp_1, kl_p_all, umask) + kl_loss(kl_lp_2, kl_p_all, umask) + kl_loss(kl_lp_3,
                                                                                                          kl_p_all,
                                                                                                          umask))


        if emotion_loss is not None:
            # Handle feature/label shapes
            mask_flat = umask.view(-1)
            valid_indices = mask_flat.nonzero().squeeze()
            if valid_indices.numel() > 0:
                # Ensure valid_indices is 1D
                if valid_indices.dim() == 0:
                    valid_indices = valid_indices.unsqueeze(0)
                # Multi-level feature constraints
                t_feat_flat = t_features.view(-1, t_features.size(-1))[valid_indices]
                a_feat_flat = a_features.view(-1, a_features.size(-1))[valid_indices]
                v_feat_flat = v_features.view(-1, v_features.size(-1))[valid_indices]
                all_feat_flat = all_features.view(-1, all_features.size(-1))[valid_indices]
                valid_labels = labels_[valid_indices]
                # Compute emotion constraint losses per level
                emotion_loss_t = emotion_loss(t_feat_flat, valid_labels, lp_1[valid_indices],
                                              emotion_loss.similarity_matrix)
                emotion_loss_a = emotion_loss(a_feat_flat, valid_labels, lp_2[valid_indices],
                                              emotion_loss.similarity_matrix)
                emotion_loss_v = emotion_loss(v_feat_flat, valid_labels, lp_3[valid_indices],
                                              emotion_loss.similarity_matrix)
                emotion_loss_all = emotion_loss(all_feat_flat, valid_labels, lp_all[valid_indices],
                                                emotion_loss.similarity_matrix)
                # Ensure the loss is a scalar
                total_emotion_loss = (emotion_loss_t + emotion_loss_a + emotion_loss_v + emotion_loss_all) / 4.0
                if total_emotion_loss.dim() > 0:
                    total_emotion_loss = total_emotion_loss.mean()
                # Add to total loss
                loss = loss + gamma_4 * total_emotion_loss


        if cluster_contrastive_loss is not None:
            if isinstance(cluster_contrastive_loss, dict):
                zero_tensor = torch.tensor(0.0, device=loss.device)
                total_contrastive_loss = cluster_contrastive_loss.get('total_loss', zero_tensor)
                loss = loss + gamma_5 * total_contrastive_loss
                
                if train and epoch % 10 == 0 and len(losses) % 100 == 0:
                    cluster_loss_val = cluster_contrastive_loss.get('cluster_loss', zero_tensor).item()
                    instance_loss_val = cluster_contrastive_loss.get('instance_loss', zero_tensor).item()
                    prototype_loss_val = cluster_contrastive_loss.get('prototype_loss', zero_tensor).item()
                    print(f"Cluster Loss - Total: {total_contrastive_loss.item():.4f}, "
                          f"Cluster: {cluster_loss_val:.4f}, "
                          f"Instance: {instance_loss_val:.4f}, "
                          f"Prototype: {prototype_loss_val:.4f}")
            else: # Assuming it's a single tensor
                loss = loss + gamma_5 * cluster_contrastive_loss
        
        # Add proto loss
        if proto_loss is not None:
            loss = loss + args.lambda_proto * proto_loss

        lp_ = all_prob.view(-1, all_prob.size()[2])

        pred_ = torch.argmax(lp_, 1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item() * masks[-1].sum())
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
            

            if train:
                with torch.no_grad():
                    if contrastive_features is not None:
                        model.enhanced_gated_attention.update_prototypes_from_projected(contrastive_features, label)

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan')

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.000153, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.305, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')
    parser.add_argument('--hidden_dim', type=int, default=1024, metavar='hidden_dim', help='output hidden size')
    parser.add_argument('--n_head', type=int, default=8, metavar='n_head', help='number of heads')
    parser.add_argument('--epochs', type=int, default=150, metavar='E', help='number of epochs')
    parser.add_argument('--temp', type=float, default=1, metavar='temp', help='temp')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')
    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test')
    parser.add_argument('--gamma_4', type=float, default=0.0, metavar='gamma_4', help='emotion constraint loss weight')
    parser.add_argument('--gamma_5', type=float, default=0.1, metavar='gamma_5', help='cluster contrastive loss weight')
    parser.add_argument('--lambda_center', type=float, default=0.0, metavar='lambda_center', help='center loss weight')
    parser.add_argument('--lambda_soft_cosine', type=float, default=0, metavar='lambda_soft_cosine', help='soft cosine loss weight')
    parser.add_argument('--lambda_proto', type=float, default=0.131, metavar='lambda_proto', help='proto loss weight')
    parser.add_argument('--projection_dim', type=int, default=32, help='Dimension for contrastive projection head')
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')

    args = parser.parse_args()
    # Force-disable emotion constraint terms (override any externally provided values)
    args.gamma_4 = 0.0
    args.lambda_center = 0.0
    args.lambda_soft_cosine = 0.0
    print(args)

    # Set random seed before any other operations
    set_random_seed(args.seed)

    cuda = torch.cuda.is_available() and not args.no_cuda
    if cuda:
        print('Running on GPU')
        torch.cuda.set_device(0)
    else:
        print('Running on CPU')

    if args.tensorboard:
        writer = SummaryWriter()
    n_epochs = args.epochs
    batch_size = args.batch_size
    feat2dim = {'IS10': 1582, 'denseface': 342, 'MELD_audio': 300}
    D_audio = feat2dim['IS10'] if args.Dataset == 'IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = 1024

    D_m = D_audio + D_visual + D_text

    n_speakers = 9 if args.Dataset == 'MELD' else 2
    n_classes = 7 if args.Dataset == 'MELD' else 6 if args.Dataset == 'IEMOCAP' else 1

    print('temp {}'.format(args.temp))

    model = Transformer_Based_Model(args.Dataset, args.temp, D_text, D_visual, D_audio, args.n_head,
                                    n_classes=n_classes,
                                    hidden_dim=args.hidden_dim,
                                    n_speakers=n_speakers,
                                    dropout=args.dropout,
                                    projection_dim=args.projection_dim)

    total_params = sum(p.numel() for p in model.parameters())
    print('total parameters: {}'.format(total_params))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('training parameters: {}'.format(total_trainable_params))

    if cuda:
        model.cuda()

    kl_loss = MaskedKLDivLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)


    emotion_constraint_loss = EmotionConstraintLoss(
        num_classes=n_classes,
        feature_dim=args.hidden_dim,
        lambda_center=args.lambda_center,
        lambda_soft_cosine=args.lambda_soft_cosine
    )

    # [Added] Define emotion similarity matrix
    if args.Dataset == 'IEMOCAP':
        # IEMOCAP: happy, sad, neutral, angry, excited, frustrated
        similarity_matrix = torch.tensor([
            [1.0, 0.1, 0.3, 0.2, 0.8, 0.3],  # happy
            [0.1, 1.0, 0.2, 0.4, 0.1, 0.6],  # sad
            [0.3, 0.2, 1.0, 0.1, 0.2, 0.1],  # neutral
            [0.2, 0.4, 0.1, 1.0, 0.3, 0.8],  # angry
            [0.8, 0.1, 0.2, 0.3, 1.0, 0.2],  # excited
            [0.3, 0.6, 0.1, 0.8, 0.2, 1.0]  # frustrated
        ])
    else:  # MELD
        # MELD: neutral, surprise, fear, sadness, joy, disgust, anger
        similarity_matrix = torch.tensor([
            [1.0, 0.2, 0.1, 0.2, 0.3, 0.1, 0.1],  # neutral
            [0.2, 1.0, 0.4, 0.1, 0.6, 0.2, 0.3],  # surprise
            [0.1, 0.4, 1.0, 0.5, 0.1, 0.3, 0.4],  # fear
            [0.2, 0.1, 0.5, 1.0, 0.1, 0.2, 0.3],  # sadness
            [0.3, 0.6, 0.1, 0.1, 1.0, 0.1, 0.2],  # joy
            [0.1, 0.2, 0.3, 0.2, 0.1, 1.0, 0.6],  # disgust
            [0.1, 0.3, 0.4, 0.3, 0.2, 0.6, 1.0]  # anger
        ])

    if cuda:
        emotion_constraint_loss.cuda()
        similarity_matrix = similarity_matrix.cuda()

    # Set similarity matrix
    emotion_constraint_loss.similarity_matrix = similarity_matrix

    if args.Dataset == 'MELD':
        loss_function = MaskedNLLLoss()
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0,
                                                                   batch_size=batch_size,
                                                                   num_workers=0)
    elif args.Dataset == 'IEMOCAP':
        loss_weights = torch.FloatTensor([1 / 0.086747,
                                          1 / 0.144406,
                                          1 / 0.227883,
                                          1 / 0.160585,
                                          1 / 0.127711,
                                          1 / 0.252668])
        loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                      batch_size=batch_size,
                                                                      num_workers=0)
    else:
        print("There is no such dataset")

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []

    for e in range(n_epochs):
        start_time = time.time()


        train_loss, train_acc, _, _, _, train_fscore = train_or_eval_model(model, loss_function, kl_loss,
                                                                           emotion_constraint_loss, train_loader, e,
                                                                           optimizer, True, gamma_4=args.gamma_4,
                                                                           gamma_5=args.gamma_5)
        valid_loss, valid_acc, _, _, _, valid_fscore = train_or_eval_model(model, loss_function, kl_loss,
                                                                           emotion_constraint_loss, valid_loader, e,
                                                                           gamma_4=args.gamma_4, gamma_5=args.gamma_5)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore = train_or_eval_model(model, loss_function,
                                                                                                 kl_loss,
                                                                                                 emotion_constraint_loss,
                                                                                                 test_loader, e,
                                                                                                 gamma_4=args.gamma_4,
                                                                                                 gamma_5=args.gamma_5)
        all_fscore.append(test_fscore)

        if best_fscore == None or best_fscore < test_fscore:
            best_fscore = test_fscore
            best_label, best_pred, best_mask = test_label, test_pred, test_mask

        if args.tensorboard:
            writer.add_scalar('Train/Loss', train_loss, e)
            writer.add_scalar('Train/Accuracy', train_acc, e)
            writer.add_scalar('Train/F1_score', train_fscore, e)
            writer.add_scalar('Valid/Loss', valid_loss, e)
            writer.add_scalar('Valid/Accuracy', valid_acc, e)
            writer.add_scalar('Valid/F1_score', valid_fscore, e)
            writer.add_scalar('Test/Loss', test_loss, e)
            writer.add_scalar('Test/Accuracy', test_acc, e)
            writer.add_scalar('Test/F1_score', test_fscore, e)

        print(
            'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
                format(e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss,
                       test_acc,
                       test_fscore, round(time.time() - start_time, 2)))
        if (e + 1) % 10 == 0:
            print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
            print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))

    if args.tensorboard:
        writer.close()

    print('Test performance..')
    print('F-Score: {}'.format(max(all_fscore)))
    print('F-Score-index: {}'.format(all_fscore.index(max(all_fscore)) + 1))

    today = datetime.date.today()
    if not os.path.exists("record_{}_{}_{}.pk".format(today.year, today.month, today.day)):
        with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day), 'wb') as f:
            pk.dump({}, f)
    with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day), 'rb') as f:
        record = pk.load(f)
    key_ = 'name_'
    if record.get(key_, False):
        record[key_].append(max(all_fscore))
    else:
        record[key_] = [max(all_fscore)]
    if record.get(key_ + 'record', False):
        record[key_ + 'record'].append(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
    else:
        record[key_ + 'record'] = [classification_report(best_label, best_pred, sample_weight=best_mask, digits=4)]
    with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day), 'wb') as f:
        pk.dump(record, f)

    print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
    print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))

    print('Final F-Score: {}'.format(max(all_fscore)))
