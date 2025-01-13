import logging
import os
from abc import abstractmethod
import wandb
import torch.nn.functional as F
import json

import torch
from numpy import inf
def fbert_score(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """
    Compute an F-BERT–style score between two sets of BERT embeddings.

    Args:
        emb1 (torch.Tensor): Embeddings from the first sequence (shape: [len1, hidden_dim]).
        emb2 (torch.Tensor): Embeddings from the second sequence (shape: [len2, hidden_dim]).

    Returns:
        float: The F-BERT score (F1-like measure of alignment between embeddings).
    """
    # Compute pairwise cosine similarity
    # sim has shape [len1, len2]
    sim = F.cosine_similarity(emb1.unsqueeze(1), emb2.unsqueeze(0), dim=-1)

    # Precision: for each token in emb1, find the best-match in emb2
    precision = sim.max(dim=1)[0].mean()

    # Recall: for each token in emb2, find the best-match in emb1
    recall = sim.max(dim=0)[0].mean()

    # F-BERT score (harmonic mean of Precision and Recall)
    f_score = 2 * precision * recall / (precision + recall + 1e-8)

    return -f_score

class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler):
        self.args = args

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        # start a new wandb run:
        wandb.init(
            # set the wandb project where this run will be logged
            project="R2GenCMN",

            # track hyperparameters and run metadata
            config={
                "model": model,
                "criterion": criterion,
                "dataset": args.dataset_name,
                "metric_ftns": metric_ftns,
                "optimizer": optimizer,
                "scheduler": lr_scheduler,
                "epochs": self.args.epochs,
            }
        )

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result,test_res,test_gts = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                            self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, test_res,test_gts,save_best=best)

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        self.logger.info('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))

        self.logger.info('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, test_res,test_gts, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
            }

        ## save results to file:
        results = {i:[test_gts[i],test_res[i]] for i in range(len(test_gts))}

        filename = os.path.join(self.checkpoint_dir, f'current_checkpoint_{self.args.run_name}.pth')
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        current_results_path = os.path.join(self.checkpoint_dir, f'model_current_results_{self.args.run_name}.json')
        with open(current_results_path, "w") as outfile:
            json.dump(results, outfile)
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, f'model_best_{self.args.run_name}.pth')
            best_results_path = os.path.join(self.checkpoint_dir, f'model_best_results_{self.args.run_name}.json')
            torch.save(state, best_path)
            with open(best_results_path, "w") as outfile:
                json.dump(results, outfile)
            # torch.save(results, best_results_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader,
                 val_dataloader, test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        # self.clinicalbert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(self.device)
        # self.clinicalbert_model.eval()
        # for param in self.clinicalbert_model.parameters():
        #     param.requires_grad = False

    def _train_epoch(self, epoch):

        self.logger.info('[{}/{}] Start to train in the training set.'.format(epoch, self.epochs))
        train_loss = 0
        self.model.train()
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):

            images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), \
                                                 reports_masks.to(self.device)
            output = self.model(images, reports_ids, mode='train')

            if epoch > 1000:
                print("not supposed to get here")
            ## bert_loss:
                # self.model.tokenizer.ClinicalBERT_model.to("cuda")
                # E = self.model.tokenizer.ClinicalBERT_model.embeddings.word_embeddings.weight.to("cuda")
                # p_i_j = torch.exp(output[:,:,1:])
                # gt_embedding_batch = E[reports_ids.to("cuda").int()][:,1:] ########## ask Elad
                # weighted_embedding_batch = torch.matmul(p_i_j.to("cuda"), E.to("cuda")) ## maybe change here to deal with batch
                # self.model.eval()
                # # Compute BERTScore:
                # # Get contextualized embeddings from RoBERTa
                # reports_masks = reports_masks[:,1:]
                # # pred_outputs = self.roberta_model(inputs_embeds=weighted_embedding_batch,
                # #                                   attention_mask=reports_masks)
                # # gt_outputs = self.roberta_model(inputs_embeds=gt_embedding_batch, attention_mask=reports_masks)
                #
                # pred_outputs = self.model.tokenizer.ClinicalBERT_model(inputs_embeds=weighted_embedding_batch,
                #                                   attention_mask=reports_masks)
                # gt_outputs = self.model.tokenizer.ClinicalBERT_model(inputs_embeds=gt_embedding_batch, attention_mask=reports_masks)
                #
                #
                # # Get last hidden states [batch_size, seq_len, hidden_dim]
                # pred_embeds = pred_outputs.last_hidden_state
                # gt_embeds = gt_outputs.last_hidden_state
                # #loss = bertscore_ce_loss(pred_embeds, gt_embeds, reports_masks, reports_masks, eps=1e-8)
                # # loss_bert = bertscore_loss(pred_embeds, gt_embeds, reports_masks, reports_masks, eps=1e-8)
                # # 7) Final BERTScore loss => we want to maximize F_bert => minimize negative
                # loss = fbert_score(pred_embeds, gt_embeds)
            else:
                loss = self.criterion(output, reports_ids, reports_masks)

            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.args.log_period == 0:
                self.logger.info('[{}/{}] Step: {}/{}, Training Loss: {:.5f}.'
                                 .format(epoch, self.epochs, batch_idx, len(self.train_dataloader),
                                         train_loss / (batch_idx + 1)))

        log = {'train_loss': train_loss / len(self.train_dataloader)}

        self.logger.info('[{}/{}] Start to evaluate in the validation set.'.format(epoch, self.epochs))
        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)

                output, _ = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)

            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            wandb.log({'val_' + k: v for k, v in val_met.items()})
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        self.logger.info('[{}/{}] Start to evaluate in the test set.'.format(epoch, self.epochs))
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output, _ = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            wandb.log({'test_' + k: v for k, v in test_met.items()})
            log.update(**{'test_' + k: v for k, v in test_met.items()})

        self.lr_scheduler.step()

        return log,test_res,test_gts


#
#
# # OUR CODE
# import logging
# import os
# from abc import abstractmethod
# import json
# from bert_score import score
# from transformers import RobertaModel, RobertaTokenizer
# import numpy as np
# from bert_score import score
# import torch.nn.functional as F
# import torch
# from numpy import inf
# import wandb
# #wandb.login()
# from transformers import AutoModel
# import torch
# import torch.nn.functional as F
#
#
# def bertscore_ce_loss(
#         pred_embeds: torch.Tensor,  # [B, L_pred, hidden_dim]
#         ref_embeds: torch.Tensor,  # [B, L_ref,  hidden_dim]
#         pred_mask: torch.Tensor,  # [B, L_pred], 1=valid token, 0=pad
#         ref_mask: torch.Tensor,  # [B, L_ref], 1=valid token, 0=pad
#         alpha: float = 5.0,
#         eps: float = 1e-9
# ) -> torch.Tensor:
#     """
#     Differentiable alignment approach for a "BERTScore-like" measure:
#       1) L2-normalize predicted & reference embeddings => cos sim
#       2) sim => [B, L_pred, L_ref]
#       3) Row-wise softmax (pred2ref) and column-wise softmax (ref2pred)
#       4) Cross Entropy:
#          CE = - sum_{i,j} pred2ref[i,j] * log(ref2pred[i,j])
#       5) Return the mean cross-entropy across the batch.
#
#     This encourages row-dist. & column-dist. to match, so predicted tokens
#     and reference tokens form a consistent alignment distribution.
#
#     It's not the original BERTScore formula (no P/R/F1), but is fully differentiable
#     and can be more stable for training than a 'max' or 'logsumexp' approach.
#
#     Args:
#         pred_embeds: [B, Lp, H], normalized or not, up to you
#         ref_embeds : [B, Lr, H]
#         pred_mask  : [B, Lp] => which pred tokens are valid
#         ref_mask   : [B, Lr] => which ref tokens are valid
#         alpha      : temperature for the softmax
#         eps        : small constant for log
#
#     Returns:
#         Scalar loss. Lower => row & column alignment distributions agree.
#     """
#
#     # 1) Normalize => cos similarities
#     pred_norm = F.normalize(pred_embeds, p=2, dim=-1)  # [B, Lp, H]
#     ref_norm = F.normalize(ref_embeds, p=2, dim=-1)  # [B, Lr, H]
#
#     # 2) Pairwise cos => sim [B, Lp, Lr]
#     sim = torch.bmm(pred_norm, ref_norm.transpose(1, 2))
#
#     # 3) Build 3D mask => [B, Lp, Lr] to zero out invalid pairs
#     pred_mask_f = pred_mask.float()  # [B, Lp]
#     ref_mask_f = ref_mask.float()  # [B, Lr]
#     mask_3d = torch.einsum("bl,br->blr", pred_mask_f, ref_mask_f)
#     # zero out sim for padded tokens so they don't affect softmax
#     sim = sim * mask_3d
#
#     # 4) Row-wise & Column-wise softmax => distributions
#     #    shape: [B, Lp, Lr]
#     #    Add a small value for safety if sim is zeroed
#     pred2ref_dist = F.softmax(alpha * sim, dim=2)  # distribution over Lr for each pred token
#     ref2pred_dist = F.softmax(alpha * sim, dim=1)  # distribution over Lp for each ref token
#
#     # 5) Cross Entropy:
#     #    CE = - sum_{i,j} pred2ref[i,j] * log(ref2pred[i,j])
#     #    we only sum over valid pairs (mask_3d)
#
#     # avoid log(0)
#     log_ref2pred = (ref2pred_dist + eps).log()
#
#     # elementwise => pred2ref_dist * log_ref2pred => sum over i,j
#     # Then mask it:
#     ce_map = pred2ref_dist * log_ref2pred * mask_3d  # [B, Lp, Lr]
#
#     # sum over i, j => [B]
#     ce_per_batch = - ce_map.sum(dim=(1, 2))
#
#     # average over batch
#     loss = ce_per_batch.mean()
#
#     return loss
#
# def fbert_score_smooth(emb1: torch.Tensor, emb2: torch.Tensor, alpha: float = 5.0) -> float:
#     """
#     Compute a 'smooth max' F-BERT–style score between two sets of embeddings
#     using a log-sum-exp approximation.
#
#     Args:
#         emb1 (torch.Tensor): Embeddings from the first sequence [len1, hidden_dim].
#         emb2 (torch.Tensor): Embeddings from the second sequence [len2, hidden_dim].
#         alpha (float)       : Temperature for the log-sum-exp; higher alpha => more max-like.
#
#     Returns:
#         float: Negative of the F-BERT score (harmonic mean) so that minimizing
#                this value => maximizing the smooth F-BERT alignment.
#     """
#     # 1) Compute pairwise cosine similarity => shape [len1, len2]
#     sim = F.cosine_similarity(emb1.unsqueeze(1), emb2.unsqueeze(0), dim=-1)
#
#     # 2) Smooth "max" with log-sum-exp over each row => shape [len1]
#     #    row i => logsumexp(alpha * sim[i,:]) / alpha
#     #    Then average => approximate precision
#     approx_pred2ref = torch.logsumexp(alpha * sim, dim=1) / alpha
#     precision = approx_pred2ref.mean()
#
#     # 3) Similarly, smooth "max" over each column => shape [len2]
#     #    column j => logsumexp(alpha * sim[:, j]) / alpha
#     #    Then average => approximate recall
#     approx_ref2pred = torch.logsumexp(alpha * sim, dim=0) / alpha
#     recall = approx_ref2pred.mean()
#
#     # 4) Harmonic mean => F
#     f_score = 2 * precision * recall / (precision + recall + 1e-8)
#
#     # 5) Return negative => so minimizing => maximizing smooth F-BERT
#     return -f_score.item()
#
# def fbert_score(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
#     """
#     Compute an F-BERT–style score between two sets of BERT embeddings.
#
#     Args:
#         emb1 (torch.Tensor): Embeddings from the first sequence (shape: [len1, hidden_dim]).
#         emb2 (torch.Tensor): Embeddings from the second sequence (shape: [len2, hidden_dim]).
#
#     Returns:
#         float: The F-BERT score (F1-like measure of alignment between embeddings).
#     """
#     # Compute pairwise cosine similarity
#     # sim has shape [len1, len2]
#     sim = F.cosine_similarity(emb1.unsqueeze(1), emb2.unsqueeze(0), dim=-1)
#
#     # Precision: for each token in emb1, find the best-match in emb2
#     precision = sim.max(dim=1)[0].mean()
#
#     # Recall: for each token in emb2, find the best-match in emb1
#     recall = sim.max(dim=0)[0].mean()
#
#     # F-BERT score (harmonic mean of Precision and Recall)
#     f_score = 2 * precision * recall / (precision + recall + 1e-8)
#
#     return -f_score.item()
#
# def bertscore_loss(
#     pred_embeds: torch.Tensor,  # [B, L_pred, H]
#     ref_embeds: torch.Tensor,   # [B, L_ref,  H]
#     pred_mask: torch.Tensor,    # [B, L_pred], 1=valid token, 0=pad
#     ref_mask: torch.Tensor,     # [B, L_ref],  1=valid token, 0=pad
#     eps: float = 1e-8
# ) -> torch.Tensor:
#     """
#     Computes a token-level BERTScore F1 (without IDF) and returns negative of it
#     (so that minimizing this loss = maximizing BERTScore).
#
#     Steps:
#     1) L2-normalize pred_embeds, ref_embeds => dot product is cosine similarity.
#     2) pairwise_cos = bmm(pred_norm, ref_norm.transpose(1,2)) => [B, L_pred, L_ref].
#     3) Mask out invalid pairs (padding).
#     4) For each predicted token => max over reference tokens => precision-like term.
#        For each reference token => max over predicted tokens => recall-like term.
#     5) P, R => average across valid tokens.
#     6) F = 2 * P * R / (P + R)
#     7) Return loss = -mean(F) over the batch.
#
#     This logic mirrors the 'greedy_cos_idf' approach from the BERTScore repo
#     (omitting IDF).
#     """
#
#     # 1) Normalize embeddings for cosine similarity
#     pred_norm = F.normalize(pred_embeds, p=2, dim=-1)  # [B, L_pred, H]
#     ref_norm  = F.normalize(ref_embeds,  p=2, dim=-1)  # [B, L_ref,  H]
#
#     # 2) Pairwise cosine: [B, L_pred, L_ref]
#     sim = torch.bmm(pred_norm, ref_norm.transpose(1, 2))
#
#     # 3) Create 3D mask => 1 where both tokens are valid
#     # pred_mask: [B, L_pred], ref_mask: [B, L_ref]
#     # => mask_3d: [B, L_pred, L_ref]
#     mask_3d = torch.einsum('bl,br->blr', pred_mask.float(), ref_mask.float())
#     sim = sim * mask_3d
#
#     # 4) For each predicted token: max similarity => shape [B, L_pred]
#     #    For each reference token: max similarity => shape [B, L_ref]
#     max_pred2ref, _ = sim.max(dim=2)  # best alignment for each pred token
#     max_ref2pred, _ = sim.max(dim=1)  # best alignment for each ref token
#
#     # 5) Average over valid tokens => precision (P), recall (R)
#     pred_mask_f = pred_mask.float()
#     ref_mask_f  = ref_mask.float()
#
#     valid_pred_lens = pred_mask_f.sum(dim=1).clamp_min(eps)  # [B]
#     valid_ref_lens  = ref_mask_f.sum(dim=1).clamp_min(eps)   # [B]
#
#     P = (max_pred2ref * pred_mask_f).sum(dim=1) / valid_pred_lens  # [B]
#     R = (max_ref2pred * ref_mask_f).sum(dim=1) / valid_ref_lens     # [B]
#
#     # 6) F = 2PR / (P+R)
#     F_bert = 2 * P * R / (P + R + eps)  # [B]
#
#     # 7) Loss = negative average F
#     return -F_bert.mean()
#
#
#
# class BaseTrainer(object):
#     def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler):
#         self.args = args
#
#         logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                             datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
#         self.logger = logging.getLogger(__name__)
#
#         # setup GPU device if available, move model into configured device
#         self.device, device_ids = self._prepare_device(args.n_gpu)
#         self.model = model.to(self.device)
#         if len(device_ids) > 1:
#             self.model = torch.nn.DataParallel(model, device_ids=device_ids)
#
#         self.criterion = criterion
#         self.metric_ftns = metric_ftns
#         self.optimizer = optimizer
#         self.lr_scheduler = lr_scheduler
#
#         self.epochs = self.args.epochs
#         self.save_period = self.args.save_period
#
#         self.mnt_mode = args.monitor_mode
#         self.mnt_metric = 'val_' + args.monitor_metric
#         self.mnt_metric_test = 'test_' + args.monitor_metric
#         assert self.mnt_mode in ['min', 'max']
#
#         self.mnt_best = inf if self.mnt_mode == 'min' else -inf
#         self.early_stop = getattr(self.args, 'early_stop', inf)
#
#         self.start_epoch = 1
#         self.checkpoint_dir = args.save_dir
#
#         self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
#                               'test': {self.mnt_metric_test: self.mnt_best}}
#
#         if not os.path.exists(self.checkpoint_dir):
#             os.makedirs(self.checkpoint_dir)
#
#         if args.resume is not None:
#             self._resume_checkpoint(args.resume)
#
#         # start a new wandb run:
#         wandb.init(
#             # set the wandb project where this run will be logged
#             project="R2GenCMN",
#
#             # track hyperparameters and run metadata
#             config={
#                 "model": model,
#                 "criterion": criterion,
#                 "dataset": args.dataset_name,
#                 "metric_ftns": metric_ftns,
#                 "optimizer": optimizer,
#                 "scheduler": lr_scheduler,
#                 "epochs": self.args.epochs,
#             }
#         )
#
#
#     @abstractmethod
#     def _train_epoch(self, epoch):
#         raise NotImplementedError
#
#     def train(self):
#         not_improved_count = 0
#         for epoch in range(self.start_epoch, self.start_epoch+self.epochs + 1):
#             result,test_res,test_gts= self._train_epoch(epoch)
#
#             # save logged informations into log dict
#             log = {'epoch': epoch}
#             log.update(result)
#             self._record_best(log)
#
#             # print logged informations to the screen
#             for key, value in log.items():
#                 self.logger.info('\t{:15s}: {}'.format(str(key), value))
#
#             # evaluate model performance according to configured metric, save best checkpoint as model_best
#             best = False
#             if self.mnt_mode != 'off':
#                 try:
#                     # check whether model performance improved or not, according to specified metric(mnt_metric)
#                     improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
#                                (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
#                 except KeyError:
#                     self.logger.warning(
#                         "Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
#                             self.mnt_metric))
#                     self.mnt_mode = 'off'
#                     improved = False
#
#                 if improved:
#                     self.mnt_best = log[self.mnt_metric]
#                     not_improved_count = 0
#                     best = True
#                 else:
#                     not_improved_count += 1
#
#                 if not_improved_count > self.early_stop:
#                     self.logger.info("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
#                         self.early_stop))
#                     break
#             ## if best model saved, save also output and GT
#             if epoch % self.save_period == 0:
#                 self._save_checkpoint(epoch, test_res,test_gts, save_best=best)
#
#     def _record_best(self, log):
#         improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
#             self.mnt_metric]) or \
#                        (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
#         if improved_val:
#             self.best_recorder['val'].update(log)
#
#         improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
#             self.mnt_metric_test]) or \
#                         (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
#                             self.mnt_metric_test])
#         if improved_test:
#             self.best_recorder['test'].update(log)
#
#     def _print_best(self):
#         self.logger.info('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
#         for key, value in self.best_recorder['val'].items():
#             self.logger.info('\t{:15s}: {}'.format(str(key), value))
#
#         self.logger.info('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
#         for key, value in self.best_recorder['test'].items():
#             self.logger.info('\t{:15s}: {}'.format(str(key), value))
#
#     def _prepare_device(self, n_gpu_use):
#         n_gpu = torch.cuda.device_count()
#         if n_gpu_use > 0 and n_gpu == 0:
#             self.logger.warning(
#                 "Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
#             n_gpu_use = 0
#         if n_gpu_use > n_gpu:
#             self.logger.warning(
#                 "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
#                     n_gpu_use, n_gpu))
#             n_gpu_use = n_gpu
#         device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
#         list_ids = list(range(n_gpu_use))
#         return device, list_ids
#
#     def _save_checkpoint(self, epoch, test_res,test_gts, save_best=False):
#         state = {
#             'epoch': epoch,
#             'state_dict': self.model.state_dict(),
#             'optimizer': self.optimizer.state_dict(),
#             'monitor_best': self.mnt_best
#             }
#
#         ## save results to file:
#         results = {i:[test_gts[i],test_res[i]] for i in range(len(test_gts))}
#
#         filename = os.path.join(self.checkpoint_dir, f'current_checkpoint_{self.args.run_name}.pth')
#         torch.save(state, filename)
#         self.logger.info("Saving checkpoint: {} ...".format(filename))
#         current_results_path = os.path.join(self.checkpoint_dir, f'model_current_results_{self.args.run_name}.json')
#         with open(current_results_path, "w") as outfile:
#             json.dump(results, outfile)
#         if save_best:
#             best_path = os.path.join(self.checkpoint_dir, f'model_best_{self.args.run_name}.pth')
#             best_results_path = os.path.join(self.checkpoint_dir, f'model_best_results_{self.args.run_name}.json')
#             torch.save(state, best_path)
#             with open(best_results_path, "w") as outfile:
#                 json.dump(results, outfile)
#             # torch.save(results, best_results_path)
#             self.logger.info("Saving current best: model_best.pth ...")
#
#     def _resume_checkpoint(self, resume_path):
#         resume_path = str(resume_path)
#         self.logger.info("Loading checkpoint: {} ...".format(resume_path))
#         checkpoint = torch.load(resume_path)
#         self.start_epoch = checkpoint['epoch'] + 1
#         self.mnt_best = checkpoint['monitor_best']
#         self.model.load_state_dict(checkpoint['state_dict'])
#         self.optimizer.load_state_dict(checkpoint['optimizer'])
#
#         self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
#
#
# class Trainer(BaseTrainer):
#     def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader,
#                  val_dataloader, test_dataloader, tokenizer):
#         super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler)
#         self.train_dataloader = train_dataloader
#         self.val_dataloader = val_dataloader
#         self.test_dataloader = test_dataloader
#         self.tokenizer = tokenizer
#
#         # Load pre-trained RoBERTa model
#         # self.roberta_model = RobertaModel.from_pretrained('roberta-large').to(self.device)
#         # self.roberta_model.eval()  # Set to evaluation mode
#         # # Freeze RoBERTa parameters
#         # for param in self.roberta_model.parameters():
#         #     param.requires_grad = False
#
#         self.clinicalbert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(self.device)
#         self.clinicalbert_model.eval()
#         for param in self.clinicalbert_model.parameters():
#             param.requires_grad = False
#
#     def _train_epoch(self, epoch):
#
#         self.logger.info('[{}/{}] Start to train in the training set.'.format(epoch, self.epochs))
#         train_loss = 0
#         self.model.train()
#         val_gts, val_res = [], []
#         for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):
#
#             images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), \
#                                                  reports_masks.to(self.device)
#             output = self.model(images, reports_ids, mode='train') ## take last word of word dimension (history)
#
#             loss =  self.criterion(output, reports_ids, reports_masks)
#             #if epoch > 0:
#             ## bert_loss:
#                         # E = self.tokenizer.model.embeddings.word_embeddings.weight.to("cuda")
#                         # p_i_j = torch.exp(output[:,:,1:])
#                         # gt_embedding_batch = E[reports_ids.to("cuda").int()][:,1:] ########## ask Elad
#                         # weighted_embedding_batch = torch.matmul(p_i_j.to("cuda"), E.to("cuda")) ## maybe change here to deal with batch
#                         # self.model.eval()
#                         # # Compute BERTScore:
#                         # # Get contextualized embeddings from RoBERTa
#                         # reports_masks = reports_masks[:,1:]
#                         # # pred_outputs = self.roberta_model(inputs_embeds=weighted_embedding_batch,
#                         # #                                   attention_mask=reports_masks)
#                         # # gt_outputs = self.roberta_model(inputs_embeds=gt_embedding_batch, attention_mask=reports_masks)
#                         #
#                         # pred_outputs = self.clinicalbert_model(inputs_embeds=weighted_embedding_batch,
#                         #                                   attention_mask=reports_masks)
#                         # gt_outputs = self.clinicalbert_model(inputs_embeds=gt_embedding_batch, attention_mask=reports_masks)
#                         #
#                         #
#                         # # Get last hidden states [batch_size, seq_len, hidden_dim]
#                         # pred_embeds = pred_outputs.last_hidden_state
#                         # gt_embeds = gt_outputs.last_hidden_state
#                         # #loss = bertscore_ce_loss(pred_embeds, gt_embeds, reports_masks, reports_masks, eps=1e-8)
#                         # # loss_bert = bertscore_loss(pred_embeds, gt_embeds, reports_masks, reports_masks, eps=1e-8)
#                         # # 7) Final BERTScore loss => we want to maximize F_bert => minimize negative
#                         # loss_bert = fbert_score_smooth(pred_embeds, gt_embeds)
#                         # # loss = - F_bert.mean()  # scalar
#                         #
#                         #
#                         #
#                         # #else:
#                         #
#                         # loss = 0.5 * loss_ce + 0.5 * loss_bert
#             train_loss += loss.item()
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
#             if batch_idx % self.args.log_period == 0:
#                 self.logger.info('[{}/{}] Step: {}/{}, Training Loss: {:.5f}.'
#                                  .format(epoch, self.epochs, batch_idx, len(self.train_dataloader),
#                                          train_loss / (batch_idx + 1)))
#                 wandb.log({'epoch': epoch + 1, 'train_loss': train_loss / (batch_idx + 1)})
#
#         log = {'train_loss': train_loss / len(self.train_dataloader)}
#
#         self.logger.info('[{}/{}] Start to evaluate in the validation set.'.format(epoch, self.epochs))
#         self.model.eval()
#         with torch.no_grad():
#             val_gts, val_res = [], []
#             for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
#                 images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
#                     self.device), reports_masks.to(self.device)
#
#                 output, _ = self.model(images, mode='sample')
#                 reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
#                 ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
#                 val_res.extend(reports)
#                 val_gts.extend(ground_truths)
#
#             val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
#                                        {i: [re] for i, re in enumerate(val_res)})
#             wandb.log({'val_' + k: v for k, v in val_met.items()})
#             log.update(**{'val_' + k: v for k, v in val_met.items()})
#
#         self.logger.info('[{}/{}] Start to evaluate in the test set.'.format(epoch, self.epochs))
#         self.model.eval()
#         with torch.no_grad():
#             test_gts, test_res = [], []
#             for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
#                 images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
#                     self.device), reports_masks.to(self.device)
#                 output, _ = self.model(images, mode='sample')
#                 reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
#                 ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
#                 test_res.extend(reports)
#                 test_gts.extend(ground_truths)
#
#             test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
#                                         {i: [re] for i, re in enumerate(test_res)})
#             wandb.log({'test_' + k: v for k, v in test_met.items()})
#             log.update(**{'test_' + k: v for k, v in test_met.items()})
#
#         self.lr_scheduler.step()
#
#         return log,test_res,test_gts
