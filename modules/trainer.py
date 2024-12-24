import logging
import os
from abc import abstractmethod
import json
from bert_score import score
from transformers import RobertaModel, RobertaTokenizer
import numpy as np
from bert_score import score

import torch
from numpy import inf
import wandb
#wandb.login()

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
        for epoch in range(self.start_epoch, self.start_epoch+self.epochs + 1):
            result,test_res,test_gts= self._train_epoch(epoch)

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
            ## if best model saved, save also output and GT
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, test_res,test_gts, save_best=best)

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
                 val_dataloader, test_dataloader, tokenizer):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.tokenizer = tokenizer

        # Load pre-trained RoBERTa model
        self.roberta_model = RobertaModel.from_pretrained('roberta-large').to(self.device)
        self.roberta_model.eval()  # Set to evaluation mode
        # Freeze RoBERTa parameters
        for param in self.roberta_model.parameters():
            param.requires_grad = False

    def _train_epoch(self, epoch):

        self.logger.info('[{}/{}] Start to train in the training set.'.format(epoch, self.epochs))
        train_loss = 0
        self.model.train()
        val_gts, val_res = [], []
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):

            images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), \
                                                 reports_masks.to(self.device)
            output = self.model(images, reports_ids, mode='train') ## take last word of word dimension (history)
            if epoch > 0:
                ## bert_loss:
                E = self.tokenizer.model.embeddings.word_embeddings.weight.to("cuda")
                p_i_j = torch.exp(output[:,:,1:])
                gt_embedding_batch = E[reports_ids.to("cuda").int()][:,1:] ########## ask Elad
                weighted_embedding_batch = torch.matmul(p_i_j.to("cuda"), E.to("cuda")) ## maybe change here to deal with batch
                self.model.eval()
                # Compute BERTScore:
                # Get contextualized embeddings from RoBERTa
                reports_masks = reports_masks[:,1:]
                pred_outputs = self.roberta_model(inputs_embeds=weighted_embedding_batch,
                                                  attention_mask=reports_masks)
                gt_outputs = self.roberta_model(inputs_embeds=gt_embedding_batch, attention_mask=reports_masks)

                # Get last hidden states [batch_size, seq_len, hidden_dim]
                pred_hidden_states = pred_outputs.last_hidden_state
                gt_hidden_states = gt_outputs.last_hidden_state

                # Apply mask to zero out padding positions [batch_size, seq_len, 1]
                reports_masks_expanded = reports_masks.unsqueeze(-1).float()
                pred_hidden_states = pred_hidden_states * reports_masks_expanded
                gt_hidden_states = gt_hidden_states * reports_masks_expanded

                # Compute mean embeddings for each sequence
                pred_sum = pred_hidden_states.sum(dim=1)  # [batch_size, hidden_dim]
                gt_sum = gt_hidden_states.sum(dim=1)  # [batch_size, hidden_dim]

                lengths = reports_masks.sum(dim=1).unsqueeze(-1)  # [batch_size, 1]

                pred_mean = pred_sum / lengths
                gt_mean = gt_sum / lengths

                # Compute cosine similarity between mean embeddings
                cos = torch.nn.CosineSimilarity(dim=-1)
                cos_sim = cos(pred_mean, gt_mean)  # [batch_size]

                # Compute loss as 1 - cosine similarity (BERT-based semantic loss)
                bert_loss = 1 - cos_sim  # [batch_size]
                loss = bert_loss.mean()  # Scalar

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
                wandb.log({'epoch': epoch + 1, 'train_loss': train_loss / (batch_idx + 1)})

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
