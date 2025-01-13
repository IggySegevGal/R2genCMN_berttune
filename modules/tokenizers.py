### tokenizer for roberta-large:
# import json
# import re
# from collections import Counter
# from transformers import RobertaModel, RobertaTokenizer
#
# class Tokenizer(object):
#     def __init__(self, args):
#         self.ann_path = args.ann_path
#         self.threshold = args.threshold
#         self.dataset_name = args.dataset_name
#         self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
#         self.model = RobertaModel.from_pretrained('roberta-large')
#         if self.dataset_name == 'iu_xray':
#             self.clean_report = self.clean_report_iu_xray
#         else:
#             self.clean_report = self.clean_report_mimic_cxr
#         self.ann = json.loads(open(self.ann_path, 'r').read())
#
#     def create_vocabulary(self):
#         # This method is no longer needed with RoBERTa tokenizer
#         pass
#
#     def clean_report_iu_xray(self, report):
#         report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
#             .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
#             .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
#             .strip().lower().split('. ')
#         sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
#                                         replace('\\', '').replace("'", '').strip().lower())
#         tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
#         report = ' . '.join(tokens) + ' .'
#         return report
#
#     def clean_report_mimic_cxr(self, report):
#         report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
#             .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
#             .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
#             .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
#             .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
#             .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
#             .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
#             .strip().lower().split('. ')
#         sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
#                                         .replace('\\', '').replace("'", '').strip().lower())
#         tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
#         report = ' . '.join(tokens) + ' .'
#         return report
#
#     def get_token_by_id(self, id):
#         return self.tokenizer.convert_ids_to_tokens(id)
#
#     def get_id_by_token(self, token):
#         return self.tokenizer.convert_tokens_to_ids(token)
#
#     def get_vocab_size(self):
#         return self.tokenizer.vocab_size
#
#     def __call__(self, report):
#         cleaned_report = self.clean_report(report)
#         #tokens = self.tokenizer.tokenize(cleaned_report)
#         #ids = self.tokenizer.convert_tokens_to_ids(tokens)
#         #ids = [self.tokenizer.cls_token_id] + ids + [self.tokenizer.sep_token_id]
#         ids = self.tokenizer.encode(cleaned_report, add_special_tokens=True)
#         return ids
#
#     def decode(self, ids):
#         return self.tokenizer.decode(ids, skip_special_tokens=True)
#
#     def decode_batch(self, ids_batch):
#         return [self.decode(ids) for ids in ids_batch]


# tokenizer for clinical bert:
# import json
# import re
# from collections import Counter
# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# from transformers import AutoModel, AutoTokenizer
#
# class Tokenizer(object):
#     def __init__(self, args):
#         """
#         Minimal changes:
#           1) We use the 'emilyalsentzer/Bio_ClinicalBERT' checkpoint for both
#              the tokenizer and the model.
#           2) We store the model as 'self.model' so you can do self.model.embeddings...
#         """
#         self.ann_path = args.ann_path
#         self.threshold = args.threshold
#         self.dataset_name = args.dataset_name
#
#         if self.dataset_name == 'iu_xray':
#             self.clean_report = self.clean_report_iu_xray
#         else:
#             self.clean_report = self.clean_report_mimic_cxr
#
#         # load your annotations if needed
#         self.ann = json.loads(open(self.ann_path, 'r').read())
#
#         # create vocab with old code, if still wanted:
#         self.token2idx, self.idx2token = self.create_vocabulary()
#
#         # load the ClinicalBERT tokenizer + model inside
#         # to do self.model.embeddings.word_embeddings.weight
#         checkpoint_name = "emilyalsentzer/Bio_ClinicalBERT"
#         self.ClinicalBERT_tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)
#         self.ClinicalBERT_model = AutoModel.from_pretrained(checkpoint_name)
#         self.ClinicalBERT_model.eval()  # (optional) if you want to freeze or eval
#
#     def create_vocabulary(self):
#         """
#         Old approach for building a custom vocab from self.ann, though
#         we now rely on BERT's subword vocab.
#         If you don't need it, just return empty dicts to keep minimal changes.
#         """
#         total_tokens = []
#         for example in self.ann['train']:
#             tokens = self.clean_report(example['report']).split()
#             for token in tokens:
#                 total_tokens.append(token)
#         counter = Counter(total_tokens)
#         vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
#         vocab.sort()
#         token2idx, idx2token = {}, {}
#         for idx, token in enumerate(vocab):
#             token2idx[token] = idx + 1
#             idx2token[idx + 1] = token
#         return token2idx, idx2token
#
#     def clean_report_iu_xray(self, report):
#         """Your original IU X-Ray cleaning logic"""
#         report_cleaner = lambda t: (
#             t.replace('..', '.')
#              .replace('..', '.')
#              .replace('..', '.')
#              .replace('1. ', '')
#              .replace('. 2. ', '. ')
#              .replace('. 3. ', '. ')
#              .replace('. 4. ', '. ')
#              .replace('. 5. ', '. ')
#              .replace(' 2. ', '. ')
#              .replace(' 3. ', '. ')
#              .replace(' 4. ', '. ')
#              .replace(' 5. ', '. ')
#              .strip().lower().split('. ')
#         )
#         sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '',
#                                         t.replace('"', '')
#                                          .replace('/', '')
#                                          .replace('\\', '')
#                                          .replace("'", '')
#                                          .strip()
#                                          .lower())
#         tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
#         report = ' . '.join(tokens) + ' .'
#         return report
#
#     def clean_report_mimic_cxr(self, report):
#         """Your original MIMIC-CXR cleaning logic"""
#         report_cleaner = lambda t: (
#             t.replace('\n', ' ')
#              .replace('__', '_').replace('__', '_').replace('__', '_')
#              .replace('__', '_').replace('__', '_').replace('__', '_')
#              .replace('__', '_').replace('  ', ' ')
#              .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')
#              .replace('  ', ' ').replace('  ', ' ')
#              .replace('..', '.').replace('..', '.').replace('..', '.')
#              .replace('..', '.').replace('..', '.').replace('..', '.')
#              .replace('1. ', '').replace('. 2. ', '. ')
#              .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ')
#              .replace(' 2. ', '. ').replace(' 3. ', '. ')
#              .replace(' 4. ', '. ').replace(' 5. ', '. ')
#              .strip().lower().split('. ')
#         )
#         sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '',
#                                         t.replace('"', '')
#                                          .replace('/', '')
#                                          .replace('\\', '')
#                                          .replace("'", '')
#                                          .strip()
#                                          .lower())
#         tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
#         report = ' . '.join(tokens) + ' .'
#         return report
#
#     def get_token_by_id(self, id):
#         """
#         If you want to convert a single ID -> subword token, you can do:
#         self.ClinicalBERT_tokenizer.convert_ids_to_tokens(id)
#         """
#         return self.ClinicalBERT_tokenizer.convert_ids_to_tokens(id)
#
#     def get_id_by_token(self, token):
#         """
#         Convert a single subword token to ID
#         """
#         return self.ClinicalBERT_tokenizer.convert_tokens_to_ids(token)
#
#     def get_vocab_size(self):
#         """
#         Return the pretrained tokenizer's vocab size
#         """
#         return self.ClinicalBERT_tokenizer.vocab_size
#
#     def __call__(self, report):
#         """
#         The main 'tokenize' method. We:
#           1) clean the report
#           2) manual approach: tokenize + convert + add special tokens
#              (similar to your old approach).
#         """
#         cleaned_report = self.clean_report(report)
#         tokens = self.ClinicalBERT_tokenizer.tokenize(cleaned_report)
#         ids = self.ClinicalBERT_tokenizer.convert_tokens_to_ids(tokens)
#         # Prepend [CLS], append [SEP]
#         # (In BERT it might be [CLS], [SEP]. For clinicalBERT it's typically same IDs)
#         cls_id = self.ClinicalBERT_tokenizer.cls_token_id
#         sep_id = self.ClinicalBERT_tokenizer.sep_token_id
#         ids = [cls_id] + ids + [sep_id]
#         return ids
#
#     def decode(self, ids):
#         """
#         Reconstruct text from IDs, skipping special tokens
#         """
#         return self.ClinicalBERT_tokenizer.decode(ids, skip_special_tokens=True)
#
#     def decode_batch(self, ids_batch):
#         """
#         decode a batch of ID lists
#         """
#         return [self.decode(ids) for ids in ids_batch]


# Original tokenizer
import json
import re
from collections import Counter


class Tokenizer(object):
    def __init__(self, args):
        self.ann_path = args.ann_path
        self.threshold = args.threshold
        self.dataset_name = args.dataset_name
        if self.dataset_name == 'iu_xray':
            self.clean_report = self.clean_report_iu_xray
        else:
            self.clean_report = self.clean_report_mimic_cxr
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.token2idx, self.idx2token = self.create_vocabulary()

    def create_vocabulary(self):
        total_tokens = []

        for example in self.ann['train']:
            tokens = self.clean_report(example['report']).split()
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token
        return token2idx, idx2token

    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            else:
                break
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out