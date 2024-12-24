from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from bert_score import score as bert_score


def compute_scores(gts, res):
    """
    Performs the evaluation using various metrics, including BLEU, METEOR, ROUGE_L, and BERTScore.

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids and their generated captions
    :return: Evaluation scores for each metric
    """

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L")
    ]
    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score

    # Compute BERTScore
    # Extract candidates and references
    ids = sorted(res.keys())
    candidates = [res[id][0] for id in ids]    # res[id] is a list with one generated caption
    references = [gts[id] for id in ids]       # gts[id] is a list of reference captions

    # Compute BERTScore
    P, R, F1 = bert_score(candidates, references, lang="en", verbose=False)
    bert_score_avg = F1.mean().item()
    eval_res['BERT_SCORE'] = bert_score_avg

    return eval_res

def compute_mlc(gt, pred, label_set):
    res_mlc = {}
    avg_aucroc = 0
    for i, label in enumerate(label_set):
        res_mlc['AUCROC_' + label] = roc_auc_score(gt[:, i], pred[:, i])
        avg_aucroc += res_mlc['AUCROC_' + label]
    res_mlc['AVG_AUCROC'] = avg_aucroc / len(label_set)

    res_mlc['F1_MACRO'] = f1_score(gt, pred, average="macro")
    res_mlc['F1_MICRO'] = f1_score(gt, pred, average="micro")
    res_mlc['RECALL_MACRO'] = recall_score(gt, pred, average="macro")
    res_mlc['RECALL_MICRO'] = recall_score(gt, pred, average="micro")
    res_mlc['PRECISION_MACRO'] = precision_score(gt, pred, average="macro")
    res_mlc['PRECISION_MICRO'] = precision_score(gt, pred, average="micro")

    return res_mlc


class MetricWrapper(object):
    def __init__(self, label_set):
        self.label_set = label_set

    def __call__(self, gts, res, gts_mlc, res_mlc):
        eval_res = compute_scores(gts, res)
        eval_res_mlc = compute_mlc(gts_mlc, res_mlc, self.label_set)

        eval_res.update(**eval_res_mlc)
        return eval_res
