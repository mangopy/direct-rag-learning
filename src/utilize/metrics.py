from collections import Counter
import re

def rounder(num):
    return round(num, 3)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

# ==================================================================================================
# EM Score
# ==================================================================================================


def eval_em(predictions, raw):
    scores = []
    for pred, example in zip(predictions, raw):
        pred = normalize_answer(pred)
        if any([normalize_answer(ans) == pred for ans in example]): #TODO
            scores.append(1)
        else:
            scores.append(0)
    return {'EM': rounder(sum(scores) * 100 / len(scores))}


# ==================================================================================================
# Acc Score
# ==================================================================================================


def eval_acc(predictions, raw):
    scores = []
    for pred, example in zip(predictions, raw):
        pred = normalize_answer(pred)
        if any([normalize_answer(ans) in pred for ans in example]): #TODO
            scores.append(1)
        else:
            scores.append(0)
    return {'Acc': rounder(sum(scores) * 100 / len(scores))}


# ==================================================================================================
# F1 Score
# ==================================================================================================

def prec_recall_f1_score(pred_items, gold_items):
    """
    Compute precision, recall and f1 given a set of gold and prediction items.

    :param pred_items: iterable of predicted values
    :param gold_items: iterable of gold values

    :return: tuple (p, r, f1) for precision, recall, f1
    """
    common = Counter(gold_items) & Counter(pred_items)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def f1_score(guess, answers):
    if guess is None or answers is None:
        return 0
    g_tokens = normalize_answer(guess).split()
    scores = [
        prec_recall_f1_score(g_tokens, normalize_answer(a).split()) for a in answers
    ]
    return max(f1 for p, r, f1 in scores)


def eval_f1(predicts, answers):
    f1 = 0.
    for predict, answer in zip(predicts, answers):
        # answer = answer.split('\t')
        f1 += f1_score(predict, answer)
    return {'F1': rounder(f1 * 100 / len(answers))}