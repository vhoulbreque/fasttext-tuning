def get_metrics(model, test_file):
    """
    Get the metrics, accuracy and recall of the model
    """

    with open(test_file, 'r') as f:
        test = [l.rstrip() for l in f]
    classes = sorted(list(set([l.split()[0][9:] for l in test])))
    n_classes = len(classes)

    confusion_matrix = [
        [0 for _ in range(n_classes)] for _ in range(n_classes)]
    examples = [[[] for _ in range(n_classes)]
                for _ in range(n_classes)]
    for example in test:
        example = example.split()
        label = example[0][9:]
        abstract = ' '.join(example[1:])

        preds = model.predict_proba([abstract], k=n_classes)[0]
        pred, proba = preds[0]

        confusion_matrix[classes.index(
            label)][classes.index(pred)] += 1

        p = dict()
        for el in preds:
            p[el[0]] = el[1]
        e = {'abstract': abstract, 'preds': p, 'true_label': label}
        examples[classes.index(label)][classes.index(pred)].append(e)

    index_brevet = classes.index('brevet')
    accuracy = confusion_matrix[index_brevet][index_brevet] / \
        sum([confusion_matrix[i][index_brevet]
             for i in range(n_classes)])
    recall = confusion_matrix[index_brevet][index_brevet] / \
        sum([confusion_matrix[index_brevet][i]
             for i in range(n_classes)])

    metrics = {'accuracy': accuracy,
               'recall': recall,
               'confusion_matrix': confusion_matrix,
               'classes': classes,
               'examples': examples
               }

    return metrics
