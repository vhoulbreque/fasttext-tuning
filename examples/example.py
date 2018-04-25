import os

from fasttuning.genetics import *


if __name__ == '__main__':

    HOME = os.path.expanduser("~")

    params = dict()
    params['epoch'] = [1, 5, 10, 15, 20, 25, 30, 35]
    params['lr'] = [i/10 for i in range(1, 11)]
    params['min_count'] = [1, 10, 20]
    params['word_ngrams'] = [1, 2, 3]

    n_individuals = 5
    n_rounds = 2
    good_label = 'clickbait'
    train_file = os.path.join(HOME, 'Desktop/data_train/train')
    test_file = os.path.join(HOME, 'Desktop/data_train/test')
    model_name = 'model_temp'

    print('Train file : ', train_file)
    print('Test file : ', test_file)

    experience = Experience(params,
                            n_individuals,
                            n_rounds=n_rounds,
                            good_label=good_label,
                            train_file=train_file,
                            test_file=test_file)

    best = experience.launch()
    best.save(model_name)
