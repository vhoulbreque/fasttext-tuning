from genetics import *


if __name__ == '__main__':

    n_individuals = 20
    n_rounds = 6
    # n_individuals = 3
    # n_rounds = 2

    params = dict()
    params['epoch'] = [1, 5, 10, 15, 20, 25, 30, 35]
    params['lr'] = [i/10 for i in range(1, 11)]
    params['min_count'] = [1, 10, 20]
    params['word_ngrams'] = [1, 2, 3]

    experience = Experience(params, n_individuals, n_rounds=n_rounds, good_label='clickbait')

    best = experience.launch()
    best.save('model_{}'.format(best.id))
