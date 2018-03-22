from genetics import *


if __name__ == '__main__':

    n_individuals = 15
    n_rounds = 5

    lr = Param('lr', [i/10 for i in range(1, 11)])
    epoch = Param('epoch', [1, 5, 10, 15, 20, 25, 30, 35])
    min_count = Param('min_count', [1, 10, 20])
    word_ngrams = Param('word_ngrams', [1, 2, 3])

    params = [lr, epoch, min_count, word_ngrams]

    experience = Experience(params, n_individuals, n_rounds=n_rounds)

    best = experience.launch()

    print(best)
