import random
import fasttext

class Population():

    def __init__(self, params, n_individuals=10, p_best=0.2, mutation=0.01, mix_rate=0.1, n_rounds=10):

        if not params:
            raise Exception

        self.params = params
        self.n_individuals = n_individuals
        self.p_best = p_best
        self.mutation = mutation
        self.mix_rate = mix_rate
        self.n_rounds = n_rounds

        self.individuals = [self.generate_random_individual() for i in range(n_individuals)]

    def generate_random_individual(self):

        ind = Individual()

        for param in self.params:
            value = random.choice(param.values)
            if param.name == 'lr':
                ind.lr = value
            elif param.name == 'epoch':
                ind.epoch = value
            elif param.name == 'min_count':
                ind.min_count = value
            elif param.name == 'word_ngrams':
                ind.word_ngrams = value

        return ind

    def get_fittest(self):
        # Computes scores and keeps the fittest
        for ind in self.individuals:
            ind.calculate_score()

        fittest = sorted(self.individuals, key=lambda x: x.score, reverse=True)


    def next_generation(self):

        fittest = self.get_fittest()

        n_kept = p_best*len(fittest)
        if n_kept < 2:
            n_kept = 2
        n_kept = int(n_kept)
        kept = self.individuals[:n_kept]

        lucky_ones = [ind for ind in fittest[n_kept:] if random.random()<mix_rate]
        kept = kept + lucky_ones

        # Procreation
        children = []
        while len(kept) + len(children) < self.n_individuals:
            i_father = random.randint(0, len(kept)-1)
            i_mother = random.randint(0, len(kept)-1)
            while i_mother == i_father:
                i_father = random.randint(0, len(kept)-1)
                i_mother = random.randint(0, len(kept)-1)
            father = kept[i_father]
            mother = kept[i_mother]

            child = Individual()
            lr, epoch, min_count, word_ngrams = 0, 0, 0, 0
            for param in self.params:
                if random.random() < 0.5:
                    gene_giver = father
                else:
                    gene_giver = mother

                if param.name == 'lr':
                    if random.random() > mutation:
                        child.lr = gene_giver.lr
                    else:
                        child.lr = random.choice(param.values)
                elif param.name == 'epoch':
                    if random.random() > mutation:
                        child.epoch = gene_giver.epoch
                    else:
                        child.epoch = random.choice(param.values)
                elif param.name == 'min_count':
                    if random.random() > mutation:
                        child.min_count = gene_giver.min_count
                    else:
                        child.min_count = random.choice(param.values)
                elif param.name == 'word_ngrams':
                    if random.random() > mutation:
                        child.word_ngrams = gene_giver.word_ngrams
                    else:
                        child.word_ngrams = random.choice(param.values)

            children.append(child)

        self.individuals = kept + children

    def lauch(self):

        for step in range(self.n_rounds):
            self.next_generation()

        fittest = self.get_fittest()
        best_individual = fittest[0]
        return best_individual


class Individual():

    def __init__(self, lr=None, epoch=None, min_count=None, word_ngrams=None, score=0):
        self.lr = lr
        self.epoch = epoch
        self.min_count = min_count
        self.word_ngrams = word_ngrams
        self.score = score
        self.model_name = 'temp_model_genetics'
        self.label = '__label__'
        self.train_file = 'train'
        self.test_file = 'test'

    def copy(self):
        return Individual(self.lr, self.epoch, self.min_count, self.word_ngrams, self.score)

    def calculate_score(self):
        classifier = fasttext.supervised(self.train_file,
                                         self.model_name,
                                         epoch=self.epoch,
                                         dim=10,
                                         word_ngrams=self.word_ngrams,
                                         lr=self.lr,
                                         min_count=self.min_count,
                                         bucket=2000000,
                                         loss='ns')
        metrics = get_metrics(classifier, self.test_file)
        recall = metrics['recall']
        self.score = recall

        # Cleaning
        classifier = None
        metrics = None


class Param():

    def __init__(self, name, values):
        if not values:
            raise Exception

        self.name = name
        self.values = sorted(list(set(values)))
