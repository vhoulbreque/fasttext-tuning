import random
import fasttext
from tqdm import tqdm


class Population():

    def __init__(self, params, n_individuals=10, p_best=0.2, mutation=0.01, mix_rate=0.1, n_rounds=10, n_tests=1):

        if not params:
            raise Exception
        if n_tests <= 0 or type(n_tests) != int:
            raise Exception

        self.params = params
        self.n_individuals = n_individuals
        self.p_best = p_best
        self.mutation = mutation
        self.mix_rate = mix_rate
        self.n_rounds = n_rounds
        self.n_tests = n_tests
        self.n_epoch = 1

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
        return fittest


    def next_generation(self):

        fittest = self.get_fittest()
        self.individuals = fittest

        self.pprint()

        n_kept = self.p_best*len(self.individuals)
        if n_kept < 2:
            n_kept = 2
        n_kept = int(n_kept)
        kept = self.individuals[:n_kept]

        lucky_ones = [ind for ind in fittest[n_kept:] if random.random()<self.mix_rate]
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
                    if random.random() > self.mutation:
                        child.lr = gene_giver.lr
                    else:
                        child.lr = random.choice(param.values)
                elif param.name == 'epoch':
                    if random.random() > self.mutation:
                        child.epoch = gene_giver.epoch
                    else:
                        child.epoch = random.choice(param.values)
                elif param.name == 'min_count':
                    if random.random() > self.mutation:
                        child.min_count = gene_giver.min_count
                    else:
                        child.min_count = random.choice(param.values)
                elif param.name == 'word_ngrams':
                    if random.random() > self.mutation:
                        child.word_ngrams = gene_giver.word_ngrams
                    else:
                        child.word_ngrams = random.choice(param.values)

            children.append(child)

        self.individuals = kept + children

    def launch(self):

        for step in tqdm(range(self.n_rounds), desc=self.n_epoch):
            self.next_generation()
            self.n_epoch += 1

        fittest = self.get_fittest()
        best_individual = fittest[0]
        return best_individual

    def pprint(self):
        print('\n' + '#'*50)
        print('Epoch # {}'.format(self.n_epoch))
        print('#'*50)
        for param in params:
            s = '{0:<8}\t{1:<8}\t{2:<8}\t{3:<8}\t{4:<8}\t'
            if param.name == 'lr':
                s = s.format(*map(str, [i.lr for i in self.individuals]))
            elif param.name == 'epoch':
                s = s.format(*map(str, [i.epoch for i in self.individuals]))
            elif param.name == 'min_count':
                s = s.format(*map(str, [i.min_count for i in self.individuals]))
            elif param.name == 'word_ngrams':
                s = s.format(*map(str, [i.word_ngrams for i in self.individuals]))
            print(s)
        s = '{0:<8}\t{1:<8}\t{2:<8}\t{3:<8}\t{4:<8}\t'
        s = s.format(*[str(i.score)[:5] for i in self.individuals])
        print(s)

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

        def get_metrics(model, test_file):

            with open(test_file, 'r') as f:
                test = [l.rstrip() for l in f]
            classes = sorted(list(set([l.split()[0][9:] for l in test])))
            n_classes = len(classes)

            confusion_matrix = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
            examples = [[[] for _ in range(n_classes)] for _ in range(n_classes)]
            for example in test:
                example = example.split()
                label = example[0][9:]
                abstract = ' '.join(example[1:])

                preds = model.predict_proba([abstract], k=n_classes)[0]
                pred, proba = preds[0]

                confusion_matrix[classes.index(label)][classes.index(pred)] += 1

                p = dict()
                for el in preds:
                    p[el[0]] = el[1]
                e = {'abstract': abstract, 'preds': p, 'true_label': label}
                examples[classes.index(label)][classes.index(pred)].append(e)

            index_brevet = classes.index('brevet')
            accuracy = confusion_matrix[index_brevet][index_brevet]/sum([confusion_matrix[i][index_brevet] for i in range(n_classes)])
            recall = confusion_matrix[index_brevet][index_brevet]/sum([confusion_matrix[index_brevet][i] for i in range(n_classes)])

            metrics = {'accuracy': accuracy,
                       'recall': recall,
                       'confusion_matrix': confusion_matrix,
                       'classes': classes,
                       'examples': examples
                      }

            return metrics

        total_recall = 0
        for i in range(n_tests):
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
            total_recall += recall

        self.score = total_recall/self.n_tests

        # Cleaning
        classifier = None
        metrics = None

    def __repr__(self):
        return ', '.join(map(str, [self.lr, self.epoch, self.min_count, self.word_ngrams, self.score]))


class Param():

    def __init__(self, name, values):
        if not values:
            raise Exception

        self.name = name
        self.values = sorted(list(set(values)))


if __name__ == '__main__':

    n_individuals = 5
    n_rounds = 3

    lr = Param('lr', [0.1, 0.2, 0.5])
    epoch = Param('epoch', [1, 5, 10])
    min_count = Param('min_count', [1, 10, 20])
    word_ngrams = Param('word_ngrams', [1, 2])

    params = [lr, epoch, min_count, word_ngrams]

    population = Population(params, n_individuals=n_individuals, n_rounds=n_rounds)

    best = population.launch()

    print(best)
