import random
import fasttext
import uuid


class Experience():

    def __init__(self, params, n_individuals, p_best=0.2, mutation=0.1, mix_rate=0.1, n_rounds=10, n_tests=1):
        self.population = Population(
            params, n_individuals=n_individuals, n_rounds=n_rounds)
        self.verbose = True
        self.n_epoch = 1
        self.params = params
        self.n_individuals = n_individuals
        self.p_best = p_best
        self.mutation = mutation
        self.mix_rate = mix_rate
        self.n_rounds = n_rounds
        self.n_tests = n_tests

    def launch(self):
        """
        Does the necessary rounds and keeps the best individual at the end
        of the process.
        """

        template_s = '{0:<17} : {1}'
        if self.verbose:
            print(template_s.format('n_individuals', self.n_individuals))
            print(template_s.format('p_best', self.p_best))
            print(template_s.format('mutation', self.mutation))
            print(template_s.format('n_rounds', self.n_rounds))
            print(template_s.format('n_tests', self.n_tests))

        self.population.init_population()
        self.population.sort_individuals()

        for step in range(self.n_rounds):
            self.population.next_generation()
            if self.verbose:
                print(self)
            self.n_epoch += 1

        return self.population.individuals[0]

    def __repr__(self):
        r_string = '\n' + '#'*50 + '\n'
        r_string += 'Epoch #{}/{}'.format(self.n_epoch, self.n_rounds) + '\n'
        r_string += '#'*50 + '\n'

        template_s = ''.join(
            ['{%s:<5}\t' % i for i in range(len(self.population.individuals))])
        s = template_s.format(*[str(i.id)[:5]
                                for i in self.population.individuals])
        r_string += s + '\n'
        fields = sorted([k for k in self.params])
        for f in fields:
            s = template_s.format(
                *map(str, [i.params[f] for i in self.population.individuals]))
            r_string += s + '\n'

        s = template_s.format(*[str(i.score)[:5]
                                for i in self.population.individuals])
        r_string += s + '\n'
        return r_string


class Population():

    """
    Represents a population of individuals procreating and being left out if
    there are not fit enough
    """

    def __init__(self, params, n_individuals=10, p_best=0.2, mutation=0.1, mix_rate=0.1, n_rounds=10, n_tests=1):

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
        self.verbose = True
        self.individuals = []

    def init_population(self):
        """
        Inits the population with individuals with random attributes
        """
        self.individuals = [self.generate_random_individual()
                            for i in range(self.n_individuals)]

    def generate_random_individual(self):
        """
        Generates an individual with random attributes
        """

        ind = Individual(dict())
        params = dict()

        fields = [f for f in self.params]
        for f in fields:
            value = random.choice(self.params[f])
            params[f] = value

        ind.params = params

        return ind

    def sort_individuals(self):
        """
        Computes scores and keeps the fittest
        """
        for ind in self.individuals:
            ind.calculate_score(self.n_tests)

        self.individuals = sorted(
            self.individuals, key=lambda x: x.score, reverse=True)

    def next_generation(self):
        """
        Computes the next generation of individuals by mixing 2 individuals from
        the previous generation
        """

        n_kept = self.p_best*len(self.individuals)
        if n_kept < 2:
            n_kept = 2
        n_kept = int(n_kept)
        kept = self.individuals[:n_kept]

        # To add some variety, some unfit individuals are kept too
        lucky_ones = [ind for ind in self.individuals[n_kept:]
                      if random.random() < self.mix_rate]
        kept = kept + lucky_ones

        # Procreation
        children = []
        i_father = i_mother = 0
        while len(kept) + len(children) < self.n_individuals:

            # Choose the 2 parents
            while i_mother == i_father:
                i_father = random.randint(0, len(kept)-1)
                i_mother = random.randint(0, len(kept)-1)
            father = kept[i_father]
            mother = kept[i_mother]

            child = Individual(dict())
            child_params = dict()
            fields = [f for f in self.params]
            for f in fields:
                if random.random() > 0.5:
                    gene_giver = father
                else:
                    gene_giver = mother

                if random.random() > self.mutation:
                    child_params[f] = gene_giver.params[f]
                else:
                    child_params[f] = random.choice(self.params[f])

            child.params = child_params
            children.append(child)

        self.individuals = kept + children

        # Sort the individuals to have the fittest first
        self.sort_individuals()

    def launch(self):
        """
        Does the necessary rounds and keeps the best individual at the end
        of the process.
        """

        if self.verbose:
            print('n_individuals : ', self.n_individuals)
            print('p_best : ', self.p_best)
            print('mutation : ', self.mutation)
            print('n_rounds : ', self.n_rounds)
            print('n_tests : ', self.n_tests)

        for step in range(self.n_rounds):
            self.next_generation()
            self.n_epoch += 1

        self.sort_individuals()
        return self.individuals[0]


class Individual():

    """
    An individual is composed of attributes (genes) of certain values (alleles).
    """

    def __init__(self, params, score=0):
        self.params = params
        self.score = score
        self.model_name = 'temp_model_genetics'
        self.label = '__label__'
        self.train_file = 'train'
        self.test_file = 'test'
        self.id = uuid.uuid4()

    def calculate_score(self, n_tests):
        """
        The score of the individual denotes of its fitness.
        """

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

        total_recall = 0
        for i in range(n_tests):
            classifier = fasttext.supervised(self.train_file,
                                             self.model_name,
                                             epoch=self.params['epoch'],
                                             dim=10,
                                             word_ngrams=self.params['word_ngrams'],
                                             lr=self.params['lr'],
                                             min_count=self.params['min_count'],
                                             bucket=2000000,
                                             loss='ns')
            metrics = get_metrics(classifier, self.test_file)
            total_recall += metrics['recall']

        self.score = total_recall/n_tests

        # Cleaning
        classifier = None
        metrics = None

    def copy(self):
        return Individual(self.params, self.score)

    def __repr__(self):
        return ', '.join(map(str, [self.params, self.score]))
