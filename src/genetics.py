import random
import fasttext
import uuid
import time

from utils import get_metrics


class Experience():

    def __init__(self, params, n_individuals, p_best=0.5, mutation=0.05,
                 mix_rate=0.1, n_rounds=10, n_tests=1, good_label='clickbait',
                 train_file='train', test_file='test'):

        if params is None:
            raise Exception('Params is None and should be a non-empty dict')
        if type(params) != dict:
            raise Exception(
                'Params is not a dict but a {}'.format(type(params)))
        if len([f for f in params]) == 0:
            raise Exception('Params is an empty dict')

        self.population = Population(params,
                                     n_individuals=n_individuals,
                                     n_rounds=n_rounds,
                                     good_label=good_label)
        self.verbose = True
        self.current_epoch = 1
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
            time_step_beginning = time.time()
            self.population.next_generation(step+1)
            time_step_end = time.time()

            if self.verbose:
                print(self)
                print('Elapsed time : {}'.format(
                    time_step_end-time_step_beginning))
            self.current_epoch += 1

        return self.population.individuals[0]

    def __repr__(self):

        template_s = '{0:<10}\t'
        template_s += ''.join(['{%s:>5}\t' % (i+1)
                               for i in range(len(self.population.individuals))])
        template_s += '\n'

        r_string = '\n{}\nEpoch {}/{}\n{}\n'.format('#'*50, self.current_epoch,
                                                    self.n_rounds, '#'*50)
        r_string += template_s.format(*(['id'] + [str(i.id)[:5]
                                                  for i in self.population.individuals]))

        fields = sorted([k for k in self.params])
        for f in fields:
            r_string += template_s.format(
                *map(str, [f] + [i.params[f] for i in self.population.individuals]))

        r_string += template_s.format(*(['gen'] + [str(i.generation)
                                                   for i in self.population.individuals]))
        r_string += template_s.format(*(['score'] + [str(i.score)[:5]
                                                     for i in self.population.individuals]))
        r_string += template_s.format(*(['time_tr'] + [str(i.training_time)[:5]
                                                       for i in self.population.individuals]))
        return r_string


class Population():

    """
    Represents a population of individuals procreating and being left out if
    there are not fit enough
    """

    def __init__(self, params, n_individuals=10, p_best=0.5, mutation=0.05,
                 mix_rate=0.1, n_rounds=10, n_tests=1, good_label='clickbait',
                 train_file='train', test_file='test'):

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
        self.verbose = True
        self.good_label = good_label
        self.train_file = train_file
        self.test_file = test_file
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

        ind = Individual(dict(), good_label=self.good_label,
                         train_file=self.train_file, test_file=self.test_file)
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

    def next_generation(self, current_epoch):
        """
        Computes the next generation of individuals by mixing 2 individuals
        from the previous generation
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

            child = Individual(dict(), generation=current_epoch)
            child_params = dict()

            # Choose the 2 parents
            while i_mother == i_father:
                i_father = random.randint(0, len(kept)-1)
                i_mother = random.randint(0, len(kept)-1)
            father = kept[i_father]
            mother = kept[i_mother]

            # Crossover
            fields = sorted([f for f in self.params])
            crossover_index = random.randint(1, len(fields)-2)
            fields_father = fields[:crossover_index]
            fields_mother = fields[crossover_index:]
            for f in fields_father:
                child_params[f] = father.params[f]
            for f in fields_mother:
                child_params[f] = mother.params[f]

            # Mutation
            for f in fields:
                if random.random() < self.mutation:
                    child_params[f] = random.choice(self.params[f])

            child.params = child_params
            child.good_label = self.good_label
            children.append(child)

        self.individuals = kept + children

        # Sort the individuals to have the fittest first
        self.sort_individuals()


class Individual():

    """
    An individual is composed of attributes (genes) of certain values
    (alleles).
    """

    def __init__(self, params, score=0, generation=0, good_label='clickbait',
                 train_file='train', test_file='test'):
        self.params = params
        self.score = score
        self.model_name = 'temp_model_genetics'
        self.label = '__label__'
        self.train_file = train_file
        self.test_file = test_file
        self.id = uuid.uuid4()
        self.verbose = True
        self.training_time = 0
        self.generation = generation
        self.good_label = good_label

    def calculate_score(self, n_tests):
        """
        The score of the individual denotes of its fitness.
        """

        beg = time.time()
        total_recall = 0
        for i in range(n_tests):
            classifier = self.create_classifier()
            metrics = get_metrics(classifier, self.test_file, self.good_label)
            total_recall += metrics['recall']
        end = time.time()

        self.score = total_recall/n_tests
        self.training_time = (end-beg)/n_tests

        # Cleaning
        classifier = None
        metrics = None

    def create_classifier(self):
        return fasttext.supervised(self.train_file,
                                   self.model_name,
                                   epoch=self.params['epoch'],
                                   dim=10,
                                   word_ngrams=self.params['word_ngrams'],
                                   lr=self.params['lr'],
                                   min_count=self.params['min_count'],
                                   bucket=2000000,
                                   loss='ns')

    def copy(self):
        return Individual(self.params, self.score, self.good_label)

    def save(self, name=None):
        if name:
            self.model_name = name
        classifier = self.create_classifier()
        metrics = get_metrics(classifier, self.test_file, self.good_label)
        self.score = metrics['recall']

        if self.verbose:
            print(self)
            print('Saved at {}'.format(self.model_name))

    def __repr__(self):
        return ', '.join(map(str, [self.params, self.score, self.generation]))
