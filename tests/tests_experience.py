import sys

from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import assert_raises
from nose.tools import raises

sys.path.insert(0, '../src/')

from genetics import *


class TestExperience():

    @classmethod
    def setup_class(klass):
        pass

    @classmethod
    def teardown_class(klass):
        pass

    def setUp(self):
        """This method is run once before _each_ test method is executed"""

    def teardown(self):
        """This method is run once after _each_ test method is executed"""

    def test_init(self):

        def instantiate_none():
            return Experience(None)

        def instantiate_not_dict():
            return Experience([])

        def instantiate_empty():
            return Experience(dict())

        assert_raises(Exception, instantiate_none)
        assert_raises(Exception, instantiate_not_dict)
        assert_raises(Exception, instantiate_empty)

    def test_e2e(self):
        n_individuals = 3
        n_rounds = 2

        params = dict()
        params['lr'] = [i/10 for i in range(1, 11)]
        params['epoch'] = [1, 5, 10, 15, 20, 25, 30, 35]
        params['min_count'] = [1, 10, 20]
        params['word_ngrams'] = [1, 2, 3]

        experience = Experience(params, n_individuals, n_rounds=n_rounds)

        best = experience.launch()
        best.save('model_{}'.format(best.id))

        assert_equal(type(best), Individual)
