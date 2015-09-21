import cufflinks as cf
import pandas as pd

import unittest


class TestIPlot(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(dict(x=[1, 2, 3], y=[4, 2, 1], c=[3, 1, 5]))

    def _iplot(self, df, **kwargs):
        return df.iplot(asFigure=True, **kwargs)

    @unittest.skip("no asFigure method")
    def test_scatter_matrix(self):
        self.df.scatter_matrix(asFigure=True)

    @unittest.skip("example from docs, but this doesnt work")
    def test_irregular_subplots(self):
        df = cf.datagen.bubble(10, 50, mode='stocks')
        figs = cf.figures(df, [
            dict(kind='histogram', keys='x', color='blue'),
            dict(kind='scatter', mode='markers', x='x', y='y', size=5),
            dict(kind='scatter', mode='markers', x='x', y='y',
                 size=5, color='teal')])
        figs.append(cf.datagen.lines(1).figure(bestfit=True, colors=['blue'],
                                               bestfit_colors=['pink']))
        base_layout = cf.tools.get_base_layout(figs)
        sp = cf.subplots(figs, shape=(3, 2), base_layout=base_layout,
                         vertical_spacing=.15, horizontal_spacing=.03,
                         specs=[[{'rowspan': 2}, {}], [None, {}],
                                [{'colspan': 2}, None]],
                         subplot_titles=['Histogram', 'Scatter 1',
                                         'Scatter 2', 'Bestfit Line'])
        sp['layout'].update(showlegend=False)

        cf.iplot(sp, asFigure=True)


def bar_input_argument_tests():
    options = {
        'kind': ['bar', 'barh'],
        'barmode': ['stack', 'overlay', 'group'],
        'bargap': [0.1],
        'orientation': ['h', 'v']
    }

    def bar_test(self, **kwargs):
        self._iplot(self.df, **kwargs)

    _generate_tests(TestIPlot, bar_test, 'bar', options)


def bar_row_input_argument_tests():
    options = {
        'kind': ['bar', 'barh'],
        'barmode': ['stack', 'overlay', 'group'],
        'sortbars': [True, False],
        'bargap': [0.1],
        'bargroupgap': [0.2]
    }

    def bar_row_test(self, **kwargs):
        self._iplot(self.df.ix[1], **kwargs)

    _generate_tests(TestIPlot, bar_row_test, 'bar_row', options)


def histogram_input_argument_tests():
    options = {
        'barmode': ['stack'],
        'bins': [20],
        'orientation': ['h', 'v'],
        'histnorm': ['', 'percent', 'propbability',
                     'density', 'propbability density'],
        'histfunc': ['count', 'sum', 'avg', 'min', 'max'],
        'subplots': [True]
    }

    def histogram_test(self, **kwargs):
        self._iplot(self.df, kind='histogram', **kwargs)

    _generate_tests(TestIPlot, histogram_test, 'histogram', options)


def box_input_argument_tests():
    options = {
        'orientation': ['h', 'v']
    }

    def box_test(self, **kwargs):
        self._iplot(self.df, kind='box', **kwargs)

    _generate_tests(TestIPlot, box_test, 'box', options)


def area_plot_input_argument_tests():
    options = {
        'fill': [True],
        'opacity': [1],
        'kind': ['area'],
        'width': [5]
    }

    def area_test(self, **kwargs):
        self._iplot(self.df, **kwargs)

    _generate_tests(TestIPlot, area_test, 'area', options)


def scatter_plot_input_argument_tests():
    options = {
        'x': ['x'],
        'y': ['y'],
        'text': ['x'],
        'mode': ['markers', 'lines', 'lines+markers'],
        'symbol': ['dot'],
        'size': [10],
        'width': [5],
        'bestfit': [
            True,
            # ['x'],
            # ['x', 'y']
        ]
    }

    def scatter_test(self, **kwargs):
        self._iplot(self.df, **kwargs)

    _generate_tests(TestIPlot, scatter_test, 'scatter', options)


def bubble_chart_argument_tests():
    options = {
        'x': ['x'], 'y': ['y'], 'size': ['c']
    }

    def bubble_test(self, **kwargs):
        self._iplot(self.df, **kwargs)

    _generate_tests(TestIPlot, bubble_test, 'bubble', options)


def subplot_input_argument_tests():
    options = {
        'shape': [(3, 1)],
        'shared_xaxes': [True],
        'vertical_spacing': [0.02],
        'fill': [True],
        'subplot_titles': [True],
        'legend': [False]
    }

    def subplot_test(self, **kwargs):
        self._iplot(self.df, subplots=True, **kwargs)

    _generate_tests(TestIPlot, subplot_test, 'subplots', options)


def shape_input_argument_tests():
    df = cf.datagen.lines(3, columns=['a', 'b', 'c'])
    options = {
        'hline': [
            [2],
            [2, 4],
            [dict(y=-1, color='blue', width=3),
             dict(y=1, color='pink', dash='dash')]],
        'vline': [
            [2],
            [2, 4],
            [dict(y=-1, color='blue', width=3),
             dict(y=1, color='pink', dash='dash')]],
        'hspan': [
            [(-1, 1), (2, 5)],
            {'x0': '2015-02-15', 'x1': '2015-03-15',
             'color': 'teal', 'fill': True, 'opacity': .4}
        ],
        'vspan': [
            [(-1, 1), (2, 5)],
            {'x0': '2015-02-15', 'x1': '2015-03-15',
             'color': 'teal', 'fill': True, 'opacity': .4}
        ]
    }

    def shape_tests(self, **kwargs):
        self._iplot(df, **kwargs)

    _generate_tests(TestIPlot, shape_tests, 'shape', options)


def universal_argument_tests():
    # test all permutations of options that are shared across all chart types
    options = {
        'kind': ['scatter', 'bar', 'box', 'spread', 'ratio',
                 'surface', 'histogram', 'bubble'],
        # 'bubble3d', 'scatter3d'], # should this work?
        # it doesn't work without specifying 'x', 'y', 'z'
        'title': ['my title'],
        'xTitle': ['my xaxis title'],
        'yTitle': ['my yaxis title'],
        'theme': cf.getThemes(),
        'colors': ['blue', ['red', 'white', 'blue'],
                   {'x': 'red', 'y': 'white', 'c': 'blue'}],
        'annotations': [{4: 'my annotation'}],
        'gridcolor': ['grey'],
        'zerolinecolor': ['blue'],
        'margin': [(3, 4, 3, 1), {'l': 3, 'r': 5, 't': 1, 'b': 0}],
        'secondary_y': ['y', ['y', 'c']],
        'colors': [('blue', 'red', 'orange'),
                   {'x': 'blue', 'y': 'red', 'c': 'green'}],
        'subplots': [True]
    }

    def iplot_tests(self, **kwargs):
        self._iplot(self.df, **kwargs)

    _generate_tests(TestIPlot, iplot_tests, 'universal', options)
    def scatter3d_tests(self, **kwargs):
        self._iplot(self.df, kind='scatter3d', x='x', y='y', z='c', **kwargs)

    def bubble3d_tests(self, **kwargs):
        self._iplot(self.df, kind='scatter3d', x='x', y='y',
                    z='c', size='c', **kwargs)

    def heatmap_tests(self, **kwargs):
        self._iplot(self.df, kind='heatmap')

    # _generate_tests(TestIPlot, iplot_tests, 'universal', options) # too many!
    _generate_tests(TestIPlot, scatter3d_tests, 'scatter3d', {})
    _generate_tests(TestIPlot, bubble3d_tests, 'bubble3d', {})
    _generate_tests(TestIPlot, heatmap_tests, 'heatmap', {})


def pie_argument_tests():
    options = {
        'sort': [True, False],
        'pull': [0.5],
        'hole': [0.4],
        'textposition': ['outside', 'inner']
    }

    def pie_tests(self, **kwargs):
        self._iplot(self.df, kind='pie', **kwargs)

    _generate_tests(TestIPlot, pie_tests, 'pie', options)


def errorbar_argument_tests():
    options = {
        'error_trace': ['x'],
        'error_color': ['blue'],
        'error_thickness': [5],
        'error_width': [3],
        'error_opacity': [0.5],
        'kind': ['bar', 'barh', 'scatter']
        # 'scatter3d'] # see comment above
        # - this doesn't work without specifying 'x', 'y', 'z'
    }

    def errorbar_tests(self, **kwargs):
        self._iplot(self.df, **kwargs)

    _generate_tests(TestIPlot, errorbar_tests, 'errorbar', options)


def spread_argument_tests():
    pass


def ratio_argument_tests():
    pass


def heatmap_argument_tests():
    pass


def surface_argument_tests():
    pass


def bubble3d_argument_tests():
    pass


def scatter3d_argument_tests():
    pass


# test generators


def _generate_tests(test_class, test_func, test_name, options):
    from itertools import chain, combinations, product

    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r)
                                   for r in range(len(s) + 1))

    key_value_tuple = {}
    for option, values in options.iteritems():
        key_value_tuple[option] = [(option, i) for i in values]

    for option_groups in powerset(key_value_tuple.values()):
        for input_kwargs in product(*option_groups):
            kwargs = {i[0]: i[1] for i in input_kwargs}
            setattr(
                test_class,
                'test_{}_{}'.format(test_name, '__'.join([
                    '_'.join([str(s) for s in i])
                    for i in kwargs.items()])),
                _generate_test(test_func, **kwargs))


def _generate_test(test_func, **kwargs):
    def test(self):
        test_func(self, **kwargs)

    return test


bar_input_argument_tests()
bar_row_input_argument_tests()
histogram_input_argument_tests()
box_input_argument_tests()
area_plot_input_argument_tests()
scatter_plot_input_argument_tests()
bubble_chart_argument_tests()
subplot_input_argument_tests()
shape_input_argument_tests()
universal_argument_tests()
pie_argument_tests()
errorbar_argument_tests()

spread_argument_tests()
ratio_argument_tests()
heatmap_argument_tests()
surface_argument_tests()
bubble3d_argument_tests()
scatter3d_argument_tests()

if __name__ == '__main__':
    unittest.main()
