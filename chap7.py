#! /usr/bin/env python
#-*- encoding: utf-8 -*-

"""

"""

import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import matplotlib.pylab as plt


MACRODATAPATH = '../pydata-book/ch07/macrodata.csv'
MOVIELENSPATH = '../pydata-book/ch02/movielens/movies.dat'
FOODJSONPATH = '../pydata-book/ch07/foods-2011-10-03.json'


def slide_3():
    print "#######Many-to-One#######"
    df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                     'data1': range(7)})
    df2 = DataFrame({'key': ['a', 'b', 'd'],
                     'data2': range(3)})
    print "***df1***"
    print df1
    print "***df2***"
    print df2

    print "***pd.merge df1 and df2***"
    print pd.merge(df1, df2)
    print pd.merge(df1, df2, on='key')

    df3 = DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                     'data1': range(7)})
    df4 = DataFrame({'rkey': ['a', 'b', 'd'],
                     'data2': range(3)})
    print "***pd.merge df3 and df4***"
    print pd.merge(df3, df4, left_on='lkey', right_on='rkey')

    print "***pd.merge outer join***"
    print pd.merge(df1, df2, how='outer')

    print "#######Many-to-Many#######"
    df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
                     'data1': range(6)})
    df2 = DataFrame({'key': ['a', 'b', 'a', 'b', 'd'],
                     'data2': range(5)})
    print "***df1***"
    print df1
    print "***df2***"
    print df2
    print "***pd.merge left join***"
    print pd.merge(df1, df2, on='key', how='left')

    print "#######Multi Keys#######"
    left = DataFrame({'key1': ['foo', 'foo', 'bar'],
                      'key2': ['one', 'two', 'one'],
                      'lval': [1, 2, 3]})
    right = DataFrame({'key1': ['foo', 'foo', 'bar', 'bar'],
                       'key2': ['one', 'one', 'one', 'two'],
                       'rval': [4, 5, 6, 7]})
    print "***left***"
    print df1
    print "***right***"
    print df2
    print "***pd.merge outer join***"
    print pd.merge(left, right, on=['key1', 'key2'], how='outer')
    print "***overlapping column***"
    print pd.merge(left, right, on='key1')
    print pd.merge(left, right, on='key1', suffixes=('_left', '_right'))


def slide_5():
    left1 = DataFrame({'key': ['a', 'b', 'a', 'a', 'b', 'c'],
                       'value': range(6)})
    right1 = DataFrame({'group_val': [3.5, 7]}, index=['a', 'b'])

    print '***left1***'
    print left1
    print '***right1***'
    print right1

    print pd.merge(left1, right1, left_on='key', right_index=True)
    print pd.merge(left1, right1, left_on='key', right_index=True, how='outer')

    lefth = DataFrame({'key1': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
                       'key2': [2000, 2001, 2002, 2001, 2002],
                       'data': np.arange(5.)})
    righth = DataFrame(np.arange(12).reshape((6, 2)),
                       index=[['Nevada',
                               'Nevada',
                               'Ohio',
                               'Ohio',
                               'Ohio',
                               'Ohio'],
                                [2001, 2000, 2000, 2000, 2001, 2002]],
                       columns=['event1', 'event2'])
    print '***lefth***'
    print lefth
    print '***righth***'
    print righth
    print "***merge lefth and merge by inner***"
    print pd.merge(lefth, righth, left_on=['key1', 'key2'], right_index=True)
    print "***merge lefth and merge by outer***"
    print pd.merge(lefth, righth,
                   left_on=['key1', 'key2'],
                   right_index=True,
                   how='outer')

    left2 = DataFrame([[1., 2.], [3., 4.], [5., 6.]],
                      index=['a', 'c', 'e'],
                      columns=['Ohio', 'Nevada'])
    right2 = DataFrame([[7., 8.], [9., 10.], [11., 12.], [13, 14]],
                       index=['b', 'c', 'd', 'e'],
                       columns=['Missouri', 'Alabama'])
    print '***left2***'
    print left2
    print '***right2***'
    print right2
    print '***merge left2 and right2***'
    print pd.merge(left2, right2,
                   how='outer',
                   left_index=True,
                   right_index=True)
    print '***join method***'
    print left2.join(right2, how='outer')
    print left1.join(right1, on='key')

    another = DataFrame([[7., 8.], [9., 10.], [11., 12.], [16., 17.]],
                        index=['a', 'c', 'e', 'f'],
                        columns=['New York', 'Oregon'])
    print '***another***'
    print left2.join([right2, another])
    print '***another, outer***'
    print left2.join([right2, another], how='outer')


def slide_6():
    arr = np.arange(12).reshape((3, 4))
    print arr
    print '***numpy.concatenate***'
    print np.concatenate([arr, arr], axis=1)
    s1 = Series([0, 1], index=['a', 'b'])
    s2 = Series([2, 3, 4], index=['c', 'd', 'e'])
    s3 = Series([5, 6], index=['f', 'g'])
    print "***s1***"
    print s1
    print "***s2***"
    print s2
    print "***s3***"
    print s3
    print '***pandas.concat # no index overlap***'
    print pd.concat([s1, s2, s3])
    print pd.concat([s1, s2, s3], axis=1)

    s4 = pd.concat([s1 * 5, s3])
    print "***s4***"
    print s4
    print '***concat s1 and s4 by axis=1***'
    print pd.concat([s1, s4], axis=1)
    print pd.concat([s1, s4], axis=1, join='inner')

    print pd.concat([s1, s4], axis=1, join_axes=[['a', 'c', 'b', 'e']])

    result = pd.concat([s1, s1, s3], keys=['one', 'two', 'three'])
    print '***result***'
    print pd.concat([s1, s1, s3])
    print result

    print pd.concat([s1, s2, s3], axis=1, keys=['one', 'two', 'three'])

    df1 = DataFrame(np.arange(6).reshape(3, 2),
                    index=['a', 'b', 'c'],
                    columns=['one', 'two'])
    df2 = DataFrame(5 + np.arange(4).reshape(2, 2),
                    index=['a', 'c'],
                    columns=['three', 'four'])
    print '***df1***'
    print df1
    print '***df2***'
    print df2
    print pd.concat([df1, df2], axis=1, keys=['level1', 'level2'])
    print pd.concat({'level1': df1, 'level2': df2}, axis=1)

    print pd.concat([df1, df2],
                    axis=1,
                    keys=['level1', 'level2'],
                    names=['upper', 'lower'])

    df1 = DataFrame(np.random.randn(3, 4), columns=['a', 'b', 'c', 'd'])
    df2 = DataFrame(np.random.randn(2, 3), columns=['b', 'd', 'a'])
    print '***df1***'
    print df1
    print '***df2***'
    print df2
    print pd.concat([df1, df2], ignore_index=True)


def slide_7():
    a = Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan],
               index=['f', 'e', 'd', 'c', 'b', 'a'])
    b = Series(np.arange(len(a), dtype=np.float64),
               index=['f', 'e', 'd', 'c', 'b', 'a'])
    print '***a***'
    print a
    print '***b***'
    print b
    b[-1] = np.nan
    print '***a***'
    print a
    print '***b***'
    print b
    print np.where(pd.isnull(a), b, a)

    print '#####combine_first#####'
    print '***b[:-2]***'
    print b[:-2]
    print '***a[2:]***'
    print a[2:]
    print 'b[:-2].combine_first(a[2:])'
    print b[:-2].combine_first(a[2:])

    df1 = DataFrame({'a': [1., np.nan, 5., np.nan],
                     'b': [np.nan, 2., np.nan, 6.],
                     'c': range(2, 18, 4)})
    df2 = DataFrame({'a': [5., 4., np.nan, 3., 7.],
                     'b': [np.nan, 3., 4., 6., 8.]})
    print '***df1***'
    print df1
    print '***df2***'
    print df2
    print df1.combine_first(df2)


def slide_8():
    data = DataFrame(np.arange(6).reshape((2, 3)),
                     index=pd.Index(['Ohio', 'Colorado'], name='state'),
                     columns=pd.Index(['one', 'two', 'three'], name='number'))
    print data
    result = data.stack()
    print '***stack()***'
    print result
    print '***unstack()***'
    print result.unstack()

    print '***unstack(0)***'
    print result.unstack(0)

    print "***unstack('state')***"
    print result.unstack('state')

    s1 = Series([0, 1, 2, 3], index=['a', 'b', 'c', 'd'])
    s2 = Series([4, 5, 6], index=['c', 'd', 'e'])
    data2 = pd.concat([s1, s2], keys=['one', 'two'])
    print '***unstack***'
    print data2.unstack()
    print '***unstack->stack***'
    print data2.unstack().stack()
    print '***unstack->stack(dropna)***'
    print data2.unstack().stack(dropna=False)

    df = DataFrame({'left': result, 'right': result + 5},
                   columns=pd.Index(['left', 'right'],
                   name='side'))
    print 'df'
    print df

    print "unstack('state')"
    print df.unstack('state')
    print "unstack('state').stack('side')"
    print df.unstack('state').stack('side')


def slide_9():
    data = pd.read_csv(MACRODATAPATH)
    periods = pd.PeriodIndex(year=data.year, quarter=data.quarter, name='date')
    data = DataFrame(data.to_records(),
                     columns=pd.Index(['realgdp', 'infl', 'unemp'],
                                      name='item'),
                     index=periods.to_timestamp('D', 'end'))

    ldata = data.stack().reset_index().rename(columns={0: 'value'})
    wdata = ldata.pivot('date', 'item', 'value')
    print ldata[:10]
    pivoted = ldata.pivot('date', 'item', 'value')
    print pivoted.head()

    ldata['value2'] = np.random.randn(len(ldata))
    print ldata[:10]

    pivoted = ldata.pivot('date', 'item')
    print pivoted[:5]
    print pivoted['value'][:5]

    unstacked = ldata.set_index(['date', 'item']).unstack('item')
    print unstacked[:7]


def slide_10():
    data = DataFrame({'k1': ['one'] * 3 + ['two'] * 4,
                      'k2': [1, 1, 2, 3, 3, 4, 4]})
    print data
    print data.duplicated()
    print data.duplicated('k1')
    print data.drop_duplicates()

    data['v1'] = range(7)
    print data
    print data.drop_duplicates(['k1'])
    print data.drop_duplicates(['k1', 'k2'], take_last=True)


def slide_11():
    data = DataFrame({'food': ['bacon', 'pulled pork', 'bacon', 'Pastrami',
                               'corned beef', 'Bacon', 'pastrami', 'honey ham',
                               'nova lox'],
                      'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
    print data

    meat_to_animal = {
        'bacon': 'pig',
        'pulled pork': 'pig',
        'pastrami': 'cow',
        'corned beef': 'cow',
        'honey ham': 'pig',
        'nova lox': 'salmon',
    }

    data['animal'] = data['food'].map(str.lower).map(meat_to_animal)
    print data['food']
    print data

    print data['food'].map(lambda x: meat_to_animal[x.lower()])


def slide_12():
    data = Series([1., -999., 2., -999., -1000., 3.])
    print data

    print data.replace(-999, np.nan)
    print data.replace([-999, -1000], np.nan)
    print data.replace([-999, -1000], [np.nan, 0])

    print data.replace({-999: np.nan, -1000: 0})


def slide_13():
    data = DataFrame(np.arange(12).reshape((3, 4)),
                     index=['Ohio', 'Colorado', 'New York'],
                     columns=['one', 'two', 'three', 'four'])
    print data.index.map(str.upper)

    data.index = data.index.map(str.upper)
    print data

    print data.rename(index=str.title, columns=str.upper)

    print data.rename(index={'OHIO': 'INDIANA'},
                      columns={'three': 'peekaboo'})

    _ = data.rename(index={'OHIO': 'INDIANA'}, inplace=True)
    print data


def slide_14():
    ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
    bins = [18, 25, 35, 60, 100]

    cats = pd.cut(ages, bins)
    print cats

    # labels じゃなくて codes を使え
    # print cats.labels
    print cats.codes
    # print cats.levels
    # levels じゃなくて categories を使え
    print cats.categories
    print pd.value_counts(cats)

    print pd.cut(ages, [18, 26, 36, 61, 100], right=False)

    group_names = ['Youth', 'YoungAdultl', 'MiddleAged', 'Senior']
    print pd.cut(ages, bins, labels=group_names)

    data = np.random.rand(20)
    print data
    print pd.cut(data, 3, precision=2)

    data = np.random.randn(1000)
    cats = pd.qcut(data, 3)
    print cats
    print pd.value_counts(cats)
    print pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.])


def slide_15():
    np.random.seed(12345)
    data = DataFrame(np.random.randn(1000, 4))
    print data.describe()

    col = data[3]
    print col[np.abs(col) > 3]

    print data[(np.abs(data) > 3).any(1)]
    data[np.abs(data) > 3] = np.sign(data) * 3
    print data.describe()


def slide_16():
    df = DataFrame(np.arange(5 * 4).reshape(5, 4))
    sampler = np.random.permutation(5)
    print sampler
    print df
    print df.take(sampler)

    print df.take(np.random.permutation(len(df))[:3])

    bag = np.array([5, 7, -1, 6, 4])
    sampler = np.random.randint(0, len(bag), size=10)
    print sampler
    draws = bag.take(sampler)
    print draws


def slide_17():
    df = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
                    'data1': range(6)})
    print pd.get_dummies(df['key'])

    dummies = pd.get_dummies(df['key'], prefix='key')
    print dummies
    df_with_dummy = df[['data1']].join(dummies)
    print df_with_dummy

    mnames = ['movie_id', 'title', 'genres']
    movies = pd.read_table(MOVIELENSPATH,
                           sep='::',
                           header=None,
                           engine='python',
                           names=mnames)
    print movies[:10]

    genre_iter = (set(x.split('|')) for x in movies.genres)
    genres = sorted(set.union(*genre_iter))
    print genres
    dummies = DataFrame(np.zeros((len(movies), len(genres))), columns=genres)

    for i, gen in enumerate(movies.genres):
        dummies.ix[i, gen.split('|')] = 1

    movies_windic = movies.join(dummies.add_prefix('Genre_'))
    print movies_windic.ix[0]

    values = np.random.rand(10)
    print values
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1]

    print pd.get_dummies(pd.cut(values, bins))


def slide_18():
    val = 'a, b,  guido'
    print val.split(',')
    pieces = [x.strip() for x in val.split(',')]
    print pieces

    first, second, third = pieces

    print first + '::' + second + '::' + third

    print '::'.join(pieces)

    print "'guido' in val is " + 'guido' in val
    print "val.index(','): %d" % val.index(',')
    print "val.find(':'): %d" % val.find(':')

    print "val.count(','): %d" % val.count(',')

    print val.replace(',', '::')
    print val.replace(',', '')


def slide_19():
    import re
    text = "foo    bar\t baz  \tqux"
    print text
    print "1つ以上の空白と消す"
    print re.split('\s+', text)

    regex = re.compile('\s+')
    print "コンパイルしてから消す"
    print regex.split(text)
    print regex.findall(text)

    text = """Dave dave@google.com Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com
"""
    pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'
    regex = re.compile(pattern, flags=re.IGNORECASE)
    print text
    print regex.findall(text)

    m = regex.search(text)
    print text[m.start():m.end()]
    print regex.match(text)

    print regex.sub('REDACTED', text)

    pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
    regex = re.compile(pattern, flags=re.IGNORECASE)

    m = regex.match('wesm@bright.net')
    print m.groups()

    print regex.findall(text)

    print regex.sub(r'Username: \1, Domain: \2, Suffix: \3', text)

    regex = re.compile(r"""
        (?P<username>[A-Z0-9._%+-]+)
        @
        (?P<domain>[A-Z0-9.-]+)
        \.
        (?P<suffix>[A-Z]{2,4})""", flags=re.IGNORECASE | re.VERBOSE)

    m = regex.match('wesm@bright.net')

    print m.groupdict()


def slide_20():
    import re
    data = {'Dave': 'dave@google.com',
            'Steve': 'steve@gmail.com',
            'Rob': 'rob@gmail.com',
            'Wes': np.nan}
    data = Series(data)
    print data
    print data.isnull()
    print data.str.contains('gmail')
    pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
    print data.str.findall(pattern, flags=re.IGNORECASE)

    matches = data.str.match(pattern, flags=re.IGNORECASE)
    print matches
    print matches.str.get(1)
    print matches.str[0]
    print data
    print data.str[:5]


def slide_21():
    import json
    db = json.load(open(FOODJSONPATH))
    print len(db)

    print db[0].keys()
    print db[0]['nutrients'][0]

    nutrients = DataFrame(db[0]['nutrients'])
    print nutrients[:7]

    info_keys = ['description', 'group', 'id', 'manufacturer']
    info = DataFrame(db, columns=info_keys)
    print info[:5]

    print pd.value_counts(info.group)[:10]

    print "今から全部のnutrientsを扱うよ"
    nutrients = []

    for rec in db:
        fnuts = DataFrame(rec['nutrients'])
        fnuts['id'] = rec['id']
        nutrients.append(fnuts)

    nutrients = pd.concat(nutrients, ignore_index=True)
    print "なんか重複多い"
    print nutrients.duplicated().sum()
    nutrients = nutrients.drop_duplicates()

    print "infoとnutrients両方にdescriptionとgroupがあるから変えよう"
    col_mapping = {'description': 'food', 'group': 'fgroup'}
    info = info.rename(columns=col_mapping, copy=False)

    col_mapping = {'description': 'nutrient', 'group': 'nutgroup'}
    nutrients = nutrients.rename(columns=col_mapping, copy=False)

    ndata = pd.merge(nutrients, info, on='id', how='outer')
    print ndata.ix[30000]

    result = ndata.groupby(['nutrient', 'fgroup'])['value'].quantile(0.5)
    result['Zinc, Zn'].order().plot(kind='barh')
    plt.show()

    by_nutrient = ndata.groupby(['nutgroup', 'nutrient'])
    get_maximum = lambda x: x.xs(x.value.idxmax())
    get_minimum = lambda x: x.xs(x.value.idxmin())

    max_foods = by_nutrient.apply(get_maximum)[['value', 'food']]

    max_foods.food = max_foods.food.str[:50]

    print max_foods.ix['Amino Acids']['food']


if __name__ == '__main__':
    # slide_3()
    # slide_5()
    # slide_6()
    # slide_7()
    # slide_8()
    # slide_9()
    # slide_10()
    # slide_11()
    # slide_12()
    # slide_13()
    # slide_14()
    # slide_15()
    # slide_16()
    # slide_17()
    # slide_18()
    # slide_19()
    # slide_20()
    slide_21()

# End of Line.
