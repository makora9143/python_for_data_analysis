# 1.usa.gov data from bit.ly

import json
from collections import defaultdict
from pandas import DataFrame, Series
import pandas as pd
from pylab import *
import matplotlib.pyplot as plt
import numpy as np



# path = 'ch02/usagov_bitly_data2012-03-16-1331923249.txt'
# print open(path).readline()

# convert a JSON string into a Python dictionary object
path = 'ch02/usagov_bitly_data2012-03-16-1331923249.txt'
records = [json.loads(line) for line in open(path)]
# print records[0]

# access individual values within records
# print records [0]['tz'] 

# counting time zone in pure python
# print time_zones = [rec['tz'] for rec in records] #NG
time_zones = [rec['tz'] for rec in records if 'tz' in rec]
# print time_zones[:10]

# produce counts by time zone(slide No.7,8)
'''
def get_counts(sequence): 
	counts = {}
	for x in sequence: 
		if x in counts:
			counts[x] += 1 
		else:
			counts[x] = 1
	return counts

counts = get_counts(time_zones)
print counts['America/New_York']
print len(time_zones)
'''

# count the top 10 time zones (slide No.8)
'''
def top_counts(count_dict, n=10):
	value_key_pairs = [(count, tz) for tz, count in count_dict.items()] 
	value_key_pairs.sort()
	return value_key_pairs[-n:]
print top_counts(counts)
'''
# use Library(slide No.8)
'''
from collections import Counter
counts = Counter(time_zones)
print counts.most_common(10)
'''

# count time zones with pandas (slide No.9,10)

frame = DataFrame(records)
#print frame
#print frame['tz'][:10]

# use matplotlib(slide No.10)
# preparing
'''
tz_counts = frame['tz'].value_counts()
print tz_counts[:10]
'''

# use fillna method(slide No.11)
'''
clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz == ''] = 'Unknown'
tz_counts = clean_tz.value_counts()
print tz_counts[:10]
'''

# Making a horizontal bar plot (slide No.11)
'''
tz_counts[:10].plot(kind='barh', rot=0)
plt.show()
'''

# decompose the top time zones into Windows and non- Windows users (slide No.12)
'''
cframe = frame[frame.a.notnull()]
operating_system = np.where(cframe['a'].str.contains('Windows'),'Windows', 'Not Windows')
print operating_system[:5]
by_tz_os = cframe.groupby(['tz', operating_system])
agg_counts = by_tz_os.size().unstack().fillna(0)
print agg_counts[:10]
'''

# 2.1.2 Movielens 
# read movielens data (slide No.14,15)
'''
unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table('ch02/movielens/users.dat', sep='::', header=None, names=unames)
rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('ch02/movielens/ratings.dat', sep='::', header=None, names=rnames)
mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('ch02/movielens/movies.dat', sep='::', header=None, names=mnames)
# print users[:5]
# print ratings[:5]
# print movies[:5]
# print ratings


# merge (slide No.15)

data = pd.merge(pd.merge(ratings, users), movies)
# print data 


# use pivot_table method (slide No.17)

#no.1
mean_ratings = data.pivot_table('rating', rows='title', cols='gender', aggfunc='mean')
#print mean_ratings[:5]


#no.2
ratings_by_title = data.groupby('title').size()
# print ratings_by_title[:10]

# no.3
active_titles = ratings_by_title.index[ratings_by_title >= 250]
# print active_titles


# 2.2.1
# Measuring rating disagreement
mean_ratings = mean_ratings.ix[active_titles]
# print mean_ratings
mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']
sorted_by_diff = mean_ratings.sort_index(by='diff')
#print sorted_by_diff[:15]

# Standard deviation of rating grouped by title (slide No.19)
rating_std_by_title = data.groupby('title')['rating'].std()

# Filter down to active_titles
rating_std_by_title = rating_std_by_title.ix[active_titles]
print rating_std_by_title.order(ascending=False)[:10]
'''

# US Baby Names 1880-2010 (slide No.20)
'''
names1880 = pd.read_csv('ch02/names/yob1880.txt', names=['name', 'sex', 'births'])
print names1880
print names1880.groupby('sex').births.sum()
'''

# (slide No.21 to 23_2)

# 2010 is the last available year right now
'''
years = range(1880, 2011)

pieces = []
columns = ['name', 'sex', 'births']

for year in years:
 	path = 'ch02/names/yob%d.txt' % year
	frame = pd.read_csv(path, names=columns)

	frame['year'] = year 
	pieces.append(frame)


# Concatenate everything into a single DataFrame 
names = pd.concat(pieces, ignore_index=True)
print names
'''

# (slide No.22)
'''
total_births = names.pivot_table('births', rows='year',cols='sex', aggfunc=sum)
print total_births.tail()
total_births.plot(title='Total births by sex and year')
plt.show()
'''
#(slide No.23)
'''
def add_prop(group):
	# Integer division floors
	births = group.births.astype(float)

	group['prop'] = births / births.sum()
	return group
names = names.groupby(['year', 'sex']).apply(add_prop)	
print names


np.allclose(names.groupby(['year', 'sex']).prop.sum(), 1)

def get_top1000(group):
	return group.sort_index(by='births', ascending=False)[:1000]
grouped = names.groupby(['year', 'sex']) 
top1000 = grouped.apply(get_top1000)

pieces = []
for year, group in names.groupby(['year', 'sex']):
	pieces.append(group.sort_index(by='births', ascending=False)[:1000]) 
	top1000 = pd.concat(pieces, ignore_index=True)
print top1000
'''

#(slide No.23_2)
'''
# Analyzing Naming Trends
boys = top1000[top1000.sex == 'M']
girls = top1000[top1000.sex == 'F']
print total_births


subset = total_births[['John', 'Harry', 'Mary', 'Marilyn']]
subset.plot(subplots=True, figsize=(12, 10), grid=False, title="Number of births per year")
plt.show()

'''
# Measuring the increase in naming diversity
'''
table = top1000.pivot_table('prop', rows='year',cols='sex', aggfunc=sum)
table.plot(title='Sum of table1000.prop by year and sex',yticks=np.linspace(0, 1.2, 13), xticks=range(1880, 2020, 10))
plt.show()

df = boys[boys.year == 2010]
print df

prop_cumsum = df.sort_index(by='prop', ascending=False).prop.cumsum()
print prop_cumsum[:10]
print prop_cumsum.searchsorted(0.5)

df = boys[boys.year == 1900]
in1900 = df.sort_index(by='prop', ascending=False).prop.cumsum()
print in1900.searchsorted(0.5) + 1

def get_quantile_count(group, q=0.5):
	group = group.sort_index(by='prop', ascending=False) 
	return group.prop.cumsum().searchsorted(q) + 1

diversity = top1000.groupby(['year', 'sex']).apply(get_quantile_count) 
diversity = diversity.unstack('sex')

print diversity.head()
diversity.plot(title="Number of popular names in top 50%")
plt.show()
'''
# extract last letter from name column 
'''
get_last_letter = lambda x: x[-1]
last_letters = names.name.map(get_last_letter) 
last_letters.name = 'last_letter'
table = names.pivot_table('births', rows=last_letters, cols=['sex', 'year'], aggfunc=sum)

subtable = table.reindex(columns=[1910, 1960, 2010], level='year')
print subtable.head()
print subtable.sum()

letter_prop = subtable / subtable.sum().astype(float)
fig, axes = plt.subplots(2, 1, figsize=(10, 8)) 
letter_prop['M'].plot(kind='bar', rot=0, ax=axes[0], title='Male') 
letter_prop['F'].plot(kind='bar', rot=0, ax=axes[1], title='Female',legend=False)

letter_prop = table / table.sum().astype(float)
dny_ts = letter_prop.ix[['d', 'n', 'y'], 'M'].T
print dny_ts.head()
dny_ts.plot()
plt.show()
'''
# Boy names that became girl names (and vice versa)(page.42)
'''
all_names = top1000.name.unique()
mask = np.array(['lesl' in x.lower() for x in all_names])
lesley_like = all_names[mask]
print lesley_like

filtered = top1000[top1000.name.isin(lesley_like)]
print filtered.groupby('name').births.sum()
table = filtered.pivot_table('births', rows='year',cols='sex', aggfunc='sum')
table = table.div(table.sum(1), axis=0)
print table.tail()

#
'''

