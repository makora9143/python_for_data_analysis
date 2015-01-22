import pandas as pd
df = pd.read_csv('../pydata-book/ch06/ex1.csv')
pd.read_table('../pydata-book/ch06/ex1.csv', sep=',')
pd.read_csv('../pydata-book/ch06/ex2.csv', header=None)

# names = ['a', 'b', 'c', 'd', 'message']
# pd.read_csv('ch06/ex2.csv', names=names, index_col='message')

## slide No.8
# list(open('ch06/ex3.txt'))
# result = pd.read_table('ch06/ex3.txt', sep='\s+')
# result

## slide No.10
# !cat ch06/ex5.csv
# result = pd.read_csv('ch06/ex5.csv')
# result

## JSON data slide No.15
# import json
# result = json.loads(obj)
# result

## HTML and XML slide No.16,17
# from lxml.html import parse
# from urllib2 import urlopen
# parsed = parse(urlopen('http://finance.yahoo.com/q/op?s=AAPL+Options'))
#doc = parsed.getroot()
# links = doc.findall('.//a')
# links[15:20]

## slide No.17
# lnk = links[28]
# lnk.get('href')
# lnk.text_content()
# urls = [lnk.get('href') for lnk in doc.findall('.//a')]
# urls[-10:]

## find table Slide no.18
# tables = doc.findall('.//table')
# calls = tables[1]
# puts = tables[2]
# rows = calls.findall('.//tr')
# def _unpack(row, kind=‘td’):
# 	elts = row.findall(‘.//%s’ % kind)
# 	return [val.text_content() for val in elts]
# _unpack(rows[0], kind='th')
# _unpack(rows[1], kind='td')


