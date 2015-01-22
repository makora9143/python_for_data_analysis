#! /usr/bin/env python
#-*- encoding: utf-8 -*-

"""

"""

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


def slide2():
    now = datetime.now()
    print now
    print now.year, now.month, now.day
    print "##############"
    delta = datetime(2011, 1, 7) - datetime(2008, 6, 24, 8, 15)
    print delta
    print "##############"
    print delta.days
    print delta.seconds
    print "##############"
    start = datetime(2011, 1, 7)
    print start + timedelta(12)
    print start - 2 * timedelta(12)


def slide3():
    print 'datetime -> string'
    stamp = datetime(2011, 1, 3)
    print str(stamp)
    print stamp.strftime('%Y-%m-%d')

    print 'string -> datetime'
    value = '2011-01-03'
    print datetime.strptime(value, '%Y-%m-%d')

    datestrs = ['7/6/2011', '8/6/2011']
    print [datetime.strptime(x, '%m/%d/%Y') for x in datestrs]

    print 'dateutil.parser'
    from dateutil.parser import parse
    print parse('2011-01-03')
    print parse('Jan 31, 1997 10:45 PM')
    print parse('6/12/2011', dayfirst=True)

    print 'pandas'
    print pd.to_datetime(datestrs)
    idx = pd.to_datetime(datestrs + [None])
    print idx


def slide4():
    dates = [datetime(2011, 1, 2), datetime(2011, 1, 5), datetime(2011, 1, 7),
        datetime(2011, 1, 8), datetime(2011, 1, 10), datetime(2011, 1, 12)]

    print 'Series sample'
    ts = Series(np.random.randn(6), index=dates)
    print ts
    print type(ts)
    print ts.index

    print 'arithmetic operations'
    print ts + ts[::2]
    print ts.index.dtype

    stamp = ts.index[2]
    print stamp
    print 'indexing'
    print ts[stamp]
    print ts['1/10/2011']
    print ts['20110110']

    longer_ts = Series(np.random.randn(1000),
                       index=pd.date_range('1/1/2000', periods=1000))
    print 'longer timestamp'
    print longer_ts
    print longer_ts['2001']
    print longer_ts['2001-05']
    print 'indexing range'
    print ts[datetime(2011, 1, 7):]
    print ts['1/6/2011':'1/11/2011']

    print 'truncate'
    print ts.truncate(after='1/9/2011')

    dates = pd.date_range('1/1/2000', periods=100, freq='W-WED')
    long_df = DataFrame(np.random.randn(100, 4),
                        index=dates,
                        columns=['Colorado', 'Texas', 'New York', 'Ohio'])
    print long_df.ix['5-2001']

    print 'duplicate'
    dates = pd.DatetimeIndex(['1/1/2000', '1/2/2000', '1/2/2000', '1/2/2000',
                              '1/3/2000'])
    dup_ts = Series(np.arange(5), index=dates)
    print dup_ts
    print dup_ts.index.is_unique
    print dup_ts['1/3/2000']
    print dup_ts['1/2/2000']

    grouped = dup_ts.groupby(level=0)
    print grouped.mean()
    print grouped.count()


def slide6():
    dates = [datetime(2011, 1, 2), datetime(2011, 1, 5), datetime(2011, 1, 7),
        datetime(2011, 1, 8), datetime(2011, 1, 10), datetime(2011, 1, 12)]
    ts = Series(np.random.randn(6), index=dates)
    print ts
    print ts.resample('D')

    index = pd.date_range('4/1/2012', '6/1/2012')
    print index
    print 'start'
    print pd.date_range(start='4/1/2012', periods=20)
    print 'end'
    print pd.date_range(end='6/1/2012', periods=20)
    print 'business end of month'
    print pd.date_range('1/1/2000', '12/1/2000', freq='BM')
    print pd.date_range('5/2/2012 12:56:31', periods=5)
    print 'normalize'
    print pd.date_range('5/2/2012 12:56:31', periods=5, normalize=True)


def slide7():
    from pandas.tseries.offsets import Hour, Minute
    hour = Hour()
    print hour
    four_hours = Hour(4)
    print four_hours
    print pd.date_range('1/1/2000', '1/3/2000 23:59', freq='4h')

    print Hour(2) + Minute(30)
    print pd.date_range('1/1/2000', periods=10, freq='1h30min')

    ts = Series(np.random.randn(4),
                index=pd.date_range('1/1/2000', periods=4, freq='M'))
    print ts
    print ts.shift(2)
    print ts.shift(-2)
    print '2 M'
    print ts.shift(2, freq='M')
    print '3 D'
    print ts.shift(3, freq='D')
    print '1 3D'
    print ts.shift(1, freq='3D')
    print '1 90T'
    print ts.shift(1, freq='90T')

    print 'shifting dates with offsets'
    from pandas.tseries.offsets import Day, MonthEnd
    now = datetime(2011, 11, 17)
    print now + 3 * Day()
    print now + MonthEnd()
    print now + MonthEnd(2)

    offset = MonthEnd()
    print offset
    print offset.rollforward(now)
    print offset.rollback(now)

    ts = Series(np.random.randn(20),
                index=pd.date_range('1/15/2000', periods=20, freq='4d'))
    print ts.groupby(offset.rollforward).mean()


def slide8():
    import pytz
    print pytz.common_timezones[-5:]

    print 'US/Eastern'
    tz = pytz.timezone('US/Eastern')
    print tz

    rng = pd.date_range('3/9/2012 9:30', periods=6, freq='D')
    ts = Series(np.random.randn(len(rng)), index=rng)
    print ts.index.tz

    print 'date_range utc'
    print pd.date_range('3/9/2012 9:30', periods=10, freq='D', tz='UTC')

    print 'tz_localize to UTC'
    ts_utc = ts.tz_localize('UTC')
    print ts_utc
    print ts_utc.index

    print 'tz_convert to us/Eastern'
    print ts_utc.tz_convert('US/Eastern')

    print 'tz_localize to us/Eastern'
    ts_eastern = ts.tz_localize('US/Eastern')
    print ts_eastern.tz_convert('UTC')

    print 'tz_convert'
    print ts_eastern.tz_convert('Europe/Berlin')

    stamp = pd.Timestamp('2011-03-12 04:00')
    stamp_utc = stamp.tz_localize('utc')
    print 'us/eastern'
    print stamp_utc.tz_convert('US/Eastern')

    stamp_moscow = pd.Timestamp('2011-03-12 04:00', tz='Europe/Moscow')
    print 'moscow'
    print stamp_moscow

    print 'nano seconds'
    print stamp_utc.value
    print stamp_utc.tz_convert('US/Eastern').value

    from pandas.tseries.offsets import Hour
    stamp = pd.Timestamp('2012-03-12 01:30', tz='US/Eastern')
    print stamp
    print '+hour'
    print stamp + Hour()
    stamp = pd.Timestamp('2012-11-04 00:30', tz='US/Eastern')
    print 'summer time'
    print stamp + 2 * Hour()

    print 'between different time zones'
    rng = pd.date_range('3/7/2012 9:30', periods=10, freq='B')
    ts = Series(np.random.randn(len(rng)), index=rng)
    print ts
    ts1 = ts[:7].tz_localize('Europe/London')
    ts2 = ts1[2:].tz_convert('Europe/Moscow')
    result = ts1 + ts2
    print result.index


def slide9():
    p = pd.Period(2007, freq='A-DEC')
    print p
    print 'after 5 years'
    print p + 5
    print 'before 2 yeaars'
    print p - 2

    print pd.Period('2014', freq='A-DEC') - p

    rng = pd.period_range('1/1/2000', '6/30/2000', freq='M')
    print rng
    print Series(np.random.randn(6), index=rng)
    values = ['2001Q3', '2002Q2', '2003Q1']
    index = pd.PeriodIndex(values, freq='Q-DEC')
    print index

    print '2007, A-DEC'
    p = pd.Period('2007', freq='A-DEC')
    print p.asfreq('M', 'start')
    print p.asfreq('M', 'end')
    print '2007, A-JUN'
    p = pd.Period('2007', freq='A-JUN')
    print p.asfreq('M', 'start')
    print p.asfreq('M', 'end')

    p = pd.Period('2007-08', 'M')
    print p.asfreq('A-JUN')

    rng = pd.period_range('2006', '2009', freq='A-DEC')
    ts = Series(np.random.randn(len(rng)), index=rng)
    print ts


def slide10():
    print '2012Q4, Q-JAN'
    p = pd.Period('2012Q4', freq='Q-JAN')
    print p
    print '2012Q4 start'
    print p.asfreq('D', 'start')
    print '2012Q4 end'
    print p.asfreq('D', 'end')

    print p.asfreq('B', 'e')
    print '4PM on the 2nd to last business day of the quater'
    print (p.asfreq('B', 'e') - 1).asfreq('T', 's')
    p4pm = (p.asfreq('B', 'e') - 1).asfreq('T', 's') + 16 * 60
    print p4pm
    print p4pm.to_timestamp()

    print 'timeseries'
    rng = pd.period_range('2011Q3', '2012Q4', freq='Q-JAN')
    ts = Series(np.arange(len(rng)), index=rng)
    print ts
    new_rng = (rng.asfreq('B', 'e') - 1).asfreq('T', 's') + 16 * 60
    print 'new range'
    print new_rng
    ts.index = new_rng.to_timestamp()
    print ts


def slide11():
    rng = pd.date_range('1/1/2000', periods=3, freq='M')
    ts = Series(np.random.randn(3), index=rng)
    pts = ts.to_period()
    print ts
    print pts

    rng = pd.date_range('1/29/2000', periods=6, freq='D')
    ts2 = Series(np.random.randn(6), index=rng)
    print ts2.to_period('M')
    print 'convert back'
    pts = ts.to_period()
    print pts
    print pts.to_timestamp(how='end')
    print 'macrodata'
    data = pd.read_csv('../pydata-book/ch08/macrodata.csv')
    print data
    print data.year
    print data.quarter

    index = pd.PeriodIndex(year=data.year, quarter=data.quarter, freq='Q-DEC')
    print index
    data.index = index
    print data.infl


def slide12():
    rng = pd.date_range('1/1/2000', periods=100, freq='D')
    ts = Series(np.random.randn(len(rng)), index=rng)
    print 'timeseries'
    print ts
    print 'resample'
    print ts.resample('M', how='mean')


def slide13():
    rng = pd.date_range('1/1/2000', periods=12, freq='T')
    ts = Series(np.arange(12), index=rng)
    print ts
    print '5min sum closed=right'
    print ts.resample('5min', how='sum', closed='right')
    print 'label=right'
    print ts.resample('5min', how='sum', closed='left', label='right')

    print ts.resample('5min', how='sum')
    print ts.resample('5min', how='sum', loffset='-1s')

    print 'OHLC'
    print ts.resample('5min', how='ohlc')

    print 'resampling with GroupBy'
    rng = pd.date_range('1/1/2000', periods=100, freq='D')
    ts = Series(np.arange(100), index=rng)
    print ts
    print 'month'
    print ts.groupby(lambda x: x.month).mean()
    print 'weekday'
    print ts.groupby(lambda x: x.month).mean()


def slide14():
    frame = DataFrame(np.random.randn(2, 4),
                      index=pd.date_range('1/1/2000', periods=2, freq='W-WED'),
                      columns=['Colorado', 'Texas', 'New York', 'Ohio'])
    print frame[:5]

    df_daily = frame.resample('D')
    print 'daily fill_method=none'
    print df_daily
    print 'daily fill_method=ffill'
    print frame.resample('D', fill_method='ffill')
    print 'daily fill_method=ffill limit=2'
    print frame.resample('D', fill_method='ffill', limit=2)

    print frame.resample('W-THU', fill_method='ffill')

    print 'resampling with periods'
    frame = DataFrame(np.random.randn(24, 4),
                      index=pd.period_range('1-2000', '12-2001', freq='M'),
                      columns=['Colorado', 'Texas', 'New York', 'Ohio'])
    print frame[:5]

    annual_frame = frame.resample('A-DEC', how='mean')
    print annual_frame
    print 'resample Quarterly'
    print annual_frame.resample('Q-DEC', fill_method='ffill')
    print annual_frame.resample('Q-DEC',
                                fill_method='ffill',
                                convention='start')

def slide15():
    close_px_all = pd.read_csv('../pydata-book/ch09/stock_px.csv',
                               parse_dates=True,
                               index_col=0)
    close_px = close_px_all[['AAPL', 'MSFT', 'XOM']]
    close_px = close_px.resample('B', fill_method='ffill')
    print close_px
    close_px['AAPL'].plot()
    close_px.ix['2009'].plot()

    close_px['AAPL'].ix['01-2011': '03-2011'].plot()

    appl_q = close_px['AAPL'].resample('Q-DEC', fill_method='ffill')
    appl_q.ix['2009':].plot()
    plt.show()


def slide16():
    close_px_all = pd.read_csv('../pydata-book/ch09/stock_px.csv',
                               parse_dates=True,
                               index_col=0)
    close_px = close_px_all[['AAPL', 'MSFT', 'XOM']]
    close_px = close_px.resample('B', fill_method='ffill')

    close_px.AAPL.plot()
    pd.rolling_mean(close_px.AAPL, 250).plot()

    appl_std250 = pd.rolling_std(close_px.AAPL, 250, min_periods=10)
    print appl_std250[5:12]

    appl_std250.plot()

    expanding_mean = lambda x: rolling_mean(x, len(x), min_periods=1)
    pd.rolling_mean(close_px, 60).plot(logy=True)

    plt.show()


if __name__ == '__main__':
    # slide2()
    # slide3()
    # slide4()
    # slide5()
    # slide6()
    # slide7()
    # slide8()
    # slide9()
    # slide10()
    # slide11()
    # slide12()
    # slide13()
    # slide14()
    # slide15()
    slide16()

# End of Line.
