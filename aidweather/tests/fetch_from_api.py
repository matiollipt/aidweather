from core import AidWeather


def test_fetch():
    weather = AidWeather("test", 0, 0, "2020-01-01", "2020-01-31")
    df = weather.get()
    assert len(df) == 31
    print(df.shape, df.head(), df.tail())


def test_filter():
    # fetch data
    weather = AidWeather("test", 0, 0, "2020-01-01", "2020-01-31")
    df = weather.get()
    assert len(df) == 31
    print()
    
    # filter data
    df = weather.filter(start="2020-01-01", end="2020-01-15")
    assert len(df) == 15
    print(df.shape, df.head(), df.tail())


test_fetch()
