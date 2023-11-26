import pandas as pd
import numpy as np 
from scipy import stats
from scipy.signal import find_peaks

def features_extract (df, accx, accy, accz, activity, window_size, step_size): 
    # features based on accx, accy, accz
    x_list = []
    y_list = []
    z_list = []
    labels = []

    # overlapping windows
    for i in range(0, len(df) - window_size + 1, step_size):
        # arrays per axis
        xs = df[accx].values[i: i + window_size]
        ys = df[accy].values[i: i + window_size]
        zs = df[accz].values[i: i + window_size]

        # label with most occurrences in window
        input_array = np.array(df[activity][i: i + window_size], dtype=float)
        label = stats.mode(input_array)[0]

        x_list.append(xs)
        y_list.append(ys)
        z_list.append(zs)
        labels.append(label)

    # converting the lists to series
    x_series_td = pd.Series(x_list)
    y_series_td = pd.Series(y_list)
    z_series_td = pd.Series(z_list)

    # converting the signals from time domain to frequency domain using FFT
    fft_size = int((window_size/2)) + 1

    x_series_fft = x_series_td.apply(lambda x: np.abs(np.fft.fft(x))[1:fft_size])
    y_series_fft = y_series_td.apply(lambda x: np.abs(np.fft.fft(x))[1:fft_size])
    z_series_fft = z_series_td.apply(lambda x: np.abs(np.fft.fft(x))[1:fft_size])

    X = pd.DataFrame()
    y = np.array(labels)
    y = y.astype(int)

    for tp in ['td', 'fft']:

        for axis in ['x','y','z']:
            
            series = locals()[f'{axis}_series_{tp}']

            ################## simple statistics features ##################
            # mean
            X[f'{axis}_mean_{tp}'] = series.apply(lambda x: x.mean())
            # mean abs diff
            X[f'{axis}_meandiff_{tp}'] = series.apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
            # min
            X[f'{axis}_min_{tp}'] = series.apply(lambda x: x.min())
            # max
            X[f'{axis}_max_{tp}'] = series.apply(lambda x: x.max())     
            # max-min diff
            X[f'{axis}_minmax_{tp}'] = X[f'{axis}_max_{tp}'] - X[f'{axis}_min_{tp}']
            # median
            X[f'{axis}_median_{tp}'] = series.apply(lambda x: np.median(x))
            # median abs diff 
            X[f'{axis}_mediandiff_{tp}'] = series.apply(lambda x: np.median(np.absolute(x - np.median(x))))
            # std dev
            X[f'{axis}_std_{tp}'] = series.apply(lambda x: x.std())
            # interquartile range
            X[f'{axis}_quart_{tp}'] = series.apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))

            # indexes
            # index of min value in window
            if tp == 'td':
                X[f'{axis}_argmin_{tp}'] = series.apply(lambda x: np.argmin(x))
                # index of max value in window
                X[f'{axis}_argmax_{tp}'] = series.apply(lambda x: np.argmax(x))
            else:
                X[f'{axis}_argmin_{tp}'] = series.apply(lambda x: np.argmin(np.abs(np.fft.fft(x))[1:fft_size]))
                # index of max value in window
                X[f'{axis}_argmax_{tp}'] = series.apply(lambda x: np.argmax(np.abs(np.fft.fft(x))[1:fft_size]))
            
            # abs max-min index diff
            X[f'{axis}_minmaxarg_{tp}'] = abs(X[f'{axis}_argmax_{tp}'] - X[f'{axis}_argmin_{tp}'])
            
            # only for time domain
            if tp == 'td':                
                # negtive values count
                X[f'{axis}_negatives_{tp}'] = series.apply(lambda x: np.sum(x < 0))
                # positive values count
                X[f'{axis}_positives_{tp}'] = series.apply(lambda x: np.sum(x > 0))
            
            # values above mean
            X[f'{axis}_meanabove_{tp}'] = series.apply(lambda x: np.sum(x > x.mean()))
            # skewness
            X[f'{axis}_skewness_{tp}'] = series.apply(lambda x: stats.skew(x))
            # kurtosis
            X[f'{axis}_kurtosis_{tp}'] = series.apply(lambda x: stats.kurtosis(x))


            ################## signal based features ##################
            # count peaks in signal
            X[f'{axis}_peaks_{tp}'] = series.apply(lambda x: len(find_peaks(x)[0]))
            # power of signal: average of the squared signal
            X[f'{axis}_power_{tp}'] = series.apply(lambda x: np.mean(x**2))
        
        # over all axis
        seriesx = locals()[f'x_series_{tp}']
        seriesy = locals()[f'y_series_{tp}']
        seriesz = locals()[f'z_series_{tp}']

        # signal magnitude area
        X[f'SMA_{tp}'] = seriesx.apply(lambda x: np.mean(abs(x))) + seriesy.apply(lambda x: np.mean(abs(x))) + seriesz.apply(lambda x: np.mean(abs(x)))

        # avg resultant
        X[f'avg_result_accl_{tp}'] = [i.mean() for i in ((seriesx**2 + seriesy**2 + seriesz**2)**0.5)]

    return X, y