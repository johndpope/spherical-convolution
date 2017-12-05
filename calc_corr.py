import pickle
import numpy as np
import numpy.ma as ma
import scipy.stats

values = pickle.load(open('data/pred_values.p', 'rb'))


m = np.array([values['pred_wt'], values['pred_mutant']])
x = ma.log(m)
x.filled(0)

delta_pred = (x[0]-np.array(values['freq_wt'])) - (x[1]-np.array(values['freq_mutant']))

assert(len(delta_pred) == len(values['ddg']))

pcorr, pvalue  = scipy.stats.pearsonr(delta_pred, values['ddg'])
print("correlation: ", pcorr)
