#!/usr/bin/env python

"calibrate a classifier's predictions using isotonic regression"

import numpy as np
import matplotlib.pyplot as plt

from sklearn.isotonic import IsotonicRegression as IR
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score as AUC

from log_loss import log_loss
from get_diagram_data import get_diagram_data
from load_data_adult import y, p

###

# train/test split (in half)

train_end = y.shape[0] / 2
test_start = train_end + 1

y_train = y[0:train_end]
y_test =y[test_start:]

p_train = p[0:train_end]
p_test =p[test_start:]

###

ir = IR( out_of_bounds = 'clip' )	# out_of_bounds param needs scikit-learn >= 0.15
ir.fit( p_train, y_train )
p_calibrated = ir.transform( p_test )

p_calibrated[np.isnan( p_calibrated )] = 0

###

acc = accuracy_score( y_test, np.round( p_test ))
acc_calibrated = accuracy_score( y_test, np.round( p_calibrated ))

auc = AUC( y_test, p_test )
auc_calibrated = AUC( y_test, p_calibrated )

ll = log_loss( y_test, p_test )
ll_calibrated = log_loss( y_test, p_calibrated )

print "accuracy - before/after:", acc, "/", acc_calibrated
print "AUC - before/after:     ", auc, "/", auc_calibrated
print "log loss - before/after:", ll, "/", ll_calibrated

"""
accuracy - before/after: 0.847788697789 / 0.845945945946
AUC - before/after:      0.878139845077 / 0.877184085166
log loss - before/after: 0.630525772871 / 0.592161024832
"""

###

print "creating diagrams..."

n_bins = 10

# uncalibrated
	
mean_predicted_values, true_fractions = get_diagram_data( y_test, p_test, n_bins )	
plt.plot( mean_predicted_values, true_fractions )

# calibrated

mean_predicted_values, true_fractions = get_diagram_data( y_test, p_calibrated, n_bins )	
plt.plot( mean_predicted_values, true_fractions, 'green' )

# perfect calibration line
plt.plot( np.linspace( 0, 1 ), np.linspace( 0, 1 ), 'gray' )

plt.show()


