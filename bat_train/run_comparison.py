import numpy as np
import matplotlib.pyplot as plt
import os
import evaluate as evl
#import create_results as res
from data_set_params import DataSetParams
#import classifier as clss
import pandas as pd
#import pickle
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import plotly.graph_objs as go
import joblib
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

import tensorflow as tf

from helper_fns import *
from tf_cnn import cnn_all, prec_recall_curves
#import os
#os.environ['THEANO_FLAGS'] = 'optimizer=None'

def read_baseline_res(baseline_file_name, test_files):
    da   = pd.read_csv(baseline_file_name)
    pos  = []
    prob = []
    for ff in test_files:
        rr   = da[da['Filename'] == ff]
        inds = np.argsort(rr.TimeInFile.values)
        pos.append(rr.TimeInFile.values[inds])
        prob.append(rr.Quality.values[inds][..., np.newaxis])
    return pos, prob


if __name__ == '__main__':
    """
    This compares several different algorithms for bat echolocation detection.

    The results can vary by a few percent from run to run. If you don't want to
    run a specific model or baseline comment it out.
    """
    # load parameters
    params = DataSetParams()

    test_set      = params.test_set
    data_set      = 'data/train_test_split/test_set_' + test_set + '.npz'
    raw_audio_dir = 'data/wav/'
    base_line_dir = 'data/baselines/'
    result_dir    = 'results/'
    model_dir     = 'data/models/'
    params.audio_dir = raw_audio_dir

    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    print('test set:', test_set)
    plt.close('all')

    # train and test_pos are in units of seconds
    loaded_data_tr  = np.load(data_set, allow_pickle = True, encoding = 'latin1')
    train_pos       = loaded_data_tr['train_pos']
    train_files     = loaded_data_tr['train_files']
    train_durations = loaded_data_tr['train_durations']
    test_pos        = loaded_data_tr['test_pos']
    test_files      = loaded_data_tr['test_files']
    test_durations  = loaded_data_tr['test_durations']

    train_files_decode = [s.decode() for s in train_files]
    test_files_decode  = [s.decode() for s in test_files]

    train_paths_decode = [raw_audio_dir + fn for fn in train_files_decode]
    test_paths_decode  = [raw_audio_dir + fn for fn in test_files_decode]

    train_positions, train_class_labels = generate_training_positions(train_files_decode, train_pos, train_durations)
    test_positions, test_class_labels   = generate_training_positions(test_files_decode, test_pos, test_durations)

    train_features, train_labels = get_audio_features_and_labels(train_class_labels, train_positions, train_durations, train_paths_decode, params)
    test_features, test_labels   = get_audio_features_and_labels(test_class_labels, test_positions, test_durations, test_paths_decode, params)
    
    train_features = np.expand_dims(train_features,-1)
    test_features  = np.expand_dims(test_features,-1)
    
    input_shape = train_features.shape[1:]
    
    train_ds   = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    test_ds    = tf.data.Dataset.from_tensor_slices((test_features, test_labels))

    # perfect shuffling, i.e. all training elements shufled
    train_ds.shuffle(len(train_ds))

    batch_size = params.batchsize
    train_ds   = train_ds.batch(batch_size)
    test_ds    = test_ds.batch(batch_size)

    pr_fig = go.Figure()
    pr_fig.update_layout(title       = 'Precision-Recall Curve'+', '+test_set.capitalize(),
                         xaxis_title = 'Recall',
                         yaxis_title = 'Precision',
                         width       = 900,
                         height      = 750)

    auc_fig = go.Figure()
    auc_fig.update_layout(title = 'ROC Curve, '+ test_set,
                    xaxis_title = 'False Positive Rate',
                    yaxis_title = 'True Positive Rate',
                    width       = 900,
                    height      = 750)
    
    #
    #CNN big
    print('\ncnn big')
    params.classification_model = 'cnn'
    # train and test
    model  = cnn_all(train_ds, test_ds, params, input_shape, 'big', result_dir + test_set)
    y_pred = np.argmax(model.predict(test_ds), axis=-1)
    y_true = [y for _, y in test_ds.unbatch()]
    y_pred_proba  = model.predict(test_ds)[:,1]
    fpr, tpr, _   = roc_curve(y_true,  y_pred_proba)
    prec_cnn_big, rec_cnn_big = prec_recall_curves(y_true,  y_pred_proba)
    # save CNN model to file
    model.save(result_dir+test_set+'_small_cnn')
    pr_fig.add_trace(go.Scatter(x = rec_cnn_big, y = prec_cnn_big, 
                                mode = 'lines', name = 'CNN',
                                line = dict(color='#1f77b4')))
    auc_fig.add_trace(go.Scatter(x = fpr, y = tpr, 
                            mode = 'lines', name = 'CNN',
                            text = 'AUC='+str(round(roc_auc_score(y_true, y_pred_proba),3)),
                            line = dict(color='#1f77b4')))
    #
    # CNN small
    print('\ncnn small')
    params.classification_model = 'cnn'
    # train and test
    model  = cnn_all(train_ds, test_ds, params, input_shape, 'small', result_dir + test_set)
    y_pred = np.argmax(model.predict(test_ds), axis=-1)
    y_pred_proba  = model.predict(test_ds)[:,1]
    fpr, tpr, _   = roc_curve(y_true,  y_pred_proba)
    prec_cnn_small, rec_cnn_small = prec_recall_curves(y_true,  y_pred_proba)
    # save CNN model to file
    model.save(result_dir+test_set+'_big_cnn')
    pr_fig.add_trace(go.Scatter(x = rec_cnn_small, y = prec_cnn_small, 
                            mode = 'lines', name = 'CNN<sub>Fast<sub>',
                            line = dict(color='black')))
    auc_fig.add_trace(go.Scatter(x = fpr, y = tpr, 
                            mode = 'lines', name = 'CNN<sub>Fast<sub>',
                            text = 'AUC='+str(round(roc_auc_score(y_true, y_pred_proba),3)),
                            line = dict(color='black')))
    
    #
    # random forest
    print('\nrandom forest')
    params.classification_model = 'rf_vanilla'
    # train and test
    train_features_s    = np.array([x for x,_ in train_ds.unbatch()]) # shuffled
    train_features_flat = train_features_s.reshape(train_features_s.shape[0], train_features_s.shape[1]*train_features_s.shape[2])
    test_features_flat  = test_features.reshape(test_features.shape[0], test_features.shape[1]*test_features.shape[2])

    rf_model = RandomForestClassifier(
        n_jobs            = -1,
        n_estimators      = params.trees,
        max_depth         = params.depth,
        min_samples_split = params.min_cnt)
    print('Fitting Random Forest Classifier')
    rf_model.fit(train_features_flat, train_labels)
    print('Done')
    # compute precision recall
    y_pred_proba_rf = rf_model.predict_proba(test_features_flat)[:,1]
    fpr_rf, tpr_rf, _  = roc_curve(y_true,  y_pred_proba_rf)
    prec_rf, rec_rf = prec_recall_curves(y_true,  y_pred_proba_rf)
    # save
    joblib.dump(rf_model, result_dir+test_set+"random_forest.joblib")
    pr_fig.add_trace(go.Scatter(x = rec_rf, y = prec_rf, 
                            mode = 'lines', name = 'Random Forest',
                            line = dict(color='forestgreen')))
    auc_fig.add_trace(go.Scatter(x = fpr_rf, y = tpr_rf, 
                            mode = 'lines', name = 'Random Forest',
                            text = 'AUC='+str(round(roc_auc_score(y_true, y_pred_proba_rf),3)),
                            line = dict(color='forestgreen')))

    #
    # light gbm
    if params.lightgbm:
        print('\nLightGBM')
        params.classification_model = 'lightgbm'
        
        lgb_model = LGBMClassifier(
            learning_rate = params.learn_rate,
            n_estimators  = params.trees,
            max_depth     = params.depth)
        rf_model = RandomForestClassifier(
            n_jobs=-1,
            n_estimators = params.trees,
            max_depth    = params.depth,
            min_samples_split = params.min_cnt)
        print('Fitting LightGBM Classifier')
        lgb_model.fit(train_features_flat, train_labels)
        print('Done')
        # compute precision recall
        y_pred_proba_lgb   = lgb_model.predict_proba(test_features_flat)[:,1]
        fpr_lgb, tpr_lgb, _  = roc_curve(y_true,  y_pred_proba_lgb)
        prec_lgb, rec_lgb  = prec_recall_curves(y_true,  y_pred_proba_lgb)
        # save
        joblib.dump(lgb_model, result_dir+test_set+"lightgbm.joblib")
        pr_fig.add_trace(go.Scatter(x = rec_lgb, y = prec_lgb, 
                                mode = 'lines', name = 'Light GBM',
                                line = dict(color='lightyellow')))
        auc_fig.add_trace(go.Scatter(x = fpr_lgb, y = tpr_lgb, 
                            mode = 'lines', name = 'Light GBM',
                            text = 'AUC='+str(round(roc_auc_score(y_true, y_pred_proba_rf),3)),
                            line = dict(color='lightyellow')))

    #
    # segment
    print('\nsegment')
    params.classification_model = 'segment'
    # train and test
    specs = []
    for file_name in test_paths_decode:
        sampling_rate, audio_samples = wavfile.read(file_name + '.wav')

        spectrogram = gen_spectrogram(audio_samples, sampling_rate, params.fft_win_length,
                    params.fft_overlap, crop_spec=params.crop_spec, max_freq=params.max_freq,
                    min_freq=params.min_freq)
        spectrogram = process_spectrogram(spectrogram, denoise_spec=params.denoise,
                    mean_log_mag=params.mean_log_mag, smooth_spec=params.smooth_spec)
        specs.append(spectrogram)
    seg_pos, seg_prob, y_pred_seg = segment_fn(specs, test_durations, spectrogram, params)
    # compute precision recall
    prec_seg, rec_seg = evl.prec_recall_1d(seg_pos, seg_prob, test_pos, test_durations, params.detection_overlap, params.window_size)
    pr_fig.add_trace(go.Scatter(x = rec_seg, y = prec_seg, 
                            mode = 'markers', name = 'Segment',
                            line = dict(color='crimson')))

    #
    # scanr
    scanr_bat_results = base_line_dir + 'scanr/test_set_'+ test_set +'_scanr.csv'
    if os.path.isfile(scanr_bat_results):
        print('\nscanr')
        scanr_pos, scanr_prob = read_baseline_res(scanr_bat_results, test_files_decode)
        prec_scanr, rec_scanr = evl.prec_recall_1d(scanr_pos, scanr_prob, test_pos, test_durations, params.detection_overlap, params.window_size)
        pr_fig.add_trace(go.Scatter(x = rec_scanr, y = prec_scanr,
                            mode = 'markers', name = 'SCAN\'R',
                            line = dict(color='indigo')))
    #
    # sonobat
    sono_bat_results = base_line_dir + 'sonobat/test_set_'+ test_set +'_sono.csv'
    if os.path.isfile(sono_bat_results):
        print('\nsonobat')
        sono_pos, sono_prob = read_baseline_res(sono_bat_results, test_files_decode)
        precision_sono, recall_sono = evl.prec_recall_1d(sono_pos, sono_prob, test_pos, test_durations, params.detection_overlap, params.window_size)
        pr_fig.add_trace(go.Scatter(x = recall_sono, y = precision_sono, 
                            mode = 'lines', name = 'SonoBat',
                            line = dict(color='burlywood')))

    #
    # kaleidoscope
    kal_bat_results = base_line_dir + 'kaleidoscope/test_set_'+ test_set +'_kaleidoscope.csv'
    if os.path.isfile(kal_bat_results):
        print('\nkaleidoscope')
        kal_pos, kal_prob = read_baseline_res(kal_bat_results, test_files_decode)
        precision_kal, recall_kal = evl.prec_recall_1d(kal_pos, kal_prob, test_pos, test_durations, params.detection_overlap, params.window_size)
        pr_fig.add_trace(go.Scatter(x = recall_kal, y = precision_kal, 
                            mode = 'markers', name = 'Kaleidoscope',
                            line = dict(color='magenta')))
    
    #pr_fig.show()
    print('Saving precision-recall curve')
    pr_fig.write_html(result_dir + test_set + '_results.html')
    #pr_fig.write_image(result_dir + test_set + '_results.pdf')  
        
    
    baseline_vals = np.arange(0.0,1.0,0.0100)
    auc_fig.add_trace(go.Scatter(x = baseline_vals, y = baseline_vals, 
                            mode = 'lines', name = 'Baseline',
                            line = dict(color='black', dash='dash')))
    #auc_fig.show()
    print('Saving ROC curve')
    auc_fig.write_html(result_dir + test_set + '_roc_curve.html')
    #pr_fig.write_image(result_dir + test_set + '_results.pdf')
    print('Done run_comparison.py')