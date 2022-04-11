import argparse, os
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from prettytable import PrettyTable

from config.Model import Model
from utils.general_util import json_load, json_dump
from utils.plot_util import draw_skel
from utils.VideoReaderFast import VideoReaderFast
from utils.VideoWriterFast import *
from utils.StitchedImage import StitchedImage
import utils.CamLib as cl


def load_data(pose_var_file, laser_data_file):
    pose_vars = json_load(pose_var_file)
    laser_data = json_load(laser_data_file)['frames']
    return pose_vars, laser_data


def to_mat(args, pose_vars, laser_data):
    # turn pose vars into a matrix
    vars, num_time_steps, var_names = list(), None, list()
    for name, data in pose_vars.items():
        data = np.array(data)
        if num_time_steps is None:
            num_time_steps = data.shape[0]
        else:
            assert num_time_steps == data.shape[0], 'This should hold.'

        vars.append(data)
        var_names.append(name)
    vars = np.stack(vars, 1)

    # create class label vector
    labels = np.zeros((num_time_steps,))  # 0 dont know, 1 laser on (sure), -1 laser off (sure)
    laser_data = np.array(laser_data)  # frames where the laser was switched on
    laser_data_off = laser_data + args.laser_pulse_width # frames where the laser was switched on
    margin_off = args.margin_laser_off
    margin_on_f = args.margin_laser_on_front
    margin_on_b = args.margin_laser_on_back
    for i in range(num_time_steps):
        for k in range(len(laser_data)):
            if i-margin_on_f >= laser_data[k] and i+margin_on_b <= laser_data_off[k]:
                labels[i] = 1.0

            if k == 0 and i+margin_off < laser_data[k]:
                labels[i] = -1.0

            if i-margin_off > laser_data_off[k]:
                if k < len(laser_data) - 1:
                    if i+margin_off < laser_data[k+1]:
                        labels[i] = -1.0
                else:
                    labels[i] = -1.0

    # from utils.mpl_setup import plt_figure
    # plt, fig, axes = plt_figure(1)
    # axes[0].plot(labels)
    # axes[0].plot(laser_data, np.ones_like(laser_data)*2, 'go')
    # axes[0].plot(laser_data_off, np.ones_like(laser_data_off)*2, 'ro')
    # plt.show()

    return var_names, vars, labels


def _classify(cls_type, class_labels, pose_mat, class_labels_eval, pose_mat_eval, return_pred=False):
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.linear_model import SGDClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import classification_report
    from scipy.stats import chi2_contingency
    from sklearn.metrics.cluster import contingency_matrix
    from sklearn.model_selection import GridSearchCV

    # create a scaler and a classifier
    scaler = StandardScaler()

    if cls_type == 'svm_linear':
        classifier = SVC(kernel="linear", C=0.001)  # linear SVM

        # C_range = [0.00001, 0.0001, 0.001, 0.1]
        # params = {"C": C_range}
        # # 511@30:0.0001, 512@30:0.001, 513@30
        # # 511@10: 0.001 , 512@10: 0.001, 513@10:
        # # grid = GridSearchCV(classifier, params, scoring='balanced_accuracy', n_jobs=4)
        # grid = GridSearchCV(classifier, params, scoring='f1_macro', n_jobs=4)
        # y_train = class_labels
        # X_train = scaler.fit_transform(pose_mat)
        # grid.fit(X_train, y_train)
        # print("The best parameters are %s with a score of %0.2f"
        #       % (grid.best_params_, grid.best_score_))
        #
        # scores = grid.cv_results_['mean_test_score'].reshape(len(C_range))
        #
        # from utils.mpl_setup import plt_figure
        # plt, fig, axes = plt_figure(1)
        # axes[0].semilogx(C_range, scores)
        # plt.show()

    elif cls_type == 'svm_linear_w':
        # fg_w = float(class_labels.shape[0]) / np.sum(class_labels > 0.0)
        classifier = SVC(kernel="linear", C=0.025, class_weight='balanced')#'{1: fg_w})  # linear SVM

    elif cls_type == 'sgd':
        classifier = SGDClassifier(loss="hinge")

    elif cls_type == 'svm_nl':
        classifier = SVC(kernel='rbf', gamma=1e-5, C=100) # RBF SVM

        # C_range = [10, 100, 1000]
        # gamma_range = [0.0001, 0.00001, 0.000001]
        # params = {"C": C_range, "gamma": gamma_range}
        #
        # grid = GridSearchCV(classifier, params, scoring='balanced_accuracy', n_jobs=4)
        #
        # y_train = class_labels
        # X_train = scaler.fit_transform(pose_mat)
        # grid.fit(X_train, y_train)
        # print("The best parameters are %s with a score of %0.2f"
        #       % (grid.best_params_, grid.best_score_))
        #
        # scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
        #                                                      len(gamma_range))
        # from utils.mpl_setup import plt_figure
        # plt, fig, axes = plt_figure(1)
        # axes[0].imshow(scores)
        # axes[0].set_xticks(np.arange(len(gamma_range)), gamma_range)
        # axes[0].set_yticks(np.arange(len(C_range)), C_range)
        # axes[0].set_xlabel('gamma')
        # axes[0].set_ylabel('C')
        # plt.show()

    elif cls_type == 'svm_nl_w':
        # fg_w = float(class_labels.shape[0]) / np.sum(class_labels > 0.0)
        # classifier = SVC(kernel='rbf', gamma='scale', C=0.025, class_weight='balanced')#{1: fg_w}) # RBF SVM
        classifier = SVC(kernel='poly', gamma='scale', C=0.025, class_weight='balanced')

    elif cls_type == 'mlp':
        classifier = MLPClassifier(alpha=1.0, max_iter=1000, batch_size=64, verbose=True)

    else:
        raise NotImplementedError('Invalid classifier type.')

    # train classifier
    y_train = class_labels
    X_train = scaler.fit_transform(pose_mat)
    classifier.fit(X_train, y_train)

    # evaluate classifier
    y_eval = class_labels_eval
    X_eval = scaler.transform(pose_mat_eval)
    y_pred = classifier.predict(X_eval)
    pred = {
        'gt': y_eval,
        'pred': y_pred
    }

    # score classifier
    report = classification_report(y_eval, y_pred, output_dict=True)
    fs = report['macro avg']['f1-score']
    # acc = np.mean(y_eval == y_pred)
    m = y_eval > 0.0
    acc = np.mean(y_eval[m] == y_pred[m])

    # chi2 test
    table = contingency_matrix(y_eval, y_pred)
    stat, pv, dof, expected = chi2_contingency(table)

    if return_pred:
        return pv, fs, acc, stat, dof, expected, pred
    return pv, fs, acc, stat, dof, expected


def attribution(cls_type,
                output_file, pose_names, pose_mat, class_labels, pose_mat_eval, class_labels_eval,
                output_file_pred, save_pred=False):
    """ Attribute the influence to some body variables. """

    # masking
    mask_train = np.abs(class_labels) > 0.5
    mask_eval = np.abs(class_labels_eval) > 0.5

    fmt = '%.3f'
    result_table = PrettyTable(['Experiment', 'p-value'])

    # classify from all factors
    if save_pred:
        ## looks like more than linear is not worth the effort
        # # for c in ['svm_linear', 'svm_linear_w',  'sgd', 'svm_nl', 'svm_nl_w', 'mlp', 'mlp', 'mlp', 'mlp']:
        # for c in ['svm_linear', 'svm_nl', 'mlp', 'mlp', 'mlp', 'mlp']:
        #     pv, fs, acc, pred = _classify(c,#cls_type,
        #                                   class_labels[mask_train], pose_mat[mask_train],
        #                                   class_labels_eval[mask_eval], pose_mat_eval[mask_eval], return_pred=save_pred)
        #     print('CLASSIFIER', c)
        #     print('ALL FACTORS ACCURACY:', acc)
        #     print('ALL FACTORS F1:', fs)
        #     print('---------')
        # exit()

        pv, fs, acc, stat, dof, expected, pred = _classify(cls_type,
                                      class_labels[mask_train], pose_mat[mask_train],
                                      class_labels_eval[mask_eval], pose_mat_eval[mask_eval], return_pred=save_pred)
        json_dump(output_file_pred, pred, verbose=True)
    else:
        pv, fs, acc, stat, dof, expected = _classify('svm_linear',
                                class_labels[mask_train], pose_mat[mask_train],
                                class_labels_eval[mask_eval], pose_mat_eval[mask_eval])
    acc_all_factors = acc
    result_table.add_row(
        ['all_factors', '%.3e' % pv]
    )

    # iterate single factors
    results = list()
    for i, name in tqdm(enumerate(pose_names), total=len(pose_names), desc='Testing single factors'):
        pv, fs, acc, stat, dof, expected = _classify('svm_linear',
                                class_labels[mask_train], pose_mat[mask_train, i:(i+1)],
                                class_labels_eval[mask_eval], pose_mat_eval[mask_eval, i:(i+1)])

        results.append((name, pv, fs, acc, stat, dof, expected,
                        class_labels[mask_train].shape[0], class_labels_eval[mask_eval].shape[0]))

    all_scores = np.array([x[1] for x in results])
    mean_score = np.mean(all_scores)
    result_table.add_row(
        ['single_factor_mean_score', '%.3f' % mean_score]
    )
    result_table.add_row(['---', '---'])

    sorted_inds = np.argsort(all_scores)[::-1]

    for i in sorted_inds:
        name = pose_names[i]
        v = all_scores[i]
        if v < 0.1:
            result_table.add_row([name, fmt % v])
    num_sig = np.sum(all_scores < 0.1)

    print('Attribution summary:')
    print('Number of significant factors %d' % num_sig)
    print('Train set: %d samples' % pose_mat.shape[0])
    print('Train set: %d samples valid' % np.sum(mask_train))
    print('Train set: %d factors' % pose_mat.shape[1])
    print('Eval set: %d samples' % pose_mat_eval.shape[0])
    print('Eval set: %d samples valid' % np.sum(mask_eval))
    print('Eval set: %d factors' % pose_mat_eval.shape[1])
    print(result_table)

    # with open(output_file, 'w') as fo:
    #     fo.write('Attribution summary: Attribution by classification\n')
    #     fo.write('Number of significant factors %d\n' % num_sig)
    #     fo.write('Train set: %d samples\n' % pose_mat.shape[0])
    #     fo.write('Train set: %d samples valid\n' % np.sum(mask_train))
    #     fo.write('Train set: %d factors\n' % pose_mat.shape[1])
    #     fo.write('Eval set: %d samples\n' % pose_mat_eval.shape[0])
    #     fo.write('Eval set: %d samples valid\n' % np.sum(mask_eval))
    #     fo.write('Eval set: %d factors\n' % pose_mat_eval.shape[1])
    #     fo.write(str(result_table))

    summary = {
        'num_train_samples': pose_mat.shape[0],
        'num_train_samples_valid': np.sum(mask_train),
        'num_eval_samples': pose_mat_eval.shape[0],
        'num_eval_samples_valid': np.sum(mask_eval),
        'num_factors': pose_mat.shape[1],
        'num_factors_sig': num_sig,
        'acc_all_factors': acc_all_factors,
        'results': results,
    }
    json_dump(output_file, summary)


def attribution_rnd_ensembles(pose_names, pose_mat, class_labels, pose_mat_eval, class_labels_eval,
                              num_runs=200, dropout_prob=0.8, show_top_n=10, show_bottom_n=10):
    """ Attribute the influence to some body variables. """

    # masking
    mask_train = np.abs(class_labels) > 0.5
    mask_eval = np.abs(class_labels_eval) > 0.5

    importance_mat = np.zeros((len(pose_names)))
    samples_mat = np.zeros((len(pose_names)))
    for i in tqdm(range(num_runs), desc='Running analysis'):
        # randomly eliminate some of the factors
        dropout_mask = (np.random.rand(1, len(pose_names)) >= dropout_prob).astype(np.float32)
        this_pose_mat = pose_mat.copy() * dropout_mask
        this_pose_mat_eval = pose_mat_eval.copy() * dropout_mask

        # classify from factors
        pv, fs, acc, stat, dof, expected = _classify(class_labels[mask_train], this_pose_mat[mask_train],
                                class_labels_eval[mask_eval], this_pose_mat_eval[mask_eval])

        m = dropout_mask[0, :] > 0.5
        importance_mat[m] += fs
        samples_mat[m] += 1.0

    # average
    importance_mat = importance_mat / (samples_mat + 1e-4)
    mean_v = importance_mat.mean()

    sorted_ids = np.argsort(importance_mat)[::-1]

    result_table = PrettyTable(['Experiment', 'Avg F1-score'])

    result_table.add_row(['mean_score', '%.3f' % mean_v])
    result_table.add_row(['---', '---'])
    for i in sorted_ids[:show_top_n]:
        name = pose_names[i]
        v = importance_mat[i]
        result_table.add_row(
            [name, '%.3f' % v]
        )
    result_table.add_row(['---', '---'])

    for i in sorted_ids[-show_bottom_n:]:
        name = pose_names[i]
        v = importance_mat[i]
        result_table.add_row(
            [name, '%.3f' % v]
        )

    print(result_table)


def _p_value_from_permutation(X, y):
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import permutation_test_score

    # classify with an linear SVM
    svm = SVC(kernel='linear')
    cv = StratifiedKFold(2)

    # scale input to unit var and zero mean
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # run permutation test
    score, permutation_scores, pvalue = permutation_test_score(
        svm, X_scaled, y, scoring="f1_macro", cv=cv, n_permutations=100, n_jobs=6)

    return pvalue, score


def attribution_by_permutation(output_file, pose_names, pose_mat, class_labels):
    """ Attribute the influence to some body variables. """

    # masking
    mask = np.abs(class_labels) > 0.5
    y = class_labels[mask]
    X = pose_mat[mask]
    n_classes = np.unique(y).size

    result_table = PrettyTable(['Experiment', 'p-value'])

    # # classify from all factors
    # v = _p_value_from_permutation(X, y)
    # result_table.add_row(
    #     ['all_factors', '%.3e' % v]
    # )

    # iterate single factors
    results = list()
    for i, name in tqdm(enumerate(pose_names), desc='Single factors', total=len(pose_names)):
        pv, fs = _p_value_from_permutation(X[:, i:(i+1)], y)
        results.append(
            (name, pv, fs)
        )

    all_scores = np.array([x[1] for x in results])
    mean_score = np.mean(all_scores)
    result_table.add_row(
        ['single_factor_mean_score', '%.3e' % mean_score]
    )
    result_table.add_row(['---', '---'])

    sorted_inds = np.argsort(all_scores)[::-1]
    for i in sorted_inds[:10]:
        name = pose_names[i]
        v = all_scores[i]
        result_table.add_row(
            [name, '%.3e' % v]
        )
    result_table.add_row(['---', '---'])

    for i in sorted_inds[-10:]:
        name = pose_names[i]
        v = all_scores[i]
        result_table.add_row(
            [name, '%.3e' % v]
        )
    num_sig = np.sum(all_scores < 0.1)

    print('Attribution summary:')
    print('Number of significant factors: %d' % num_sig)
    print('Data set: %d samples' % pose_mat.shape[0])
    print('Data set: %d samples valid' % np.sum(mask))
    print('Data set: %d factors' % pose_mat.shape[1])
    print(result_table)

    # with open(output_file, 'w') as fo:
    #     fo.write('Attribution summary Attribution by permutation\n')
    #     fo.write('Number of significant factors: %d\n' % num_sig)
    #     fo.write('Data set: %d samples\n' % pose_mat.shape[0])
    #     fo.write('Data set: %d samples valid\n' % np.sum(mask))
    #     fo.write('Data set: %d factors\n' % pose_mat.shape[1])
    #     fo.write(str(result_table))

    summary = {
        'num_samples': pose_mat.shape[0],
        'num_samples_valid': np.sum(mask),
        'num_factors': pose_mat.shape[1],
        'num_factors_sig': num_sig,
        'results': results,
    }
    json_dump(output_file, summary)


def correlate_body_vars(pose_names, pose_mat, class_labels,
                        block_size=25):
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    from scipy.cluster import hierarchy

    # masking
    mask = np.abs(class_labels) > 0.5
    y = class_labels[mask]
    X = pose_mat[mask]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    corr = spearmanr(X).correlation
    corr_linkage = hierarchy.ward(corr)
    dendro = hierarchy.dendrogram(corr_linkage, labels=pose_names, ax=ax1,
                                  leaf_rotation=90)
    dendro_idx = np.arange(0, len(dendro['ivl']))

    ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
    ax2.set_yticklabels(dendro['ivl'])
    ax2.tick_params(axis='both', labelsize=6)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show predictions.')
    parser.add_argument('pose_vars', type=str, help='Input video file.')
    parser.add_argument('laser_data', type=str, help='Input laser file.')
    parser.add_argument('pose_vars_test', type=str, help='Input video file.')
    parser.add_argument('laser_data_test', type=str, help='Input video file.')

    parser.add_argument('--mode', type=str, default='chi', help='What test to run.')
    parser.add_argument('--name', type=str, help='Given name, is reflected in the output file.')
    parser.add_argument('--cls_type', type=str, help='Classifier type.')

    parser.add_argument('--random', action='store_true', help='Randomize labels for sanity check.')
    parser.add_argument('--save_pred', action='store_true', help='Save the classifiers predictions.')

    parser.add_argument('--laser_pulse_width', type=int, default=150, help='Length of laser on in frames.')
    parser.add_argument('--margin_laser_off', type=int, default=150, help='Margin in laser off times.')
    parser.add_argument('--margin_laser_on_front', type=int, default=40, help='Margin between laser on and "on" class start.')
    parser.add_argument('--margin_laser_on_back', type=int, default=0, help='Margin between laser on and "on" class end.')
    args = parser.parse_args()

    print('Running: %s' % args.mode)

    # figure out output file name
    name = '' if args.name is None else args.name
    if args.name is not None:
        name += '_'
    cls_type = 'svm_linear' if args.cls_type is None else args.cls_type
    output_file = os.path.join(
        os.path.dirname(args.laser_data),
        'with_stats_attr_%s%s_%s_off%03d_on%03d_%03d.json' % (name, cls_type,
                                                   args.mode, args.margin_laser_off,
                                                   args.margin_laser_on_front, 150 - args.margin_laser_on_back)
    )
    output_file_pred = os.path.join(
        os.path.dirname(args.pose_vars_test), 'pred_%s%s.json' % (cls_type, name))

    # load files and turn into large matrices
    pose_vars, laser_data = load_data(args.pose_vars, args.laser_data)
    pose_names, pose_mat, class_labels = to_mat(args, pose_vars, laser_data)

    # load files and turn into large matrices
    pose_vars_test, laser_data_test = load_data(args.pose_vars_test, args.laser_data_test)
    _, pose_mat_test, class_labels_test = to_mat(args, pose_vars_test, laser_data_test)

    # shuffle labels as a sanity check: Gives F1-score=0.486 everywhere
    if args.random:
        np.random.shuffle(class_labels)
        np.random.shuffle(class_labels_test)

    # correlate_body_vars(pose_names,
    #                     np.concatenate([pose_mat, pose_mat_test], 0),
    #                     np.concatenate([class_labels, class_labels_test], 0))

    if args.mode == 'chi':
        # attribution by classification
        attribution(cls_type, output_file,
                    pose_names, pose_mat, class_labels,
                    pose_mat_test, class_labels_test,
                    output_file_pred, save_pred=args.save_pred)
        # attribution_rnd_ensembles(pose_names, pose_mat, class_labels,
        #             pose_mat_test, class_labels_test, scoring)

    elif args.mode == 'perm':
        # permutation test
        attribution_by_permutation(output_file,
                                   pose_names,
                                   np.concatenate([pose_mat, pose_mat_test], 0),
                                   np.concatenate([class_labels, class_labels_test], 0))


