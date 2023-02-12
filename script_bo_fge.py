import tensorflow as tf
import tensorflow.keras as tk

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)

from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import *
from tensorflow.keras.regularizers import *
from tensorflow.keras import *
from tensorflow.keras.callbacks import *
from tensorflow.python.keras import backend as K

import os

# must set these before loading numpy:
os.environ["OMP_NUM_THREADS"] = '2'  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = '2'  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = '2'  # export MKL_NUM_THREADS=6

import sys
import numpy as np
from sklearn.utils import shuffle
import random
import h5py
import time
import glob
from datetime import datetime
from scipy.stats import norm

import keras_tuner as kt
from keras_tuner.tuners import *

data_folder = "/tudelft.net/staff-umbrella/dlsca/Guilherme/"
save_folder = "/tudelft.net/staff-umbrella/dlsca/Guilherme/paper_fast_ge_bo"
save_folder_bo = "/tudelft.net/staff-umbrella/dlsca/Guilherme/paper_fast_ge_bo"
# data_folder = "D:/traces/"
# save_folder = "D:/postdoc/paper_fast_ge/results_fast_ge_bo"
# save_folder_bo = "D:\\postdoc\\paper_fast_ge\\results_fast_ge_bo"

ge_epochs = []
ge_epochs_time = []
mi_epochs = []
mi_epochs_time = []
output_probabilities_acc = []
y_pred_acc = []

AES_Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])


# labelize for key guesess for guessing entropy and success rate
def aes_labelize_ge_sr(trace_data, byte, round_key, leakage_model):
    plaintext = [row[byte] for row in trace_data]
    key_byte = np.full(len(plaintext), round_key[byte])
    state = [int(x) ^ int(k) for x, k in zip(np.asarray(plaintext[:]), key_byte)]
    intermediate_values = AES_Sbox[state]

    if leakage_model == "HW":
        return [bin(iv).count("1") for iv in intermediate_values]
    else:
        return intermediate_values


def aes_labelize(trace_data, byte, leakage_model):
    plaintext = [row[byte] for row in trace_data]
    key_byte = [row[byte + 16] for row in trace_data]
    key_byte = np.asarray(key_byte[:])
    state = [int(x) ^ int(k) for x, k in zip(np.asarray(plaintext[:]), key_byte)]
    intermediate_values = AES_Sbox[state]

    if leakage_model == "HW":
        return [bin(iv).count("1") for iv in intermediate_values]
    else:
        return intermediate_values


def load_dataset(dataset_file, n_profiling, n_validation, n_attack, first_sample, number_of_samples, target_byte):
    in_file = h5py.File(dataset_file, "r")

    profiling_samples = np.array(in_file['Profiling_traces/traces'], dtype=np.float64)
    attack_samples = np.array(in_file['Attack_traces/traces'], dtype=np.float64)
    profiling_key = in_file['Profiling_traces/metadata']['key']
    attack_key = in_file['Attack_traces/metadata']['key']

    profiling_data = np.zeros((n_profiling, 32))
    validation_data = np.zeros((n_validation, 32))
    attack_data = np.zeros((n_attack, 32))

    profiling_plaintext = in_file['Profiling_traces/metadata']['plaintext']
    attack_plaintext = in_file['Attack_traces/metadata']['plaintext']
    for i in range(n_profiling):
        profiling_data[i][0:16] = profiling_plaintext[i]
        profiling_data[i][16:32] = profiling_key[i]

    for i in range(n_validation):
        validation_data[i][0:16] = attack_plaintext[i]
        validation_data[i][16:32] = attack_key[i]

    for i in range(n_attack):
        attack_data[i][0:16] = attack_plaintext[n_validation + i]
        attack_data[i][16:32] = attack_key[n_validation + i]

    X_profiling = profiling_samples[0:n_profiling, first_sample:first_sample + number_of_samples]
    X_validation = attack_samples[0:n_validation, first_sample:first_sample + number_of_samples]
    X_attack = attack_samples[n_validation:n_validation + n_attack, first_sample:first_sample + number_of_samples]

    validation_plaintext = attack_plaintext[0:n_validation]
    attack_plaintext = attack_plaintext[n_validation:n_validation + n_attack]

    return (X_profiling, X_validation, X_attack), (profiling_data, validation_data, attack_data), (
        profiling_plaintext[:, target_byte], attack_plaintext[:, target_byte], validation_plaintext[:, target_byte])


def create_z_score_norm(dataset):
    z_score_mean = np.mean(dataset, axis=0)
    z_score_std = np.std(dataset, axis=0)
    return z_score_mean, z_score_std


def apply_z_score_norm(dataset, z_score_mean, z_score_std):
    for index in range(len(dataset)):
        dataset[index] = (dataset[index] - z_score_mean) / z_score_std


# guessing entropy and success rate
def sca_metrics(runs, key_rank_attack_traces, model, x_attack, labels_key_hypothesis, correct_key, leakage_model):
    nt = len(x_attack)
    key_rank_report_interval = 1
    nt_interval = int(key_rank_attack_traces / key_rank_report_interval)
    key_ranking_sum = np.zeros(nt_interval)
    success_rate_sum = np.zeros(nt_interval)

    output_probabilities = np.log(model.predict(x_attack) + 1e-36)

    if leakage_model == "HW":
        probabilities_kg_all_traces = np.choose(labels_key_hypothesis, output_probabilities.T).T
    else:
        probabilities_kg_all_traces = np.zeros((nt, 256))
        for index in range(nt):
            probabilities_kg_all_traces[index] = output_probabilities[index][
                np.asarray([int(leakage[index]) for leakage in labels_key_hypothesis[:]])
            ]

    # ---------------------------------------------------------------------------------------------------------#
    # run key rank "runs" times and average results.
    # ---------------------------------------------------------------------------------------------------------#
    for run in range(runs):
        probabilities_kg_all_traces_shuffled = shuffle(probabilities_kg_all_traces, random_state=random.randint(0, 100000))
        key_probabilities = np.zeros(256)

        kr_count = 0
        for index in range(key_rank_attack_traces):

            key_probabilities += probabilities_kg_all_traces_shuffled[index]
            key_probabilities_sorted = np.argsort(key_probabilities)[::-1]

            if (index + 1) % key_rank_report_interval == 0:
                key_ranking_good_key = list(key_probabilities_sorted).index(correct_key) + 1
                key_ranking_sum[kr_count] += key_ranking_good_key

                if key_ranking_good_key == 1:
                    success_rate_sum[kr_count] += 1

                kr_count += 1

    guessing_entropy = key_ranking_sum / runs
    success_rate = success_rate_sum / runs

    result_number_of_traces_val = key_rank_attack_traces
    if guessing_entropy[nt_interval - 1] < 2:
        for index in range(nt_interval - 1, -1, -1):
            if guessing_entropy[index] > 2:
                result_number_of_traces_val = (index + 1) * key_rank_report_interval
                break

    print("GE = {}".format(guessing_entropy[nt_interval - 1]))
    print("SR = {}".format(success_rate[nt_interval - 1]))
    print("Number of traces to reach GE = 1: {}".format(result_number_of_traces_val))

    return guessing_entropy, success_rate, result_number_of_traces_val


def fast_ge(nt, y_true, y_pred, nt_kr, runs=100):
    start_time = time.time()

    # ---------------------------------------------------------------------------------------------------------#
    # predict output probabilities for shuffled test or validation set
    # ---------------------------------------------------------------------------------------------------------#
    output_probabilities = np.log(y_pred + 1e-36)

    key_ranking_sum = 0
    if leakage_model == "HW":
        probabilities_kg_all_traces = np.choose(labels_key_hypothesis_fast, output_probabilities.T).T
    else:
        probabilities_kg_all_traces = np.zeros((nt, 256))
        for index in range(nt):
            probabilities_kg_all_traces[index] = output_probabilities[index][
                np.asarray([int(leakage[index]) for leakage in labels_key_hypothesis_fast[:]])
            ]

    # ---------------------------------------------------------------------------------------------------------#
    # run key rank "runs" times and average results.
    # ---------------------------------------------------------------------------------------------------------#
    for run in range(runs):
        r = np.random.choice(range(nt), nt_kr, replace=False)
        probabilities_kg_all_traces_shuffled = probabilities_kg_all_traces[r]
        key_probabilities = np.sum(probabilities_kg_all_traces_shuffled[:nt_kr], axis=0)
        key_probabilities_sorted = np.argsort(key_probabilities)[::-1]
        key_ranking_sum += list(key_probabilities_sorted).index(correct_key) + 1

    guessing_entropy = key_ranking_sum / runs
    print("GE = {}".format(guessing_entropy))

    ge_epochs.append(guessing_entropy)
    ge_epochs_time.append(time.time() - start_time)
    if len(ge_epochs) == epochs:
        np.savez("{}/{}/ge_epochs_temp.npz".format(save_folder, project_folder),
                 ge_epochs=ge_epochs,
                 ge_epochs_time=ge_epochs_time
                 )
        ge_epochs.clear()
        ge_epochs_time.clear()

    return guessing_entropy


def fast_ge_batch_size(nt, y_true, y_pred, nt_kr, runs=100):
    # ---------------------------------------------------------------------------------------------------------#
    # predict output probabilities for shuffled test or validation set
    # ---------------------------------------------------------------------------------------------------------#
    output_probabilities = np.log(y_pred + 1e-36)
    output_probabilities_acc.append(output_probabilities)

    if len(output_probabilities_acc) == int(validation_traces_fast / batch_size):

        start_time = time.time()

        o = np.asarray(output_probabilities_acc)
        output_probabilities_acc_np = o.reshape(o.shape[0] * o.shape[1], o.shape[2])

        key_ranking_sum = 0
        if leakage_model == "HW":
            probabilities_kg_all_traces = np.choose(labels_key_hypothesis_fast, output_probabilities_acc_np.T).T
        else:
            probabilities_kg_all_traces = np.zeros((nt, 256))
            for index in range(nt):
                probabilities_kg_all_traces[index] = output_probabilities_acc_np[index][
                    np.asarray([int(leakage[index]) for leakage in labels_key_hypothesis_fast[:]])
                ]

        # ---------------------------------------------------------------------------------------------------------#
        # run key rank "runs" times and average results.
        # ---------------------------------------------------------------------------------------------------------#
        for run in range(runs):
            r = np.random.choice(range(nt), nt_kr, replace=False)
            probabilities_kg_all_traces_shuffled = probabilities_kg_all_traces[r]
            key_probabilities = np.sum(probabilities_kg_all_traces_shuffled[:nt_kr], axis=0)
            key_probabilities_sorted = np.argsort(key_probabilities)[::-1]
            key_ranking_sum += list(key_probabilities_sorted).index(correct_key) + 1

        guessing_entropy = key_ranking_sum / runs
        print("GE = {}".format(guessing_entropy))

        ge_epochs.append(guessing_entropy)
        ge_epochs_time.append(time.time() - start_time)
        if len(ge_epochs) == epochs:
            np.savez("{}/{}/ge_epochs_temp.npz".format(save_folder, project_folder),
                     ge_epochs=ge_epochs,
                     ge_epochs_time=ge_epochs_time
                     )
            ge_epochs.clear()
            ge_epochs_time.clear()

        output_probabilities_acc.clear()

        return guessing_entropy
    else:
        return 0


def geea_batch_size(nt, y_true, y_pred, nt_kr, runs=100):
    output_probabilities = np.log(y_pred + 1e-36)
    output_probabilities_acc.append(output_probabilities)

    if len(output_probabilities_acc) == int(validation_traces_fast / batch_size):

        start_time = time.time()

        o = np.asarray(output_probabilities_acc)
        output_probabilities_acc_np = o.reshape(o.shape[0] * o.shape[1], o.shape[2])

        if leakage_model == "HW":
            probabilities_kg_all_traces = np.choose(labels_key_hypothesis_fast, output_probabilities_acc_np.T).T
        else:
            probabilities_kg_all_traces = np.zeros((nt, 256))
            for index in range(nt):
                probabilities_kg_all_traces[index] = output_probabilities_acc_np[index][
                    np.asarray([int(leakage[index]) for leakage in labels_key_hypothesis_fast[:]])
                ]

        kc = probabilities_kg_all_traces[:, correct_key]
        kc = kc[:, np.newaxis]
        sr = np.subtract(probabilities_kg_all_traces, kc)

        mean_kg_t = np.zeros(256)
        var_kg_t = np.zeros(256)

        for kg in range(256):
            if kg != correct_key:
                mean_kg_t[kg] = np.divide(np.sum(sr[:, kg]), nt)

        for kg in range(256):
            if kg != correct_key:
                var_kg_t[kg] = np.sqrt(np.divide(np.sum(np.square(np.subtract(sr[:, kg], mean_kg_t[kg]))), nt))

        q = np.sqrt(nt)
        guessing_entropy = 0
        for kg in range(256):
            if kg != correct_key:
                guessing_entropy += norm.cdf(q * mean_kg_t[kg] / var_kg_t[kg])
        guessing_entropy += 1

        print("GEEA = {}".format(guessing_entropy))

        ge_epochs.append(guessing_entropy)
        ge_epochs_time.append(time.time() - start_time)
        if len(ge_epochs) == epochs:
            np.savez("{}/{}/ge_epochs_temp.npz".format(save_folder, project_folder),
                     ge_epochs=ge_epochs,
                     ge_epochs_time=ge_epochs_time
                     )
            ge_epochs.clear()
            ge_epochs_time.clear()

        output_probabilities_acc.clear()

        return guessing_entropy
    else:
        return 0


def mi_batch_size(y_true, y_pred):
    output_probabilities_acc.append(y_pred)
    y_pred_acc.append(y_true[:, :classes])

    if len(output_probabilities_acc) == int(validation_traces_fast / batch_size):
        start_time = time.time()
        o = np.asarray(output_probabilities_acc)
        output_probabilities_acc_np = o.reshape(o.shape[0] * o.shape[1], o.shape[2])
        y = np.asarray(y_pred_acc)
        y_true_acc_np = y.reshape(y.shape[0] * y.shape[1], y.shape[2])

        num_of_bins = 100

        p_xs, unique_inverse_x = get_unique_probabilities(y_true_acc_np)

        bins = np.linspace(-1, 1, num_of_bins, dtype='float32')
        digitized = bins[np.digitize(np.squeeze(output_probabilities_acc_np.reshape(1, -1)), bins) - 1].reshape(
            len(output_probabilities_acc_np), -1)
        p_ts, _ = get_unique_probabilities(digitized)

        entropy_activations = -np.sum(p_ts * np.log(p_ts))
        entropy_activations_given_input = 0.
        for x in np.arange(len(p_xs)):
            p_t_given_x, _ = get_unique_probabilities(digitized[unique_inverse_x == x, :])
            entropy_activations_given_input += - p_xs[x] * np.sum(p_t_given_x * np.log(p_t_given_x))

        mi = entropy_activations - entropy_activations_given_input
        print("MI = {}".format(mi))

        mi_epochs.append(mi)
        mi_epochs_time.append(time.time() - start_time)
        if len(mi_epochs) == epochs:
            np.savez("{}/{}/mi_epochs_temp.npz".format(save_folder, project_folder),
                     mi_epochs=mi_epochs,
                     mi_epochs_time=mi_epochs_time
                     )
            mi_epochs.clear()
            mi_epochs_time.clear()

        output_probabilities_acc.clear()
        y_pred_acc.clear()

        return mi
    else:
        return 0


def get_unique_probabilities(x):
    unique_ids = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    _, unique_inverse, unique_counts = np.unique(unique_ids, return_index=False, return_inverse=True,
                                                 return_counts=True)
    return np.asarray(unique_counts / float(sum(unique_counts))), unique_inverse


# ================= Core =====================
# calculate key prob for all keys
def calculate_fast_ge(y_true, y_pred):
    plt_attack = y_true[:, classes:]
    if plt_attack[0][0] == 1:  # check if data is from validation set, then compute GE
        ge = np.zeros(1)
        ge[0] = fast_ge_batch_size(validation_traces_fast, y_true, y_pred, number_of_traces_key_rank)
        # ge[0] = fast_ge(validation_traces_fast, y_true, y_pred, number_of_traces_key_rank)
        GE = np.float32(ge)
    else:  # otherwise, return zeros
        GE = np.float32(np.zeros(1))
    return GE


# ================= Core =====================
# calculate key prob for all keys
def calculate_geea(y_true, y_pred):
    plt_attack = y_true[:, classes:]
    if plt_attack[0][0] == 1:  # check if data is from validation set, then compute GE
        ge = np.zeros(1)
        ge[0] = geea_batch_size(validation_traces_fast, y_true, y_pred, number_of_traces_key_rank)
        # ge[0] = fast_ge(validation_traces_fast, y_true, y_pred, number_of_traces_key_rank)
        GE = np.float32(ge)
    else:  # otherwise, return zeros
        GE = np.float32(np.zeros(1))
    return GE


# ================= Core =====================
# calculate key prob for all keys
def calculate_mi(y_true, y_pred):
    plt_attack = y_true[:, classes:]
    if plt_attack[0][0] == 1:  # check if data is from validation set, then compute GE
        mi = np.zeros(1)
        mi[0] = mi_batch_size(y_true, y_pred)
        MI = np.float32(mi)
    else:  # otherwise, return zeros
        MI = np.float32(np.zeros(1))
    return MI


@tf.function
def tf_calculate_fast_ge(y_true, y_pred):
    _ret = tf.numpy_function(calculate_fast_ge, [y_true, y_pred], tf.float32)
    return _ret


@tf.function
def tf_calculate_geea(y_true, y_pred):
    _ret = tf.numpy_function(calculate_geea, [y_true, y_pred], tf.float32)
    return _ret


@tf.function
def tf_calculate_mi(y_true, y_pred):
    _ret = tf.numpy_function(calculate_mi, [y_true, y_pred], tf.float32)
    return _ret


# Objective: GE
def rk_key(final_ge):
    return np.float32(final_ge)


# loss: categorical_crossentropy
def custom_loss(y_true, y_pred):
    return tk.backend.categorical_crossentropy(y_true[:, :classes], y_pred)


class key_rank_Metric_Fast(tk.metrics.Metric):
    def __init__(self, name='fast_ge', **kwargs):
        super(key_rank_Metric_Fast, self).__init__(name=name, **kwargs)
        self.ge_sum = self.add_weight(name='ge_sum', shape=1, initializer='zeros')
        self.ge_epochs = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.ge_sum.assign_add(tf_calculate_fast_ge(y_true, y_pred))

    def result(self):
        return tf.numpy_function(rk_key, [self.ge_sum], tf.float32)

    def reset_states(self):
        self.ge_sum.assign(K.zeros(1))


class key_rank_Metric(tk.metrics.Metric):
    def __init__(self, name='ge', **kwargs):
        super(key_rank_Metric, self).__init__(name=name, **kwargs)
        self.ge_sum = self.add_weight(name='ge_sum', shape=1, initializer='zeros')
        self.ge_epochs = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.ge_sum.assign_add(tf_calculate_fast_ge(y_true, y_pred))

    def result(self):
        return tf.numpy_function(rk_key, [self.ge_sum], tf.float32)

    def reset_states(self):
        self.ge_sum.assign(K.zeros(1))


class geea_Metric(tk.metrics.Metric):
    def __init__(self, name='geea', **kwargs):
        super(geea_Metric, self).__init__(name=name, **kwargs)
        self.geea_sum = self.add_weight(name='geea_sum', shape=1, initializer='zeros')
        self.geea_epochs = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.geea_sum.assign_add(tf_calculate_geea(y_true, y_pred))

    def result(self):
        return tf.numpy_function(rk_key, [self.geea_sum], tf.float32)

    def reset_states(self):
        self.geea_sum.assign(K.zeros(1))


class mutual_information_Metric(tk.metrics.Metric):
    def __init__(self, name='mutual_information', **kwargs):
        super(mutual_information_Metric, self).__init__(name=name, **kwargs)
        self.mi_sum = self.add_weight(name='mi_sum', shape=1, initializer='zeros')
        self.mi_epochs = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.mi_sum.assign_add(tf_calculate_mi(y_true, y_pred))

    def result(self):
        return tf.numpy_function(rk_key, [self.mi_sum], tf.float32)

    def reset_states(self):
        self.mi_sum.assign(K.zeros(1))


class Callback_EarlyStopping(Callback):
    def __init__(self, X_profiling_reshape, X_attack_reshape, X_validation_reshape, Y_profiling, Y_attack, Y_validation, validation_traces,
                 attacking_traces, labels_key_hypothesis, labels_key_hypothesis_attack, leakage_model, correct_key, batch_size, epochs,
                 project_folder, objective):
        self.X_profiling_reshape = X_profiling_reshape
        self.X_attack_reshape = X_attack_reshape
        self.X_validation_reshape = X_validation_reshape
        self.Y_profiling = Y_profiling
        self.Y_attack = Y_attack
        self.Y_validation = Y_validation
        self.validation_traces = validation_traces
        self.attacking_traces = attacking_traces
        self.labels_key_hypothesis = labels_key_hypothesis
        self.labels_key_hypothesis_attack = labels_key_hypothesis_attack
        self.leakage_model = leakage_model
        self.correct_key = correct_key
        self.batch_size = batch_size
        self.epochs = epochs
        self.project_folder = project_folder
        self.weights = []
        self.objective = objective

    def on_epoch_end(self, epoch, logs=None):
        self.weights.append(self.model.get_weights())

    def on_train_end(self, logs=None):
        if self.objective == "val_fast_ge" or self.objective == "val_ge" or self.objective == "val_geea":
            file_npz = np.load("{}/{}/ge_epochs_temp.npz".format(save_folder, project_folder), allow_pickle=True)
            self.model.set_weights(self.weights[np.argmin(file_npz["ge_epochs"])])
            metric_epochs = file_npz["ge_epochs"]
            metric_epochs_time = file_npz["ge_epochs_time"]
        else:
            file_npz = np.load("{}/{}/mi_epochs_temp.npz".format(save_folder, project_folder), allow_pickle=True)
            self.model.set_weights(self.weights[np.argmin(file_npz["mi_epochs"])])
            metric_epochs = file_npz["mi_epochs"]
            metric_epochs_time = file_npz["mi_epochs_time"]

        ge_validation_fast, sr_validation_fast, nt_validation_fast = sca_metrics(100, self.validation_traces, self.model,
                                                                                 self.X_validation_reshape,
                                                                                 self.labels_key_hypothesis, self.correct_key,
                                                                                 self.leakage_model)

        print("GE validation fast: {}".format(ge_validation_fast[self.validation_traces - 1]))
        print("SR validation fast: {}".format(sr_validation_fast[self.validation_traces - 1]))
        print("NT validation fast: {}".format(nt_validation_fast))

        ge_attack_fast, sr_attack_fast, nt_attack_fast = sca_metrics(100, self.attacking_traces, self.model, self.X_attack_reshape,
                                                                     self.labels_key_hypothesis_attack, self.correct_key,
                                                                     self.leakage_model)

        print("GE attack fast: {}".format(ge_attack_fast[self.attacking_traces - 1]))
        print("SR attack fast: {}".format(sr_attack_fast[self.attacking_traces - 1]))
        print("NT attack fast: {}".format(nt_attack_fast))

        file_count = 0
        for name in glob.glob(
                "{}/{}/fast_ge_bo_{}_{}_{}_{}_*.npz".format(save_folder, self.project_folder, dataset, self.leakage_model,
                                                            model_name, number_of_traces_key_rank)):
            file_count += 1

        np.savez("{}/{}/fast_ge_bo_{}_{}_{}_{}_{}.npz".format(save_folder, self.project_folder, dataset, self.leakage_model,
                                                              model_name, number_of_traces_key_rank, file_count + 1),
                 epochs=self.epochs,
                 batch_size=self.batch_size,
                 ge_validation_fast=ge_validation_fast,
                 sr_validation_fast=sr_validation_fast,
                 nt_validation_fast=nt_validation_fast,
                 ge_attack_fast=ge_attack_fast,
                 sr_attack_fast=sr_attack_fast,
                 nt_attack_fast=nt_attack_fast,
                 metric_epochs=metric_epochs,
                 metric_epochs_time=metric_epochs_time,
                 params=self.model.count_params(),
                 profiling_traces=profiling_traces,
                 attacking_traces=attacking_traces,
                 validation_traces=validation_traces,
                 validation_traces_total=validation_traces_total,
                 attacking_traces_total=attacking_traces_total,
                 elapsed_time=0,
                 )


def build_cnn_model(hp):
    tf_random_seed = np.random.randint(1048576)
    tf.random.set_seed(tf_random_seed)

    layers = hp.Choice('n_dense_layers', values=[1, 2, 3, 4])
    neurons = hp.Choice('neurons', values=[10, 20, 30, 40, 50, 100, 200, 300, 400, 500])
    activation = hp.Choice('activation', values=["relu", "selu", "elu"])
    learning_rate = hp.Choice("learning_rate", values=[0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001])
    optimizer = hp.Choice('optimizer', values=["Adam", "RMSprop"])
    kernel_initializer = hp.Choice('kernel_initializer', values=["random_uniform", "glorot_uniform", "he_uniform"])
    conv_layers = hp.Choice('n_conv_layers', values=[1, 2, 3, 4])
    pooling_type = hp.Choice('pooling_type', ['Average', 'Max'])
    kernels = []
    strides = []
    filters = []
    pooling_types = []
    pooling_sizes = []
    pooling_strides = []

    for conv_layer in range(1, conv_layers + 1):
        kernel = hp.Int('conv_kernel_size', min_value=4, max_value=21, step=1)
        kernels.append(kernel)
        strides.append(hp.Choice('conv_stride', values=[1, 2, 3, 4]))
        if conv_layer == 1:
            filters.append(hp.Choice('conv_filters', values=[2, 4, 8, 12, 16]))
        else:
            filters.append(filters[conv_layer - 2] * 2)
        pool_size = hp.Choice('pool_size'.format(conv_layer - 1), values=[2, 4, 6, 8, 10])
        pooling_sizes.append(pool_size)
        pooling_strides.append(pool_size)
        pooling_types.append(pooling_type)

    model = tk.models.Sequential()
    for conv_layer in range(conv_layers):
        if conv_layer == 0:
            model.add(Conv1D(filters[conv_layer], kernels[conv_layer], strides=strides[conv_layer], padding='same', input_shape=(ns, 1)))
        else:
            model.add(Conv1D(filters[conv_layer], kernels[conv_layer], strides=strides[conv_layer], padding='same'))
        model.add(Activation(activation))
        if pooling_type == 'max':
            model.add(MaxPooling1D(pooling_sizes[conv_layer], strides=pooling_strides[conv_layer], padding="same"))
        else:
            model.add(AveragePooling1D(pooling_sizes[conv_layer], strides=pooling_strides[conv_layer], padding="same"))

    model.add(Flatten())
    for l_i in range(layers - 1):
        model.add(Dense(neurons, activation=activation, kernel_initializer=kernel_initializer))

    model.add(Dense(classes, activation='softmax'))
    model.compile(optimizer=get_optimizer(optimizer, learning_rate), loss=custom_loss, metrics=metric)
    return model


def get_optimizer(optimizer, learning_rate):
    if optimizer == "Adam":
        return Adam(lr=learning_rate)
    elif optimizer == "RMSprop":
        return RMSprop(lr=learning_rate)
    elif optimizer == "Adadelta":
        return Adadelta(lr=learning_rate)
    elif optimizer == "Adagrad":
        return Adagrad(lr=learning_rate)
    elif optimizer == "SGD":
        return SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    else:
        return Adam(lr=learning_rate)


if __name__ == "__main__":
    print("Preparing data...")

    leakage_model = sys.argv[1]
    dataset = sys.argv[2]
    model_name = sys.argv[3]
    number_of_traces_key_rank = int(sys.argv[4])
    validation_traces_fast = int(sys.argv[5])
    min_id = int(sys.argv[6])
    max_id = int(sys.argv[7])
    objective = sys.argv[8]

    # leakage_model = "HW"
    # dataset = "ASCAD"
    # model_name = "cnn"
    # number_of_traces_key_rank = 50
    # min_id = 1
    # max_id = 100
    # objective = "val_fast_ge"

    filename = data_folder + "{}".format(dataset)

    epochs = 200

    if leakage_model == "HW":
        classes = 9
    else:
        classes = 256

    if dataset == "ASCAD" or dataset == "ASCAD_desync50" or dataset == "ASCAD_desync100":
        fs = 0
        ns = 700
        target_byte = 2
        correct_key = 224
        round_key = "4DFBE0F27221FE10A78D4ADC8E490469"
        attacking_traces_total = 5000
        validation_traces_total = 5000
        attacking_traces = 3000
        validation_traces = 3000
        profiling_traces = 50000
        from_output = False
    elif dataset == "ascad-variable":
        fs = 0
        ns = 1400
        target_byte = 2
        correct_key = 34
        round_key = "00112233445566778899AABBCCDDEEFF"
        attacking_traces_total = 10000
        validation_traces_total = 10000
        attacking_traces = 5000
        validation_traces = 5000
        profiling_traces = 200000
        from_output = False
    else:
        fs = 0
        ns = 4000
        target_byte = 2
        correct_key = 242
        round_key = "175cf2997a8583413c77dfac7e6c59d8"
        attacking_traces_total = 5000
        validation_traces_total = 5000
        attacking_traces = 3000
        validation_traces = 3000
        profiling_traces = 30000
        from_output = False

    key_rank_report_interval = 1
    key_rank_runs = 100
    batch_size = 500

    print("Loading dataset: {}.h5".format(dataset))

    # Load the profiling traces
    (X_profiling, X_validation, X_attack), (profiling_data, validation_data, attack_data), (
        profiling_plaintext, attack_plaintext, validation_plaintext) = load_dataset("{}.h5".format(filename),
                                                                                    profiling_traces,
                                                                                    validation_traces_total,
                                                                                    attacking_traces_total,
                                                                                    fs, ns, target_byte)

    print("Normalizing dataset...")

    # normalize with z-score
    z_score_mean, z_score_std = create_z_score_norm(X_profiling)
    apply_z_score_norm(X_profiling, z_score_mean, z_score_std)
    apply_z_score_norm(X_validation, z_score_mean, z_score_std)
    apply_z_score_norm(X_attack, z_score_mean, z_score_std)

    X_profiling = X_profiling.astype('float32')
    X_validation = X_validation.astype('float32')
    X_attack = X_attack.astype('float32')

    X_profiling_reshape = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
    X_validation_reshape = X_validation.reshape((X_validation.shape[0], X_validation.shape[1], 1))
    X_attack_reshape = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))

    print("Generating labels...")

    profiling_labels = aes_labelize(profiling_data, target_byte, leakage_model)
    validation_labels = aes_labelize(validation_data, target_byte, leakage_model)
    attack_labels = aes_labelize(attack_data, target_byte, leakage_model)

    Y_profiling = to_categorical(profiling_labels, num_classes=classes)
    Y_validation = to_categorical(validation_labels, num_classes=classes)
    Y_attack = to_categorical(attack_labels, num_classes=classes)

    # ---------------------------------------------------------------------------------------------------------#
    # compute labels for key hypothesis
    # ---------------------------------------------------------------------------------------------------------#
    # validation_traces_fast = number_of_traces_key_rank * 10
    if objective == 'val_fast_ge':
        metric = [key_rank_Metric_Fast()]
        direction = 'min'
    elif objective == 'val_ge':
        metric = [key_rank_Metric()]
        direction = 'min'
    elif objective == 'val_geea':
        metric = [geea_Metric()]
        direction = 'min'
    else:
        objective = 'val_mutual_information'
        metric = [mutual_information_Metric()]
        direction = 'max'

    labels_key_hypothesis = np.zeros((256, validation_traces_total), dtype='int64')
    for key_byte_hypothesis in range(0, 256):
        key_h = bytearray.fromhex(round_key)
        key_h[target_byte] = key_byte_hypothesis
        labels_key_hypothesis[key_byte_hypothesis][:] = aes_labelize_ge_sr(validation_data, target_byte, key_h, leakage_model)
    labels_key_hypothesis_fast = labels_key_hypothesis[:, :validation_traces_fast]

    labels_key_hypothesis_attack = np.zeros((256, attacking_traces_total), dtype='int64')
    for key_byte_hypothesis in range(0, 256):
        key_h = bytearray.fromhex(round_key)
        key_h[target_byte] = key_byte_hypothesis
        labels_key_hypothesis_attack[key_byte_hypothesis][:] = aes_labelize_ge_sr(attack_data, target_byte, key_h, leakage_model)

    start_time_full_process = time.time()

    Y_profiling = np.concatenate((Y_profiling, np.zeros((len(profiling_plaintext), 1)),
                                  profiling_plaintext.reshape((len(profiling_plaintext), 1))), axis=1)
    Y_attack = np.concatenate((Y_attack, np.ones((len(attack_plaintext), 1)),
                               attack_plaintext.reshape((len(attack_plaintext), 1))), axis=1)
    Y_validation = np.concatenate((Y_validation, np.ones((len(validation_plaintext), 1)),
                                   validation_plaintext.reshape((len(validation_plaintext), 1))), axis=1)

    now = datetime.now()
    now_str = now.strftime("%d_%m_%Y_%H_%M_%S")
    project_folder = "{}_{}_{}_{}_{}".format(dataset, leakage_model, model_name, objective, now_str)
    callback_early_stopping = Callback_EarlyStopping(X_profiling_reshape, X_attack_reshape, X_validation_reshape,
                                                     Y_profiling, Y_attack, Y_validation,
                                                     validation_traces, attacking_traces, labels_key_hypothesis,
                                                     labels_key_hypothesis_attack,
                                                     leakage_model, correct_key, batch_size, epochs, project_folder, objective)

    tuner = BayesianOptimization(build_cnn_model,
                                 objective=kt.Objective(objective, direction=direction),
                                 max_trials=max_id,
                                 executions_per_trial=1,
                                 directory=save_folder_bo,
                                 project_name=project_folder,
                                 overwrite=True)
    # tuner.on_epoch_end()

    tuner.search_space_summary()
    tuner.search(x=X_profiling_reshape,
                 y=Y_profiling,
                 epochs=epochs,
                 batch_size=batch_size,
                 validation_data=(X_validation_reshape[:validation_traces_fast], Y_validation[:validation_traces_fast]),
                 verbose=2,
                 callbacks=[callback_early_stopping])
    tuner.results_summary()

    hp_all = []
    for hp in tuner.get_best_hyperparameters(max_id):
        hp_all.append(hp.values)

    total_time = time.time() - start_time_full_process
    np.savez("{}/{}/hyperparameters_bo.npz".format(save_folder, project_folder),
             hyperparameters=hp_all, elapsed_time=total_time)
