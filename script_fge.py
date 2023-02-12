import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)

from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import *
from tensorflow.keras.regularizers import *
from tensorflow.keras import *
from tensorflow.keras.callbacks import *

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
from scipy.stats import norm
import matplotlib.pyplot as plt

data_folder = "/tudelft.net/staff-umbrella/dlsca/Guilherme/"
save_folder = "/tudelft.net/staff-umbrella/dlsca/Guilherme/paper_fast_ge_only"
# data_folder = "D:/traces/"
# save_folder = "D:/postdoc/paper_fast_ge/results_fast_ge_only"

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


def load_dataset(dataset_file, n_profiling, n_validation, n_attack, first_sample, number_of_samples):
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

    return (X_profiling, X_validation, X_attack), (profiling_data, validation_data, attack_data)


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


# guessing entropy and success rate
def fast_ge(runs, nt_kr, model, x_attack, labels_key_hypothesis, correct_key, leakage_model):
    start_time = time.time()
    nt = len(x_attack)

    # ---------------------------------------------------------------------------------------------------------#
    # predict output probabilities for shuffled test or validation set
    # ---------------------------------------------------------------------------------------------------------#
    output_probabilities = np.log(model.predict(x_attack) + 1e-36)
    key_ranking_sum = 0
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
        r = np.random.choice(range(nt), nt_kr, replace=False)
        probabilities_kg_all_traces_shuffled = probabilities_kg_all_traces[r]
        key_probabilities = np.sum(probabilities_kg_all_traces_shuffled[:nt_kr], axis=0)
        key_probabilities_sorted = np.argsort(key_probabilities)[::-1]
        key_ranking_sum += list(key_probabilities_sorted).index(correct_key) + 1

    guessing_entropy = key_ranking_sum / runs
    print("GE = {}".format(guessing_entropy))

    return guessing_entropy, time.time() - start_time


# guessing entropy and success rate
def geea(runs, nt_kr, model, x_attack, labels_key_hypothesis, correct_key, leakage_model):
    start_time = time.time()
    nt = len(x_attack)

    # ---------------------------------------------------------------------------------------------------------#
    # predict output probabilities for shuffled test or validation set
    # ---------------------------------------------------------------------------------------------------------#
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
    key_probabilities = np.sum(probabilities_kg_all_traces, axis=0)

    d_kg_square = np.multiply(key_probabilities, key_probabilities)
    d_kc = d_kg_square[correct_key]
    d_kg = d_kg_square - d_kc
    mean_kg = d_kg / nt_kr

    diff = key_probabilities - key_probabilities[correct_key] - mean_kg
    var_kg = np.sqrt(np.multiply(diff, diff) / nt_kr)

    guessing_entropy = 0
    q = np.sqrt(nt_kr)
    for kg in range(256):
        if kg != correct_key:
            guessing_entropy += norm.cdf(q * mean_kg[kg] / var_kg[kg])

    print("GEEA = {}".format(256 - guessing_entropy))

    return 256 - guessing_entropy, time.time() - start_time


def get_unique_probabilities(x):
    unique_ids = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    _, unique_inverse, unique_counts = np.unique(unique_ids, return_index=False, return_inverse=True,
                                                 return_counts=True)
    return np.asarray(unique_counts / float(sum(unique_counts))), unique_inverse


def mutual_info(x_validation, y_validation):
    start_time = time.time()
    activations = model.predict(x_validation)

    num_of_bins = 100

    p_xs, unique_inverse_x = get_unique_probabilities(y_validation)

    bins = np.linspace(-1, 1, num_of_bins, dtype='float32')
    digitized = bins[np.digitize(np.squeeze(activations.reshape(1, -1)), bins) - 1].reshape(len(activations), -1)
    p_ts, _ = get_unique_probabilities(digitized)

    entropy_activations = -np.sum(p_ts * np.log(p_ts))
    entropy_activations_given_input = 0.
    for x in np.arange(len(p_xs)):
        p_t_given_x, _ = get_unique_probabilities(digitized[unique_inverse_x == x, :])
        entropy_activations_given_input += - p_xs[x] * np.sum(p_t_given_x * np.log(p_t_given_x))
    return entropy_activations - entropy_activations_given_input, time.time() - start_time


class Callback_EarlyStopping(Callback):
    def __init__(self, nt_kr, x_attack, labels_key_hypothesis, correct_key, leakage_mode, type="GE"):
        super().__init__()
        self.guessing_entropy = []
        self.guessing_entropy_time = []
        self.x_attack = x_attack
        self.nt_kr = nt_kr
        self.labels_key_hypothesis = labels_key_hypothesis
        self.correct_key = correct_key
        self.leakage_model = leakage_model
        self.min_ge = 256
        self.best_weights = None
        self.good_ge_count = 0
        self.type = type

    def on_epoch_end(self, epoch, logs=None):
        if self.type == "GE":
            ge, ge_time = fast_ge(100, self.nt_kr, self.model, self.x_attack, self.labels_key_hypothesis, self.correct_key,
                                  self.leakage_model)
        else:
            ge, ge_time = geea(100, self.nt_kr, self.model, self.x_attack, self.labels_key_hypothesis, self.correct_key, self.leakage_model)

        self.guessing_entropy.append(ge)
        self.guessing_entropy_time.append(ge_time)

        if ge < self.min_ge:
            self.best_weights = self.model.get_weights()
            self.min_ge = ge
            self.good_ge_count = 0
        else:
            if self.min_ge < 80:
                self.good_ge_count += 1

        # if self.good_ge_count == 20:
        #     self.model.stop_training = True

    def get_guessing_entropy(self):
        return self.guessing_entropy

    def get_guessing_entropy_time(self):
        return self.guessing_entropy_time

    def get_best_weights(self):
        return self.best_weights


class Callback_EarlyStopping_MI(Callback):
    def __init__(self, x_validation, y_validation):
        super().__init__()
        self.mutual_information = []
        self.mutual_information_time = []
        self.x_validation = x_validation
        self.y_validation = y_validation
        self.max_mi = 0
        self.best_weights = None
        self.type = type

    def on_epoch_end(self, epoch, logs=None):
        mi, mi_time = mutual_info(self.x_validation, self.y_validation)
        print("MI: {}".format(mi))
        self.mutual_information.append(mi)
        self.mutual_information_time.append(mi_time)

        if mi > self.max_mi:
            self.max_mi = mi
            self.best_weights = self.model.get_weights()

    def get_mutual_information(self):
        return self.mutual_information

    def get_mutual_information_time(self):
        return self.mutual_information_time

    def get_best_weights(self):
        return self.best_weights


def mlp_random(classes, number_of_samples, neurons, layers, activation, kernel_initializer, learning_rate, optimizer):
    tf_random_seed = np.random.randint(1048576)
    tf.random.set_seed(tf_random_seed)

    input_shape = (number_of_samples)
    img_input = Input(shape=input_shape)

    dense_layers = []
    for l_i in range(layers):
        if l_i == 0:
            dense_layers.append(
                Dense(neurons, activation=activation, kernel_initializer=kernel_initializer, name='dense_{}'.format(l_i),
                      input_shape=(number_of_samples,)))
        else:
            dense_layers.append(
                Dense(neurons, activation=activation, kernel_initializer=kernel_initializer, name='dense_{}'.format(l_i)))

    layer_output = []
    for l_i, dense_layer in enumerate(dense_layers):
        if l_i == 0:
            layer_output.append(dense_layer(img_input))
        else:
            layer_output.append(dense_layer(layer_output[l_i - 1]))

    output = Dense(classes, activation='softmax', name='predictions')(layer_output[layers - 1])

    inputs = img_input
    model = Model(inputs, output, name='mlp_random')
    optimizer = get_optimizer(optimizer, learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model, tf_random_seed


def cnn_random(classes, number_of_samples, neurons, layers, activation, kernel_initializer, learning_rate, optimizer, conv_layers, kernels,
               strides, filters, pooling_layers):
    tf_random_seed = np.random.randint(1048576)
    tf.random.set_seed(tf_random_seed)

    model = Sequential()
    for conv_layer in range(1, conv_layers + 1):
        if conv_layer == 1:
            model.add(
                Conv1D(kernel_size=kernels[conv_layer - 1], strides=strides[conv_layer - 1], filters=filters[conv_layer - 1],
                       activation=activation, input_shape=(number_of_samples, 1), padding="same"))
        else:
            model.add(
                Conv1D(kernel_size=kernels[conv_layer - 1], strides=strides[conv_layer - 1], filters=filters[conv_layer - 1],
                       activation=activation, padding="same"))
        model.add(pooling_layers[conv_layer - 1])
        model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(neurons, activation=activation, kernel_initializer=kernel_initializer, input_shape=(number_of_samples,)))
    for l_i in range(layers - 1):
        model.add(Dense(neurons, activation=activation, kernel_initializer=kernel_initializer))

    model.add(Dense(classes, activation='softmax'))
    model.summary()
    model.compile(optimizer=get_optimizer(optimizer, learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model, tf_random_seed


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
    min_id = int(sys.argv[5])
    max_id = int(sys.argv[6])

    # leakage_model = "HW"
    # dataset = "ches_ctf"
    # model_name = "cnn"
    # number_of_traces_key_rank = 50
    # min_id = 1
    # max_id = 100

    filename = data_folder + "{}".format(dataset)

    epochs = 100

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

    print("Loading dataset: {}.h5".format(dataset))

    # Load the profiling traces
    (X_profiling, X_validation, X_attack), (profiling_data, validation_data, attack_data) = load_dataset("{}.h5".format(filename),
                                                                                                         profiling_traces,
                                                                                                         validation_traces_total,
                                                                                                         attacking_traces_total,
                                                                                                         fs, ns)

    print("Normalizing dataset...")

    # normalize with z-score
    z_score_mean, z_score_std = create_z_score_norm(X_profiling)
    apply_z_score_norm(X_profiling, z_score_mean, z_score_std)
    apply_z_score_norm(X_validation, z_score_mean, z_score_std)
    apply_z_score_norm(X_attack, z_score_mean, z_score_std)

    X_profiling = X_profiling.astype('float32')
    X_validation = X_validation.astype('float32')
    X_attack = X_attack.astype('float32')

    if model_name == "mlp":
        X_profiling_reshape = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1]))
        X_validation_reshape = X_validation.reshape((X_validation.shape[0], X_validation.shape[1]))
        X_attack_reshape = X_attack.reshape((X_attack.shape[0], X_attack.shape[1]))
    else:
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
    labels_key_hypothesis = np.zeros((256, validation_traces_total), dtype='int64')
    for key_byte_hypothesis in range(0, 256):
        key_h = bytearray.fromhex(round_key)
        key_h[target_byte] = key_byte_hypothesis
        labels_key_hypothesis[key_byte_hypothesis][:] = aes_labelize_ge_sr(validation_data, target_byte, key_h, leakage_model)

    labels_key_hypothesis_attack = np.zeros((256, attacking_traces_total), dtype='int64')
    for key_byte_hypothesis in range(0, 256):
        key_h = bytearray.fromhex(round_key)
        key_h[target_byte] = key_byte_hypothesis
        labels_key_hypothesis_attack[key_byte_hypothesis][:] = aes_labelize_ge_sr(attack_data, target_byte, key_h, leakage_model)

    print("Generating labels... done!")

    for id in range(min_id, max_id + 1, 1):

        start_time_full_process = time.time()

        # validation_traces_fast = number_of_traces_key_rank * 10
        validation_traces_fast = 500

        print("key rank: {} of {}".format(number_of_traces_key_rank, validation_traces_fast))

        batch_size = random.randrange(100, 1100, 100)
        neurons = random.choice([10, 20, 30, 40, 50, 100, 200, 300, 400, 500])
        activation = random.choice(["relu", "selu", "elu"])
        learning_rate = random.choice([0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001])
        optimizer = random.choice(["Adam", "RMSprop"])
        kernel_initializer = random.choice(["random_uniform", "glorot_uniform", "he_uniform"])
        conv_layers = random.choice([1, 2, 3, 4])
        kernels = []
        strides = []
        filters = []
        pooling_types = []
        pooling_layers = []
        pooling_sizes = []
        pooling_strides = []

        pooling_type = random.choice(["Average", "Max"])

        for conv_layer in range(1, conv_layers + 1):
            kernel = random.randrange(4, 21, 1)
            kernels.append(kernel)
            strides.append(random.randrange(1, 4, 1))
            if conv_layer == 1:
                filters.append(random.choice([2, 4, 8, 12, 16]))
            else:
                filters.append(filters[conv_layer - 2] * 2)
            pool_size = random.choice([2, 4, 6, 8, 10])
            pooling_sizes.append(pool_size)
            pooling_strides.append(pool_size)
            pooling_types.append(pooling_type)
            if pooling_types[conv_layer - 1] == "Average":
                pooling_layers.append(
                    AveragePooling1D(pool_size=pooling_sizes[conv_layer - 1], strides=pooling_strides[conv_layer - 1], padding="same"))
            elif pooling_types[conv_layer - 1] == "Max":
                pooling_layers.append(
                    MaxPool1D(pool_size=pooling_sizes[conv_layer - 1], strides=pooling_strides[conv_layer - 1], padding="same"))
            else:
                pooling_layers.append(
                    AveragePooling1D(pool_size=pooling_sizes[conv_layer - 1], strides=pooling_strides[conv_layer - 1], padding="same"))

        if model_name == "mlp":
            layers = random.randrange(1, 9, 1)
            model, seed = mlp_random(classes, ns, neurons, layers, activation, kernel_initializer, learning_rate, optimizer)
            X_profiling_reshape = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1]))
            X_validation_reshape = X_validation.reshape((X_validation.shape[0], X_validation.shape[1]))
            X_attack_reshape = X_attack.reshape((X_attack.shape[0], X_attack.shape[1]))
        else:
            layers = random.choice([1, 2, 3, 4])
            X_profiling_reshape = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
            X_validation_reshape = X_validation.reshape((X_validation.shape[0], X_validation.shape[1], 1))
            X_attack_reshape = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))
            model, seed = cnn_random(classes, ns, neurons, layers, activation, kernel_initializer, learning_rate, optimizer, conv_layers,
                                     kernels, strides, filters, pooling_layers)

        callback_fast_ge = Callback_EarlyStopping(number_of_traces_key_rank, X_validation_reshape[:validation_traces_fast],
                                                  labels_key_hypothesis[:, :validation_traces_fast], correct_key, leakage_model)
        callback_fast_geea = Callback_EarlyStopping(validation_traces, X_validation_reshape, labels_key_hypothesis, correct_key,
                                                    leakage_model, type="GEEA")
        callback_slow_ge = Callback_EarlyStopping(validation_traces, X_validation_reshape, labels_key_hypothesis, correct_key,
                                                  leakage_model)
        callback_mi = Callback_EarlyStopping_MI(X_validation_reshape, Y_validation)

        history = model.fit(
            x=X_profiling_reshape,
            y=Y_profiling,
            batch_size=batch_size,
            verbose=2,
            epochs=epochs,
            shuffle=True,
            validation_data=(X_validation_reshape, Y_validation),
            callbacks=[callback_fast_ge, callback_fast_geea, callback_slow_ge, callback_mi])

        ge_fast_epochs = callback_fast_ge.get_guessing_entropy()
        ge_fast_time = callback_fast_ge.get_guessing_entropy_time()

        ge_geea_epochs = callback_fast_geea.get_guessing_entropy()
        ge_geea_time = callback_fast_geea.get_guessing_entropy_time()

        ge_slow_epochs = callback_slow_ge.get_guessing_entropy()
        ge_slow_time = callback_slow_ge.get_guessing_entropy_time()

        plt.plot(ge_fast_epochs)
        plt.plot(ge_geea_epochs)
        plt.plot(ge_slow_epochs)
        plt.show()

        print("GE overhead fast: {}".format(np.sum(ge_fast_time)))
        print("GE best epoch fast: {}".format(np.argmin(ge_fast_epochs)))

        print("GEEA overhead fast: {}".format(np.sum(ge_geea_time)))
        print("GEEA best epoch fast: {}".format(np.argmin(ge_geea_epochs)))

        print("GE overhead slow: {}".format(np.sum(ge_slow_time)))
        print("GE best epoch slow: {}".format(np.argmin(ge_slow_epochs)))

        print("GE overhead MI: {}".format(np.sum(callback_mi.get_mutual_information_time())))
        print("GE best epoch MI: {}".format(np.argmax(callback_mi.get_mutual_information())))

        accuracy = history.history["accuracy"]
        val_accuracy = history.history["val_accuracy"]
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]

        model.set_weights(callback_fast_ge.get_best_weights())
        ge_validation_fast, sr_validation_fast, nt_validation_fast = sca_metrics(100, validation_traces, model, X_validation_reshape,
                                                                                 labels_key_hypothesis, correct_key, leakage_model)

        print("GE validation fast: {}".format(ge_validation_fast[validation_traces - 1]))
        print("SR validation fast: {}".format(sr_validation_fast[validation_traces - 1]))
        print("NT validation fast: {}".format(nt_validation_fast))

        ge_attack_fast, sr_attack_fast, nt_attack_fast = sca_metrics(100, attacking_traces, model, X_attack_reshape,
                                                                     labels_key_hypothesis_attack, correct_key, leakage_model)

        print("GE attack fast: {}".format(ge_attack_fast[attacking_traces - 1]))
        print("SR attack fast: {}".format(sr_attack_fast[attacking_traces - 1]))
        print("NT attack fast: {}".format(nt_attack_fast))

        total_time = time.time() - start_time_full_process

        file_count = 0
        for name in glob.glob(
                "{}/fast_ge_{}_{}_{}_{}_*.npz".format(save_folder, dataset, leakage_model, model_name, number_of_traces_key_rank)):
            file_count += 1

        np.savez("{}/fast_ge_{}_{}_{}_{}_{}.npz".format(save_folder, dataset, leakage_model, model_name, number_of_traces_key_rank,
                                                        file_count + 1),
                 epochs=epochs,
                 batch_size=batch_size,
                 accuracy=accuracy,
                 val_accuracy=val_accuracy,
                 loss=loss,
                 val_loss=val_loss,
                 neurons=neurons,
                 layers=layers,
                 conv_layers=conv_layers,
                 filter=filters,
                 kernels=kernels,
                 strides=strides,
                 pooling_types=pooling_types,
                 pooling_sizes=pooling_sizes,
                 pooling_strides=pooling_strides,
                 activation=activation,
                 learning_rate=learning_rate,
                 optimizer=optimizer,
                 kernel_initializer=kernel_initializer,
                 ge_validation_fast=ge_validation_fast,
                 sr_validation_fast=sr_validation_fast,
                 nt_validation_fast=nt_validation_fast,
                 ge_attack_fast=ge_attack_fast,
                 sr_attack_fast=sr_attack_fast,
                 nt_attack_fast=nt_attack_fast,
                 ge_fast_epochs=ge_fast_epochs,
                 ge_fast_time=ge_fast_time,
                 params=model.count_params(),
                 seed=seed,
                 profiling_traces=profiling_traces,
                 attacking_traces=attacking_traces,
                 validation_traces=validation_traces,
                 validation_traces_total=validation_traces_total,
                 attacking_traces_total=attacking_traces_total,
                 elapsed_time=total_time,
                 )
