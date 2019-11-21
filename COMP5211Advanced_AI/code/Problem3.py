import os
import numpy as np
import argparse
import IPython
import csv


def main(args):
    sensors, labels = load_data(args.training_dataset)

    weight_number = sensors.shape[1]
    weight_gen_0, threshold_gen_0 = initial_generation(weight_number, args.init_sample_number)
    # print("weight initial: ", weight_gen_0)
    # print("threshold initial: ", threshold_gen_0)
    prob_threshold_init = 0.1
    rand_add_max_init = 0.3
    prob_threshold = prob_threshold_init
    rand_add_max = rand_add_max_init

    weight_gen_i = weight_gen_0
    threshold_gen_i = threshold_gen_0
    fitness_max_save = 0
    stay_generation_num = 0
    for generation_idx in range(args.max_iteration):
        shuffle_idxs = np.arange(0, weight_gen_i.shape[0])
        np.random.shuffle(shuffle_idxs)
        weight_gen_i = weight_gen_i[shuffle_idxs]
        threshold_gen_i = threshold_gen_i[shuffle_idxs]

        fitness_ratio = fitness_function(sensors, labels, weight_gen_i, threshold_gen_i)
        fitness_sort_idx = np.argsort(-1 * fitness_ratio)

        weight_gen_next = np.zeros(weight_gen_i.shape)
        threshold_gen_next = np.zeros(threshold_gen_i.shape)

        # copy 10%
        copy_ratio_set = 10
        copy_num = weight_gen_i.shape[0] // copy_ratio_set
        if copy_num < 2:
            copy_num = 2
        # for i in range(copy_num):
        #     if i == copy_num - 1:
        #         max_idx = np.argmax(fitness_ratio[i * copy_ratio_set:])
        #         weight_i = weight_gen_i[i * copy_ratio_set:]
        #         threshold_i = threshold_gen_i[i * copy_ratio_set:]
        #     else:
        #         max_idx = np.argmax(fitness_ratio[i*copy_ratio_set : (i+1)*copy_ratio_set])
        #         weight_i = weight_gen_i[i*copy_ratio_set : (i+1)*copy_ratio_set]
        #         threshold_i = threshold_gen_i[i*copy_ratio_set : (i+1)*copy_ratio_set]
        #     weight_gen_next[i] = weight_i[max_idx]
        #     threshold_gen_next[i] = threshold_i[max_idx]

        # print("weight after copy: ", weight_gen_next)
        # print("threshold after copy: ", threshold_gen_next)
        weight_gen_next[:copy_num] = weight_gen_i[fitness_sort_idx[:copy_num]]
        threshold_gen_next[:copy_num] = threshold_gen_i[fitness_sort_idx[:copy_num]]

        # crossover 90%
        corssover_num = weight_gen_next.shape[0] - copy_num
        crossover_threshold = 0.8
        for i in range(corssover_num):
            chosen_idx = np.random.choice(weight_gen_i.shape[0], copy_ratio_set * 2, replace=False)
            max_idx = np.argmax(fitness_ratio[chosen_idx[:copy_ratio_set]])
            # print("fitness_father = ", np.max(fitness_ratio[chosen_idx[:copy_ratio_set]]))
            weight_temp = weight_gen_i[chosen_idx[:copy_ratio_set]]
            threshold_temp = threshold_gen_i[chosen_idx[:copy_ratio_set]]
            weight_father = weight_temp[max_idx]
            threshold_father = threshold_temp[max_idx]

            max_idx = np.argmax(fitness_ratio[chosen_idx[copy_ratio_set:]])
            # print("fitness_mother = ", np.max(fitness_ratio[chosen_idx[copy_ratio_set:]]))
            weight_temp = weight_gen_i[chosen_idx[copy_ratio_set:]]
            threshold_temp = threshold_gen_i[chosen_idx[copy_ratio_set:]]
            weight_mother = weight_temp[max_idx]
            threshold_mother = threshold_temp[max_idx]

            # ran_num = np.random.random(2)
            # start_pos = int(ran_num[0] * weight_gen_next.shape[1])
            # seq_len = int((weight_gen_next.shape[1] - start_pos) * ran_num[1])
            # # print("start position: ", start_pos, " seq_len: ", seq_len)
            # corssover_individual = weight_father
            # corssover_individual[start_pos:start_pos+seq_len] = weight_mother[start_pos:start_pos+seq_len]
            # weight_gen_next[copy_num + i] = corssover_individual
            # # print("weight_gen_next ", copy_num + i, " = ", corssover_individual)
            # threshold_gen_next[copy_num + i] = (1 - ran_num[1]) * threshold_father + ran_num[1] * threshold_mother

            # another crossover method
            ran_num = np.random.random(weight_gen_next.shape[1])
            idx = np.where(ran_num > crossover_threshold)
            # print("crossover change ", str(idx))
            corssover_individual = weight_father
            corssover_individual[idx] = weight_mother[idx]
            weight_gen_next[copy_num + i] = corssover_individual

            ran_num = np.random.random(1)
            if (ran_num[0] > crossover_threshold):
                threshold_gen_next[copy_num + i] = threshold_mother
            else:
                threshold_gen_next[copy_num + i] = threshold_father
            # threshold_gen_next[copy_num + i] = (1 - ran_num[1]) * threshold_father + ran_num[1] * threshold_mother

        # print("weight after crossover: ", weight_gen_next)
        # print("threshold after crossover: ", threshold_gen_next)
        # mutation probability 10%


        ran_num = np.random.random(weight_gen_next.shape[0])
        mutation_chosen_idx = np.where(ran_num < prob_threshold)
        for i in range(mutation_chosen_idx[0].shape[0]):
            ran_num = np.random.random(weight_gen_next.shape[1])
            mut_pix_chosen_idx = np.where(ran_num < prob_threshold)
            for j in range(mut_pix_chosen_idx[0].shape[0]):
                rand_add = np.random.random(1)
                rand_add = rand_add_max * (0.5 - rand_add)
                weight_gen_next[mutation_chosen_idx[0][i], mut_pix_chosen_idx[0][j]] = np.clip(weight_gen_next[
                    mutation_chosen_idx[0][i], mut_pix_chosen_idx[0][j]] + rand_add, a_min=0, a_max=1)
                # print("add weight_gen_next [", mutation_chosen_idx[0][i], ", ", mut_pix_chosen_idx[0][j], "] a ", rand_add)
        # if prob_threshold
        ran_num = np.random.random(weight_gen_next.shape[0])
        mutation_chosen_idx = np.where(ran_num < prob_threshold)
        for i in range(mutation_chosen_idx[0].shape[0]):
            rand_add = np.random.random(1)
            rand_add = rand_add_max * (0.5 - rand_add)
            # rand_add = 0.2 * (0.5 - rand_add)
            threshold_gen_next[mutation_chosen_idx[0][i]] = np.clip(threshold_gen_next[mutation_chosen_idx[0][i]] + rand_add, a_min=0, a_max=1)

        # mutation replace 10%
        rand_mut_ratio_set = 3
        rand_mut_num = weight_gen_i.shape[0] // rand_mut_ratio_set
        rand_mut_weight = np.random.random((rand_mut_num, weight_gen_next.shape[1]))
        rand_mut_threshold = np.random.random(rand_mut_num)
        weight_gen_next[-rand_mut_num:] = rand_mut_weight
        threshold_gen_next[-rand_mut_num:] = rand_mut_threshold

        weight_gen_i = weight_gen_next
        threshold_gen_i = threshold_gen_next
        # print("weight: ", weight_gen_i)
        # print("threshold: ", threshold_gen_i)
        print("fitness_mean: ", np.mean(fitness_ratio), " fitness_max: ", np.max(fitness_ratio))

        fitness_max_this = np.max(fitness_ratio)

        if fitness_max_save != fitness_max_this:
            stay_generation_num = 0
        else:
            stay_generation_num += 1
        fitness_max_save = fitness_max_this

        if stay_generation_num > 2000:
            prob_threshold = prob_threshold_init + 0.1 * (stay_generation_num // 2000)
            rand_add_max = rand_add_max_init + 0.1 * (stay_generation_num // 5000)
            print("new prob_threshold is ", prob_threshold, " new rand_add_max = ", rand_add_max)
        else:
            prob_threshold = prob_threshold_init

        if np.max(fitness_ratio) == 49:
            print("Reach the best result.")
            idx = np.argmax(fitness_ratio)
            print("best weight is: ", weight_gen_i[idx])
            print("best threshold is: ", threshold_gen_i[idx])
            break

        # IPython.embed()


def fitness_function(sensors, labels, weights, thresholds):
    X = np.expand_dims(sensors, 0) * np.expand_dims(weights, 1)
    pred = np.mean(X, -1)
    thresholds_exp = np.zeros((weights.shape[0], sensors.shape[0])) + np.expand_dims(thresholds, -1)
    fit_value = (pred >= thresholds_exp)
    fit_value = fit_value.astype(int)
    labels_exp = np.zeros((weights.shape[0], sensors.shape[0])) + labels

    fitness = 1 - abs(fit_value - labels_exp)
    fitness_ratio = np.sum(fitness, -1)
    # print(fitness_ratio)

    return fitness_ratio


def initial_generation(weight_number, init_sample_number):
    return 10 * np.random.random((init_sample_number, weight_number)) - 5, 10 * np.random.random(init_sample_number) - 5


def load_data(data_path):
    with open(data_path, 'r') as f:
        data = list(csv.reader(f, delimiter=','))

    line_num = len(data)
    sensor_num = len(data[0]) - 1

    data_f = np.zeros((line_num, sensor_num))
    for i in range(line_num):
        for j in range(sensor_num):
            data_f[i, j] = float(data[i][j])

    label = np.zeros((line_num))
    for i in range(line_num):
        label[i] = data[i][-1]

    return data_f, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Path related arguments
    parser.add_argument('--training_dataset',
                        default='/home/skwang/PYProject/COMP5211Advanced_AI/project1/gp-training-set.csv')


    # Model related arguments
    parser.add_argument('--max_iteration', default=50,
                        help='max iteration while evolution.') # default 150
    # parser.add_argument('--weight_number', default=20, type=int,
    #                     help='number of the weight.')
    parser.add_argument('--init_sample_number', default=2000, type=int,
                        help='number of the procedures in the initial generation. # set n^2') # default 500

    args = parser.parse_args()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    main(args)
