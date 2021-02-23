import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook as tqdm
from scipy import stats

def mse(a, b):
    if isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b)
        a = np.array(a)
        b = np.array(b)
        MSE = ((a - b) **2).sum()/len(a)
    elif isinstance(a, list) and isinstance(b, float):
        a = np.array(a)
        b = np.array(b)
        MSE = ((a - b) **2).sum()/len(a)
    return MSE

def eval_dist(data:list):
    mean = np.mean(data)
    std  = np.std(data)
    mu, sigma, loc = stats.lognorm.fit(data)
    ppf = stats.lognorm(mu, sigma, loc).ppf
    NF = ppf(0.05)
    NN = ppf(0.01)
    return mean, std, NF, NN

adv_dict = torch.load("adv_dict_9874.pt")
gaussian_table = torch.load("gaussian_0.1_2D_9874.pt").to(torch.int16).numpy()
gac_dict = torch.load("gradient_accent_dict_9874.pt")
# GT = np.load("GT.npy")
filename = "log_adv_QT.txt"

acc_list = []
num_inputs = len(gaussian_table[0])
for i in range(len(gaussian_table)):
    acc_list.append(gaussian_table[i].sum() / num_inputs)
the_file = open(filename, "a+")
for i in range(len(gaussian_table)):
    acc_list.append(gaussian_table[i].sum() / num_inputs)
correct_mean, correct_std, correct_NF, correct_NN = eval_dist(acc_list)
print(f"Correct || mean: {correct_mean}, std: {correct_std}, 95: {correct_NF}, 99: {correct_NN}  ")
the_file.write(f"Correct || mean: {correct_mean}, std: {correct_std}, 95: {correct_NF}, 99: {correct_NN}  \n")
the_file.close()
# key_list = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2]
key_list = [0.2]
for key in key_list:
    the_file = open(filename + str(key), "a+")
    print(f"### {key}  ")
    the_file.write(f"### {key}  \n")
    sens_table = adv_dict[key]
    good_list = []
    bad_list = []
    for i in range(len(sens_table)):
        if sens_table[i] == 1:
            good_list.append(i)
        else:
            bad_list.append(i)
    bad_list  = np.array(bad_list)
    good_list = np.array(good_list)
    tot_list = np.array(list(range(len(sens_table))))
    p_bad  = len(bad_list)/len(sens_table)
    p_good = len(good_list)/len(sens_table)
    print(f"pGood: {p_good}, pBad: {p_bad}  ")
    the_file.write(f"pGood: {p_good}, pBad: {p_bad}  \n")
    the_file.close()
    
    result_list = []
    result_len = len(gaussian_table[0])
    # print(gaussian_table.dtype)

    num_sample_list = [10, 100, 1000, 2500]
    num_run_list = [10, 100, 1000, 2500]
    
    for num_runs in num_run_list:
        for num_samples in num_sample_list:
            the_file = open(filename + str(key), "a+")
            mean_s_list = []
            mean_r_list = []
            mean_c_list = []
            std_s_list = []
            std_r_list = []
            std_c_list = []
            NF_s_list = []
            NF_r_list = []
            NF_c_list = []
            NN_s_list = []
            NN_r_list = []
            NN_c_list = []
            
            if len(good_list) > num_samples and len(bad_list) > num_samples:
                for _ in range(10000):#tqdm(range(100), leave=False):
                    real_list = []
                    samp_list = []
                    rand_list = []
                    good_job = np.random.choice(good_list, num_samples, replace=False)
                    bad_job  = np.random.choice(bad_list, num_samples, replace=False)
                    rand_job = np.random.choice(tot_list, 2 * num_samples, replace=False)
                    gaussian_samples = np.random.choice(list(range(len(gaussian_table))), num_runs, replace=False)
                    gaussian_samples = gaussian_table[gaussian_samples,:]
                    for i in range (num_runs):
                        results = gaussian_samples[i]
                        
                        acc_good = results[good_job].sum() / num_samples
                        acc_bad  = results[bad_job].sum()  / num_samples
                        acc_rand = results[rand_job].sum() / num_samples / 2
                        # real_acc = results.sum() / result_len
                        samp_acc = p_bad*acc_bad + p_good*acc_good
                        # real_list.append(real_acc)
                        samp_list.append(samp_acc)
                        rand_list.append(acc_rand)
                    # c_mean, c_std = np.mean(real_list) , np.std(real_list)
                    # s_mean, s_std = np.mean(samp_list) , np.std(samp_list)
                    # r_mean, r_std = np.mean(rand_list) , np.std(rand_list)
                    s_mean, s_std, s_NF, s_NN = eval_dist(samp_list)
                    r_mean, r_std, r_NF, r_NN = eval_dist(rand_list)
                    mean_s_list.append(s_mean)
                    mean_r_list.append(r_mean)
                    # mean_c_list.append(c_mean)
                    std_s_list.append(s_std)
                    std_r_list.append(r_std)
                    NF_s_list.append(s_NF)
                    NF_r_list.append(r_NF)
                    NN_s_list.append(s_NN)
                    NN_r_list.append(r_NN)
                    # std_c_list.append(c_std)
                    # print(f"real: {c_mean:.4f}, samp: {s_mean:.4f}, rand: {r_mean:.4f}" + f" real: {c_std:.4f}, samp: {s_std:.4f}, rand: {r_std:.4f}")
                # s_mean_err, s_std_err = np.sqrt(mse(mean_s_list, mean_c_list)), np.sqrt(mse(std_s_list, std_c_list))
                # r_mean_err, r_std_err = np.sqrt(mse(mean_r_list, mean_c_list)), np.sqrt(mse(std_r_list, std_c_list))
                s_mean_err, s_std_err, s_NF_err, s_NN_err = np.sqrt(mse(mean_s_list, correct_mean)), np.sqrt(mse(std_s_list, correct_std)), np.sqrt(mse(NF_s_list, correct_NF)), np.sqrt(mse(NN_s_list, correct_NN))
                r_mean_err, r_std_err, r_NF_err, r_NN_err = np.sqrt(mse(mean_r_list, correct_mean)), np.sqrt(mse(std_r_list, correct_std)), np.sqrt(mse(NF_r_list, correct_NF)), np.sqrt(mse(NN_r_list, correct_NN))
                print(f"run: {num_runs:4d}, samp: {num_samples:4d} || mean, samp: {s_mean_err:.5f} / rand: {r_mean_err:.5f} || std: samp: {s_std_err:.5f} / rand: {r_std_err:.5f}  ")
                the_file.write(f"run: {num_runs:4d}, samp: {num_samples:4d} || mean, samp: {s_mean_err:.5f} / rand: {r_mean_err:.5f} || std: samp: {s_std_err:.5f} / rand: {r_std_err:.5f}  \n")
                print(f" "*25 + f"  95, samp: {s_NF_err:.5f} / rand: {r_NF_err:.5f} ||  99: samp: {s_NN_err:.5f} / rand: {r_NN_err:.5f}  ")
                the_file.write(f" "*25 + f"  95, samp: {s_NF_err:.5f} / rand: {r_NF_err:.5f} ||  99: samp: {s_NN_err:.5f} / rand: {r_NN_err:.5f}  \n")
            the_file.close()