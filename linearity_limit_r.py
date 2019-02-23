import numpy as np
from matplotlib import pyplot as plt
from phonecal import linearity as lin

number_of_intensities = 15

number_of_measurements_per_stack = 10

bit_number = 12
max_value = 2**bit_number - 1
saturate = 0.95 * max_value

bias = 528
read_noise = 10  # ADU

intensity_typical_error = 0.05  # %

intensities_real = np.linspace(0, 1, number_of_intensities)
digital_values = bias + intensities_real * (max_value - bias)

r_pure = lin.pearson_r_single(intensities_real, digital_values, saturate)
print(f"Pearson r on pure I, M:       {r_pure:.3f}")

intensities_observed = intensities_real + np.random.normal(0, intensity_typical_error, size=len(intensities_real)) * intensities_real
r_noisyI = lin.pearson_r_single(intensities_observed, digital_values, saturate)
print(f"Pearson r on noisy I, pure M: {r_noisyI:.3f}")

noise_ADU = np.array([np.random.normal(0, np.sqrt(D/number_of_measurements_per_stack) + read_noise) for D in digital_values])
digital_values_noisy = digital_values + noise_ADU
r_noisyIM = lin.pearson_r_single(intensities_observed, digital_values_noisy, saturate)
print(f"Pearson r on noisy I, M:      {r_noisyIM:.3f}")

plt.scatter(intensities_observed, digital_values_noisy, c='b')
plt.plot(intensities_real, digital_values, c='k')
plt.xlabel("Relative intensity")
plt.ylabel("Digital value")
plt.show()
plt.close()

def noisy_r(i, d):
    i_o = i + np.random.normal(0, intensity_typical_error, size=len(i)) * i
    n = np.array([np.random.normal(0, np.sqrt(D/number_of_measurements_per_stack) + read_noise) for D in d])
    d_n = d + n
    r = lin.pearson_r_single(i_o, d_n, saturate)
    return r

r_sample = np.array([noisy_r(intensities_real, digital_values) for j in range(25000)])
plt.hist(r_sample, bins=np.linspace(0.9, 1, 150))
plt.show()
plt.close()
print(f"Mean: {r_sample.mean():.3f} +- {r_sample.std():.3f}")
