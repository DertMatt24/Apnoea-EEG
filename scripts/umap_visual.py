# from skfda import FDataGrid

# from src.Filter import *
# from src.MNEDataPreparation import MNEDataPreparation
# from loader.eeg_recording import EEGLoader
from loader.patients import PatientsCSVLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
#from skfda.representation.basis import FourierBasis, BSplineBasis
# from skfda.preprocessing.dim_reduction import FPCA
import pandas as pd
import mne

from src.PatientsDSsetup import PatientsDSsetup

if __name__ == '__main__':

    mne.set_log_level('ERROR')
    completed_samples = []
    for i in range(1, 40 + 1):
        for j in range(1, 2 + 1):
            completed_samples.append((i, j))

    completed_samples.remove((8, 1))
    completed_samples.remove((14, 2))

    # patients = np.random.choice(range(1, 40), size=60, replace=False, p=None)

   # samples =
    [
        #    (patient, 1) for patient in patients
        (1, 1), (2, 2)
    ]

    X = []
    K = 4  # output dimensions per channel
    # basis = BSplineBasis(n_basis=K)

    sub_inputs = []
    ep_amt = 0
    samples = completed_samples

    # completed_samples = []
    for (p_id, n_id) in samples:

        break
        print(f"ANALYZING {p_id}, {n_id}")

        try:
            data = MNEDataPreparation.loading_data(p_id, n_id, cut=5000000)
            filtered, ica = MNEDataPreparation.cleansing_data(data, do_ica=False)
            epochs = MNEDataPreparation.creating_epochs(filtered, 60)
        except Exception:
            continue
        completed_samples.append((p_id, n_id))
        ep_amt = len(epochs)
        # Compute power spectral density
        print("NUM EPOCHS: ", ep_amt)
        spectrum = epochs.compute_psd(fmax=40.0)

        # Extract the data
        psds, freqs = spectrum.get_data(return_freqs=True)

        print(f"PSD shape: {psds.shape}")  # (n_epochs, n_channels, n_freqs)
        print(f"Freq shape: {freqs.shape}")  # (n_freqs,)

        # eeg = EEGLoader.load('C:/Users/picul/Videos/Applied/Dataset/', 1, 1)
        # print(eeg.data)

        last_shape = psds.shape[2]

        # psds shape: (n_epochs, n_channels, n_freqs)
        # freqs shape: (n_freqs,)

        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (12, 30),
            'gamma': (30, 40)
        }

        band_powers = {}
        for band_name, (f_low, f_high) in bands.items():
            # Get frequency indices in band
            band_mask = (freqs >= f_low) & (freqs <= f_high)

            # Integrate (trapz for proper integration, or just mean)
            from scipy.integrate import trapezoid

            band_power = trapezoid(psds[:, :, band_mask], freqs[band_mask], axis=2)
            # shape: (n_epochs, n_channels)

            band_powers[band_name] = band_power

        for element in band_powers.values():
            print("Shape of element: ", element.shape)

        result = np.concatenate([band_powers[k] for k in sorted(band_powers.keys())], axis=1)

        # Combine into feature matrix
        print("FEATURES SHAPE:", result.shape)
        sub_inputs.append(result)

        # ep_data: np.ndarray = epochs.get_data()
        # num_ep = ep_data.shape[0]

        # for ep in range(num_ep):
        #     for ch in range(6):
        #         sub_inputs.append(ep_data[ep, :, ch])

        """
        fd = FDataGrid(sub_inputs)

        basis_fd = fd.to_basis(basis)
        fpca = FPCA(n_components=K,  # components shown at video, no larger than original data
                    centering=True,  # Subtract the average curve of the data
                    components_basis=basis,  # Fourier FPCA
        )

        fpca.fit(basis_fd)

        # After fitting fpca
        # scores = fpca.transform(basis_fd)  # Shape: (n_subwindows, K)
        from sklearn.preprocessing import StandardScaler

        scores = fpca.transform(basis_fd)
        scores_scaled = StandardScaler().fit_transform(scores)

        # Reconstruct per EEG
        X = []

        idx = 0
        for (p_id, n_id) in samples:
            # Number of sub-windows for this EEG
            n_subwindows = ep_amt * 6  # (n_epochs * n_channels)

            # Extract scores for this EEG's sub-windows
            eeg_scores = scores[idx:idx + n_subwindows, :]  # Shape: (n_subwindows, K)

            print(eeg_scores)
            # Concatenate all sub-window scores into a single feature vector
            eeg_feature_vector = eeg_scores.flatten()  # Shape: (n_subwindows * K,)

            X.append(eeg_feature_vector)

            idx += n_subwindows
        """
    # X = np.array(sub_inputs)

    # np.save("C:/Users/picul/Videos/Applied/Dataset_Full/Embeddings/windowed_power_embeddings.npy", X)
    X = np.load("C:/Users/picul/Videos/Applied/Dataset_Full/Embeddings/windowed_power_embeddings.npy")
    # X = X.reshape(X.shape[0], -1)

    print(X.shape)
    spectral_centroids = []
    for example in X:
        # example shape: (num_windows, 30)
        num_windows = example.shape[0]
        n_channels = 6

        # indices: delta=0, theta=1, alpha=2, beta=3, gamma=4
        low_freqs = [0, 1]  # delta, theta
        high_freqs = [2, 3, 4]  # alpha, beta, gamma

        channel_ratios = np.zeros((n_channels, num_windows))

        for ch in range(n_channels):
            power_all_bands = example[:, ch::n_channels]  # (num_windows, n_bands)

            low_power = power_all_bands[:, low_freqs].sum(axis=1)
            high_power = power_all_bands[:, high_freqs].sum(axis=1)

            ratio = high_power / (low_power + 1e-8)  # avoid division by zero

            window_size = 3
            filtered = np.convolve(ratio, np.ones(window_size) / window_size, mode='same')
            channel_ratios[ch, :] = filtered

        spectral_centroids.append(channel_ratios)
    np.save("C:/Users/picul/Videos/Applied/Dataset_Full/Embeddings/power_ratios_2.npy", spectral_centroids)

    exit(0)

    spectral_centroids = []

    for example in X:
        # example shape: (num_windows, 30)
        num_windows = example.shape[0]
        n_bands = 5
        n_channels = 6
        band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        freqs = np.array([2, 6, 10, 20, 40])

        channel_centroids = np.zeros((n_channels, num_windows))

        for ch in range(n_channels):
            # Extract power for this channel across all bands and windows
            power_all_bands = example[:, ch::n_channels]  # (num_windows, n_bands)

            # Apply spectral_centroid
            centroid = np.sum(power_all_bands * freqs, axis=1) / np.sum(power_all_bands, axis=1)

            window_size = 3
            filtered = np.convolve(centroid, np.ones(window_size) / window_size, mode='same')
            channel_centroids[ch, :] = filtered

        spectral_centroids.append(channel_centroids)

    spectral_centroids = np.array(spectral_centroids)
    np.save("C:/Users/picul/Videos/Applied/Dataset_Full/Embeddings/spectral_centroids.npy", spectral_centroids)

    # Shape: (num_examples, num_channels, num_windows)

    print("Shape: ", spectral_centroids.shape)
    # plt.plot(spectral_centroids[0].T)

    # plt.show()
    exit(0)

    # Filling the samples with the couples of patient and night id
    samples = []
    for patient_id in range(1, 40 + 1):
        for night_id in range(1, 2 + 1):
            samples.append((patient_id, night_id))

    # Removing problematic samples (Totally not understandable)
    samples.remove((8, 1))
    samples.remove((14, 2))

    df = PatientsCSVLoader.load_dataframe('C:/Users/picul/Videos/Applied/Dataset_Full/patients.csv')
    # shape: (n_epochs, n_channels * 5)

    # Extract physiological features
    patient_features = []
    y_regression = []
    y_classification = []

    for (p_id, n_id) in samples:
        row = df[(df['user_id'] == p_id) & (df['night_id'] == n_id)]

        attacks = row.iloc[0]['AHI']
        if pd.isna(attacks) or attacks == '' or attacks == 'NaN':
            attacks = 0
        else:
            attacks = float(attacks.replace(',', '.'))

        # odi = row.iloc[0]['ODI']
        # if pd.isna(odi) or odi == '' or odi == 'NaN':
        #     odi = 0

        age = row.iloc[0]['age']
        sex = 1 if row.iloc[0]['sex'] == 'M' else 0
        height = row.iloc[0]['height']
        weight = row.iloc[0]['weight']
        pulse = row.iloc[0]['pulse']
        bp = row.iloc[0]['BPsys/BPdia']

        patient_features.append([age, sex, height, weight, pulse, float(bp.split('/')[0])])
        y_regression.append(attacks)
        y_classification.append(1 if attacks > 0 else 0)

    X = np.array(patient_features)

    print(X.shape)
    # X = X.reshape(X.shape[0], -1)
    import umap

    reducer = umap.UMAP(n_components=2)
    X_2d = reducer.fit_transform(np.array(X))

    # PCA instead of UMAP
    # pca = PCA(n_components=2)
    # X_2d = pca.fit_transform(X)

    #  plt.scatter(X_2d[:, 0], X_2d[:, 1])
    # plt.show()

    plt.figure(figsize=(10, 8))
    plt.scatter(X_2d[:, 0], X_2d[:, 1])

    print("Shape: ", X_2d.shape)
    # for i, (p_id, n_id) in enumerate(completed_samples):
        #plt.annotate(f"{p_id}_{n_id}", (X_2d[i, 0], X_2d[i, 1]), fontsize=8)


    for i in range(X.shape[0]):
        plt.annotate(f"{i}", (X_2d[i, 0], X_2d[i, 1]), fontsize=8)


    plt.show()
    df = PatientsCSVLoader.load_dataframe('C:/Users/picul/Videos/Applied/Dataset_Full/patients.csv')
    # shape: (n_epochs, n_channels * 5)

    # Extract attack counts for each sample
    attack_counts = []
    for (p_id, n_id) in completed_samples:
        # Query the dataframe for this patient/night pair
        row = df[(df['user_id'] == p_id) & (df['night_id'] == n_id)]

        attacks = row.iloc[0]['NAp']

        # Handle empty/NaN values
        if pd.isna(attacks) or attacks == '' or attacks == 'NaN':
            attacks = 0
        else:
            attacks = int(attacks)  # Convert to int if it's a string

        attack_counts.append(attacks)

    print(attack_counts)
    attack_counts = np.array(attack_counts)

    attack_counts = np.arange(X.shape[0])
    print("Attacks counts: ", attack_counts)

    # Plot with green-to-red colormap
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1],
                          c=attack_counts,
                          cmap='RdYlGn_r',  # Red-Yellow-Green reversed (red=high)
                          s=100,
                          vmin=0,
                          vmax=attack_counts.max())

    cbar = plt.colorbar(scatter, label='Number of Attacks')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title('UMAP of EEG FPCA Features (colored by attacks)')
    plt.show()
    # Reshape and average
