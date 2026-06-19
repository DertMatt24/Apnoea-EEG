import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

from loader.patients import PatientsCSVLoader

if __name__ == '__main__':

    X_raw = np.load("C:/Users/picul/Videos/Applied/Dataset_Full/Embeddings/windowed_power_embeddings.npy")
    # shape: (78, 416, 30)

    X = (X_raw.mean(axis=2))
    # shape: (78, 416)

    X = StandardScaler().fit_transform(X)

    samples = []
    for patient_id in range(1, 40 + 1):
        for night_id in range(1, 2 + 1):
            samples.append((patient_id, night_id))
    samples.remove((8, 1))
    samples.remove((14, 2))

    df = PatientsCSVLoader.load_dataframe('C:/Users/picul/Videos/Applied/Dataset_Full/patients.csv')

    y_regression = []
    for (p_id, n_id) in samples:
        row = df[(df['user_id'] == p_id) & (df['night_id'] == n_id)]
        attacks = row.iloc[0]['AHI']
        if pd.isna(attacks) or attacks == '' or attacks == 'NaN':
            attacks = 0
        else:
            attacks = float(attacks.replace(',', '.'))
        y_regression.append(attacks)

    y = np.array(y_regression)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    print("Explained variance ratio:", pca.explained_variance_ratio_)

    # 3D scatter plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    norm = plt.Normalize(vmin=y.min(), vmax=y.max())
    colors = plt.cm.RdYlGn_r(norm(y))

    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                    c=y,
                    cmap='RdYlGn_r',
                    s=80,
                    vmin=y.min(),
                    vmax=y.max())

    for i in range(X_pca.shape[0]):
        ax.text(X_pca[i, 0], X_pca[i, 1], X_pca[i, 2], str(i), fontsize=7)

    plt.colorbar(sc, ax=ax, label='AHI', pad=0.1)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')
    ax.set_title('3D PCA of Windowed Power Embeddings (colored by AHI)')
    plt.tight_layout()
    plt.show()

    # Loadings for first 3 components
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    for i, ax in enumerate(axes):
        loadings = pca.components_[i]
        ax.plot(loadings, linewidth=0.8, color='steelblue')
        ax.axhline(0, color='black', linewidth=0.6, linestyle='--')
        ax.set_title(f'PC{i+1} Loadings ({pca.explained_variance_ratio_[i]*100:.1f}% explained variance)')
        ax.set_xlabel('Window index (0-415)')
        ax.set_ylabel('Loading')
    plt.tight_layout()
    plt.show()