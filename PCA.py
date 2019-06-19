import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

def load_csv_file():
    data = pd.read_csv(filepath_or_buffer='C:/Users/wedu/Desktop/Working Repository/PCA/IRIS.csv')
    return data

def main():
    dataframe = load_csv_file()
    dataframe.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
    featuresframe = dataframe.drop(axis = 1, labels = 'class')
    labelsframe = dataframe.drop(axis = 1, labels = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid'] )


    labels = labelsframe.values

    features = featuresframe.values
    features_mat= np.asarray(features)
    covariance = np.cov(features_mat.T)
    eigenvalues, eigenvectors = np.linalg.eig(covariance)

    compressed_eigen = eigenvectors[:,0:2]
    #print(compressed_eigen.T)
    #print(features_mat.T)
    eigen_space_list = np.matmul(compressed_eigen.T,features_mat.T)
    #print(eigen_space_list.T)

    final_PCA = eigen_space_list.T
    uncompressed = np.matmul(compressed_eigen,eigen_space_list)
    print(eigenvalues)
    #print(np.mean(features.T-uncompressed))
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(6, 4))
        for sample in range(120):
            color_dict = {0:'red',1:'blue',2:'green'}
            plt.scatter(final_PCA[sample,0],
                        final_PCA[sample, 1], label = labels[sample], c = color_dict[labels[sample,0]])
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
       # plt.legend(loc='lower center')
        #plt.tight_layout()
        plt.show()
if __name__ == '__main__':
    main()