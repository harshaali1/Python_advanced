import argparse
import time
from multiprocessing import Process, Queue
import os
import math
import numpy as np
import scipy as sp
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score, adjusted_rand_score, adjusted_mutual_info_score
from sympy.stats import Rademacher
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Type
from keras.datasets import mnist
import networkx
from networkx import grid_graph
import sgtl
import sgtl.random
import sgtl.clustering
import sgtl.graph
import sgtl.clustering
import logging
import sys
import pandas as pd

# Logging Configuration
logger = logging.getLogger('sc')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('sc.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(funcName)s() - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_formatter = logging.Formatter("%(message)s")
stdout_handler.setFormatter(stdout_formatter)
stdout_handler.setLevel(logging.INFO)
logger.addHandler(stdout_handler)


# Evaluation Functions
def clusters_to_labels(clusters, num_data_points=None):
    if num_data_points is None:
        num_data_points = sum([len(cluster) for cluster in clusters])
    labels = [0] * num_data_points
    for i, cluster in enumerate(clusters):
        for elem in cluster:
            labels[elem] = i
    return labels


def adjusted_rand_index(gt_labels, found_clusters) -> float:
    if isinstance(found_clusters[0], list):
        found_labels = clusters_to_labels(found_clusters, num_data_points=len(gt_labels))
    else:
        found_labels = found_clusters
    return adjusted_rand_score(gt_labels, found_labels)


def rand_index(gt_labels, found_clusters) -> float:
    if isinstance(found_clusters[0], list):
        found_labels = clusters_to_labels(found_clusters, num_data_points=len(gt_labels))
    else:
        found_labels = found_clusters
    return rand_score(gt_labels, found_labels)


def mutual_information(gt_labels, found_clusters) -> float:
    if isinstance(found_clusters[0], list):
        found_labels = clusters_to_labels(found_clusters, num_data_points=len(gt_labels))
    else:
        found_labels = found_clusters
    return adjusted_mutual_info_score(gt_labels, found_labels)


# Clustering Objective Functions
class ClusteringObjectiveFunction(ABC):
    @staticmethod
    @abstractmethod
    def apply(graph: sgtl.Graph, clusters: List[List[int]]) -> float:
        pass

    @staticmethod
    @abstractmethod
    def better(val_1: float, val_2: float) -> bool:
        pass


class KWayExpansion(ClusteringObjectiveFunction):
    @staticmethod
    def apply(graph: sgtl.Graph, clusters: List[List[int]]) -> float:
        try:
            conductances = []
            for cluster in clusters:
                if len(cluster) > 0:
                    conductances.append(graph.conductance(cluster))
                else:
                    conductances.append(1)
        except ZeroDivisionError:
            return 1
        return max(conductances)

    @staticmethod
    def better(val_1: float, val_2: float) -> bool:
        return val_1 < val_2


# Datasets Functions
class Dataset(object):
    def __init__(self, data_file=None, gt_clusters_file=None, graph_file=None, num_data_points=None, graph_type="knn10"):
        self.raw_data = None
        self.gt_clusters: Optional[List[List[int]]] = None
        self.gt_labels: Optional[List[int]] = None
        self.graph: Optional[sgtl.Graph] = None
        self.num_data_points = num_data_points
        if graph_file is None:
            self.load_data(data_file)
        self.load_gt_clusters(gt_clusters_file)
        self.load_graph(graph_file, graph_type=graph_type)

    @staticmethod
    def set_default_files(data_file: Optional[str], gt_clusters_file: Optional[str], graph_file: Optional[str], kwargs: Dict[str, Optional[str]]):
        if 'data_file' not in kwargs:
            kwargs['data_file'] = data_file
        if 'gt_clusters_file' not in kwargs:
            kwargs['gt_clusters_file'] = gt_clusters_file
        if 'graph_file' not in kwargs:
            kwargs['graph_file'] = graph_file
        return kwargs

    def load_data(self, data_file):
        if data_file is not None:
            self.raw_data = None

    def load_gt_clusters(self, gt_clusters_file):
        if gt_clusters_file is not None:
            self.gt_clusters = None

    def load_graph(self, graph_file=None, graph_type="knn10"):
        if graph_file is not None:
            logger.info(f"Loading edgelist graph for the {self.__class__.__name__} from {graph_file}...")
            self.graph = sgtl.graph.from_edgelist(graph_file, num_vertices=self.num_data_points)
        elif self.raw_data is not None:
            pass


# Sketching Functions
def Sketch(_type_="GA"):
    d = np.shape(A)[0]
    m = np.shape(A)[1]
    delta = 0.1
    epsilon = 0.4
    epsilon_2 = epsilon ** 2 
    if _type_ == "GA":
        GA = Gaussian_JL(alpha=1, delta=delta, d=d, m=m, eps_2=epsilon_2)
        return GA
    if _type_ == "SG":
        SG_A = Sub_Gaussian_JL(alpha=12, delta=delta, d=d, m=m, eps_2=epsilon_2)
        return SG_A
    if _type_ == "SRHT":
        Z_A = zero_padding(d=d, m=m)
        d_new = np.shape(Z_A)[0]
        SRHT_A = SRHT(Z_A=Z_A, alpha=20, delta=delta, d=d_new, m=m, eps_2=epsilon_2)
        return SRHT_A
    if _type_ == "Sp_SRHT":
        Z_A = zero_padding(d=d, m=m)
        d_new = np.shape(Z_A)[0]
        sparsity_alpha = 2
        Sp_SRHT_A = Sparse_RHT(Z_A=Z_A, alpha=20, delta=delta, sparsity_alpha=sparsity_alpha, d=d_new, m=m, eps_2=epsilon_2)
        return Sp_SRHT_A
    return


def zero_padding(d, m):
    degree = int(2 ** (np.ceil(math.log(d, 2))) - d)
    zero_pad = np.zeros((degree, m))
    t_A = np.concatenate((A, zero_pad), axis=0)
    return t_A


def Gaussian_JL(alpha, delta, d, m, eps_2):
    sketch_size = int(alpha * 12 * math.log(m / delta) / eps_2)
    S = np.random.random((sketch_size, d)) / math.sqrt(sketch_size)
    GA = S.dot(A)
    return GA


def Sub_Gaussian_JL(alpha, delta, d, m, eps_2):
    sketch_size = int(alpha * ((1 / math.sqrt(np.log(2))) ** 4) * math.log(m / delta) / eps_2)
    S = np.random.binomial(1, 0.5, size=(sketch_size, d))
    S[S == 0] = -1
    S = S / math.sqrt(sketch_size)
    SG_A = S.dot(A)
    return SG_A


def SRHT(Z_A, alpha, delta, d, m, eps_2):
    padded_dim = np.shape(Z_A)[0]
    sketch_size = int(alpha * (np.log(padded_dim / 2) ** 2) * np.log(m / delta) / eps_2)
    diag = np.random.binomial(1, 0.5, size=padded_dim)
    diag[diag == 0] = -1
    RHT = np.zeros(np.shape(Z_A))

    for i in range(m):
        RHT[:, i] = Hadamard(diag * Z_A[:, i])

    Sub_sample = np.random.choice(np.arange(padded_dim), sketch_size)
    SRHT_A = np.sqrt(padded_dim / sketch_size) * RHT[Sub_sample, :]
    return SRHT_A


def Sparse_RHT(Z_A, alpha, delta, sparsity_alpha, d, m, eps_2):
    print("Still to be coded")
    return


# Spectral Clustering Functions
def sc_precomputed_eigenvectors(eigvecs, num_clusters, num_eigenvectors):
    labels = KMeans(n_clusters=num_clusters).fit_predict(eigvecs[:, :num_eigenvectors])
    clusters = [[] for _ in range(num_clusters)]
    for idx, label in enumerate(labels):
        clusters[label].append(idx)
    return clusters


# Experiment Functions
def basic_experiment(dataset, k):
    logger.info(f"Running basic experiment with {dataset.__class__.__name__}.")

    def sub_process(num_eigenvalues: int, q):
        logger.info(f"Starting clustering: {dataset} with {num_eigenvalues} eigenvalues.")
        start_time = time.time()
        found_clusters = sgtl.clustering.spectral_clustering(dataset.graph, num_clusters=k, num_eigenvectors=num_eigenvalues)
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"Finished clustering: {dataset} with {num_eigenvalues} eigenvalues.")
        this_rand_score = adjusted_rand_index(dataset.gt_labels, found_clusters)
        this_mutual_info = mutual_information(dataset.gt_labels, found_clusters)
        this_conductance = KWayExpansion.apply(dataset.graph, found_clusters)
        q.put((num_eigenvalues, this_rand_score, this_mutual_info, this_conductance, total_time))

    rand_scores = {}
    mutual_info = {}
    conductances = {}
    times = {}
    q = Queue()
    processes = []
    for i in range(2, k + 1):
        p = Process(target=sub_process, args=(i, q))
        p.start()
        processes.append(p)

    logger.info(f"All sub-processes started for {dataset}.")

    for p in processes:
        p.join()

    logger.info(f"All sub-processes finished for {dataset}.")

    while not q.empty():
        num_vectors, this_rand_sc, this_mut_info, this_conductance, this_time = q.get()
        rand_scores[num_vectors] = this_rand_sc
        mutual_info[num_vectors] = this_mut_info
        conductances[num_vectors] = this_conductance
        times[num_vectors] = this_time

    return rand_scores, mutual_info, conductances, times


# Moved experiment_instance function out to avoid pickling error
def experiment_instance(d, nn, q):
    (train_images, train_labels), _ = mnist.load_data()
    # Use MNIST data for the clustering experiment
    this_rand_scores, this_mut_info, this_conductances, this_times = basic_experiment(
        Dataset(train_images, train_labels), nn
    )
    q.put((d, nn, this_rand_scores, this_mut_info, this_conductances, this_times))


def run_mnist_experiment():
    for k in range(3, 21):
        q = Queue()
        p = Process(target=experiment_instance, args=(None, k, q))
        p.start()
        p.join()


# Plotting Function
def plot_results():
    file_name = "cycle_results_5_1000_0.01.csv"
    df = pd.read_csv(file_name, sep=',', skipinitialspace=True)

    for num_eigen, group in df.groupby("eigenvectors"):
        plt.plot(group["poverq"], group["rand"], label=f"{num_eigen} vectors")
    plt.legend()
    plt.xlabel(r"$p/q$")
    plt.ylabel("Correctly Classified")
    plt.show()


# Main function to run experiments
def parse_args():
    parser = argparse.ArgumentParser(description='Run the experiments.')
    parser.add_argument('experiment', type=str,
                        choices=['mnist'],
                        help="which experiment to perform")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.experiment == 'mnist':
        run_mnist_experiment()

    plt.show()


if __name__ == "__main__":
    main()
