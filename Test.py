
from Model.GenerateTestData import GenerateDataFeatureBased
from sklearn.metrics.cluster import normalized_mutual_info_score
import time
from sklearn.cluster import KMeans, SpectralClustering
from Model.DataSetFeatureBased import DataSetFeatureBased
from Model.TangleCluster import *
from Model.SearchTree import *
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="Graph is not fully connected")


class Test:
    def __init__(self):
        print("Test class constructor")


    def spectral(self, points, ground_truth, k):
        print("start")
        start_time = time.time()
        spectral = SpectralClustering(n_clusters=k)
        spectral.fit_predict(points)
        
        end_time = time.time()
        
        total_time = end_time - start_time
        try:
            nmi_score = normalized_mutual_info_score(ground_truth, spectral.labels_)
        except:
            nmi_score = 0

        return total_time, nmi_score, spectral.labels_

    def kmeans(self, points, ground_truth, k):
        
        
        start_time = time.time()

        kmeans = KMeans(n_clusters=k)
        kmeans.fit(points)
        
        end_time = time.time()

        total_time = end_time - start_time

        try:
            nmi_score = normalized_mutual_info_score(ground_truth, kmeans.labels_)
        except:
            nmi_score = 0

        return total_time, nmi_score, kmeans.labels_
    
    def tangles(self, points, ground_truth, a):
         # Creating the tangles
        tangle = DataSetFeatureBased(a)

        tangle.points = points

        start_time = time.time()
        
        tangle.mean_cut()
        print("cuts")
        tangle.mean_cost()
        print("costs")
        root = create_searchtree(tangle)
        tangle_root = condense_tree(root)
        contracting_search_tree(tangle_root)

        if tangle_root.numb_tangles == 0:
            print("No tangles found")
        
        soft = soft_clustering(tangle_root)
        hard = hard_clustering(soft)
    

        end_time = time.time()

        total_time = end_time - start_time
        try: 
            nmi_score = normalized_mutual_info_score(ground_truth, hard)
        except:
            nmi_score = 0
        return total_time, nmi_score, hard

    def tangles2(self, points, ground_truth, a):
         # Creating the tangles
        tangle = DataSetFeatureBased(a)

        tangle.points = points

        start_time = time.time()
        
        tangle.cut_generator_axis_dimensions()
        print("cuts")
        tangle.mean_cost()
        print("costs")

        root = create_searchtree(tangle)
        print("searchtree")
        tangle_root = condense_tree(root)
        contracting_search_tree(tangle_root)
        print("contracting")
        if tangle_root.numb_tangles == 0:
            print("No tangles found")
        
        soft = soft_clustering(tangle_root)
        hard = hard_clustering(soft)
        print("clustering")

        end_time = time.time()

        total_time = end_time - start_time
        try: 
            nmi_score = normalized_mutual_info_score(ground_truth, hard)
        except:
            nmi_score = 0
        return total_time, nmi_score, hard
    
    def tangles3(self, points, ground_truth, a):
         # Creating the tangles
        tangle = DataSetFeatureBased(a)

        tangle.points = points

        start_time = time.time()
        
        tangle.adjusted_cut()
        tangle.mean_cost()


        root = create_searchtree(tangle)
        tangle_root = condense_tree(root)
        contracting_search_tree(tangle_root)

        if tangle_root.numb_tangles == 0:
            print("No tangles found")
        
        soft = soft_clustering(tangle_root)
        hard = hard_clustering(soft)
    

        end_time = time.time()

        total_time = end_time - start_time
        try: 
            nmi_score = normalized_mutual_info_score(ground_truth, hard)
        except:
            nmi_score = 0
        return total_time, nmi_score, hard
    
    def tangles_time_checking(self, points, ground_truth, a):
         # Creating the tangles
        
        print("________")
        tangle = DataSetFeatureBased(a)

        tangle.points = points

        start_time = time.time()
        
        cut_start = time.time()
        tangle.cut_axis()
        cut_end = time.time()
        print(f"Cut generator time: {cut_end - cut_start:.5f}")

        cost_start = time.time()
        tangle.mean_cost()
        cost_end = time.time()
        print(f"Cost function time: {cost_end - cost_start:.5f}")

        root_start = time.time()
        root = create_searchtree(tangle)
        root_end = time.time()
        print(f"Create searchtree time: {root_end - root_start:.5f}")

        condense_start = time.time()
        tangle_root = condense_tree(root)
        condense_end = time.time()
        print(f"Condense tree time: {condense_end - condense_start:.5f}")

        contracting_start = time.time()
        contracting_search_tree(tangle_root)
        contracting_end = time.time()
        print(f"Contracting search tree time: {contracting_end - contracting_start:.5f}")

        if tangle_root.numb_tangles == 0:
            print("No tangles found")
        
        soft_start = time.time()
        soft = soft_clustering(tangle_root)
        soft_end = time.time()
        print(f"Soft clustering time: {soft_end - soft_start:.5f}")
        
        hard_start = time.time()
        hard = hard_clustering(soft)
        hard_end = time.time()
        print(f"Hard clustering time: {hard_end - hard_start:.5f}")

        end_time = time.time()

        total_time = end_time - start_time
        try: 
            nmi_score = normalized_mutual_info_score(ground_truth, hard)
        except:
            nmi_score = 0
        return total_time, nmi_score, hard

    def random_data(self, numb_clusters, std, numb_points, dimension, overlap):
        data = GenerateDataFeatureBased(numb_clusters, std)
        data.random_clusters(numb_points, dimension, overlap)
        return data

    def random_data_different(self, numb_clusters, std, numb_points, dimension, value_range, overlap):
        data = GenerateDataFeatureBased(numb_clusters, std)
        data.random_different_clusters(numb_points, dimension, value_range, overlap)
        return data


    def fixed_data(self, numb_clusters, std, numb_points, centroids):
        data = GenerateDataFeatureBased(numb_clusters, std)
        data.fixed_clusters(numb_points, centroids)
        return data


    def kmeans_run_perf_test(self, data, k):
        kmeans_time = []
        kmeans_nmi = []
        for _ in range(10):
            time, nmi_score,_ = self.kmeans(data.points, data.ground_truth, k)
            kmeans_nmi.append(nmi_score)
            kmeans_time.append(time)
        print(f"mean kmeans_nmi: {sum(kmeans_nmi)/len(kmeans_nmi)}")
        print(f"mean kmeans_time: {sum(kmeans_time)/len(kmeans_time)}")
    
    def spectral_run_perf_test(self, data, k):
        spectral_time = []
        spectral_nmi = []
        for _ in range(10):
            time, nmi_score,_ = self.spectral(data.points, data.ground_truth, k)
            spectral_nmi.append(nmi_score)
            spectral_time.append(time)
        print(f"mean spectral_nmi: {sum(spectral_nmi)/len(spectral_nmi)}")
        print(f"mean spectral_time: {sum(spectral_time)/len(spectral_time)}")
    
    def tangles_run_perf_test(self, data, a):
        tangles_time = []
        tangles_nmi = []
        for _ in range(10):
            time, nmi_score,_ = self.tangles(data.points, data.ground_truth, a)
            tangles_nmi.append(nmi_score)
            tangles_time.append(time)
        print(f"mean tangles_nmi: {sum(tangles_nmi)/len(tangles_nmi)}")
        print(f"mean tangles_time: {sum(tangles_time)/len(tangles_time)}")

    def tester(self, data, a, k):
        tangles_nmi = []
        tangles_time = []
        spectral_nmi = []
        spectral_time = []
        kmeans_nmi = []
        kmeans_time = []

        for _ in range(10):        
            # Tangles
            time, nmi_score,_ = self.tangles(data.points, data.ground_truth, a)
            tangles_nmi.append(nmi_score)
            tangles_time.append(time)


            # Spectral clustering
            time, nmi_score,_ = self.spectral(data.points, data.ground_truth, k)
            spectral_nmi.append(nmi_score)
            spectral_time.append(time)

            # Kmeans clustering
            time, nmi_score,_ = self.kmeans(data.points, data.ground_truth, k)
            kmeans_nmi.append(nmi_score)
            kmeans_time.append(time)

        print(f"mean tangles_nmi: {sum(tangles_nmi)/len(tangles_nmi)}")
        print(f"mean tangles_time: {sum(tangles_time)/len(tangles_time)}")
        print(f"mean spectral_nmi: {sum(spectral_nmi)/len(spectral_nmi)}")
        print(f"mean spectral_time: {sum(spectral_time)/len(spectral_time)}")
        print(f"mean kmeans_nmi: {sum(kmeans_nmi)/len(kmeans_nmi)}")
        print(f"mean kmeans_time: {sum(kmeans_time)/len(kmeans_time)}")


def k_mean_tester(noisy_circles, noisy_moons, blobs, no_structure, aniso, varied):
    test = Test()
    noisy_circles_time = []
    noisy_circles_nmi = []
    noisy_moons_time = []
    noisy_moons_nmi = []
    blobs_time = []
    blobs_nmi = []
    no_structure_time = []
    no_structure_nmi = []
    aniso_time = []
    aniso_nmi = []
    varied_time = []
    varied_nmi = []

    numb = 10

    data_points = [noisy_circles[0], noisy_moons[0], blobs[0], no_structure[0], aniso[0], varied[0]]
    plot_names = ["Noisy circles", "Noisy moons", "Blobs", "No structure", "Aniso", "Varied"]

    for i in range(numb):
        t, nmi_score,labels_circle = test.kmeans(noisy_circles[0], noisy_circles[1], 2)
        noisy_circles_time.append(t)
        noisy_circles_nmi.append(nmi_score)

        t, nmi_score,labels_moon = test.kmeans(noisy_moons[0], noisy_moons[1], 2)
        noisy_moons_time.append(t)
        noisy_moons_nmi.append(nmi_score)

        t, nmi_score,labels_blobs = test.kmeans(blobs[0], blobs[1], 3)
        blobs_time.append(t)
        blobs_nmi.append(nmi_score)

        t, nmi_score,labels_no = test.kmeans(no_structure[0], no_structure[1], 3)
        no_structure_time.append(t)

        t, nmi_score,labels_aniso = test.kmeans(aniso[0], aniso[1], 3)
        aniso_time.append(t)
        aniso_nmi.append(nmi_score)

        t, nmi_score, labels_varied = test.kmeans(varied[0], varied[1], 3)
        varied_time.append(t)
        varied_nmi.append(nmi_score)

        if i == numb-1: 
            ground_truth = [labels_circle, labels_moon, labels_blobs, labels_no, labels_aniso, labels_varied]

            fig, axs = plt.subplots(1, 6, figsize=(18, 3))
            for i, (ax, points, labels) in enumerate(zip(axs, data_points, ground_truth), 1):
                ax.scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis')  # Assuming points is an array-like object with x and y values
                ax.set_title(f'{plot_names[i-1]}')
            
            plt.suptitle("K-means")  
            plt.tight_layout()  # Adjust subplot parameters to give specified padding
            plt.show()

    print("-----Kmeans-----")
    print("Noisy circles time: ", sum(noisy_circles_time)/len(noisy_circles_time))
    print("Noisy circles nmi: ", sum(noisy_circles_nmi)/len(noisy_circles_nmi))
    print("Noisy moons time: ", sum(noisy_moons_time)/len(noisy_moons_time))
    print("Noisy moons nmi: ", sum(noisy_moons_nmi)/len(noisy_moons_nmi))
    print("Blobs time: ", sum(blobs_time)/len(blobs_time))
    print("Blobs nmi: ", sum(blobs_nmi)/len(blobs_nmi))
    print("No structure time: ", sum(no_structure_time)/len(no_structure_time))
    print("Aniso time: ", sum(aniso_time)/len(aniso_time))
    print("Aniso nmi: ", sum(aniso_nmi)/len(aniso_nmi))
    print("Varied time: ", sum(varied_time)/len(varied_time))
    print("Varied nmi: ", sum(varied_nmi)/len(varied_nmi))

def spectral_tester(noisy_circles, noisy_moons, blobs, no_structure, aniso, varied):
    test = Test()
    noisy_circles_time = []
    noisy_circles_nmi = []
    noisy_moons_time = []
    noisy_moons_nmi = []
    blobs_time = []
    blobs_nmi = []
    no_structure_time = []
    no_structure_nmi = []
    aniso_time = []
    aniso_nmi = []
    varied_time = []
    varied_nmi = []

    numb = 10

    data_points = [noisy_circles[0], noisy_moons[0], blobs[0], no_structure[0], aniso[0], varied[0]]
    plot_names = ["Noisy circles", "Noisy moons", "Blobs", "No structure", "Aniso", "Varied"]

    for i in range(10):
        t, nmi_score, labels_circle = test.spectral(noisy_circles[0], noisy_circles[1], 2)
        noisy_circles_time.append(t)
        noisy_circles_nmi.append(nmi_score)

        t, nmi_score, labels_moon  = test.spectral(noisy_moons[0], noisy_moons[1], 2)
        noisy_moons_time.append(t)
        noisy_moons_nmi.append(nmi_score)

        t, nmi_score, labels_blobs = test.spectral(blobs[0], blobs[1], 3)
        blobs_time.append(t)
        blobs_nmi.append(nmi_score)

        t, nmi_score, labels_no = test.spectral(no_structure[0], no_structure[1], 3)
        no_structure_time.append(t)

        t, nmi_score, labels_aniso = test.spectral(aniso[0], aniso[1], 3)
        aniso_time.append(t)
        aniso_nmi.append(nmi_score)

        t, nmi_score, labels_varied = test.spectral(varied[0], varied[1], 3)
        varied_time.append(t)
        varied_nmi.append(nmi_score)

        if i == numb-1: 
            ground_truth = [labels_circle, labels_moon, labels_blobs, labels_no, labels_aniso, labels_varied]

            fig, axs = plt.subplots(1, 6, figsize=(18, 3))
            for i, (ax, points, labels) in enumerate(zip(axs, data_points, ground_truth), 1):
                ax.scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis')  # Assuming points is an array-like object with x and y values
                ax.set_title(f'{plot_names[i-1]}')
            
            plt.suptitle("Spectral")  
            plt.tight_layout()  # Adjust subplot parameters to give specified padding
            plt.show()

    print("-----Spectral-----")
    print("Noisy circles time: ", sum(noisy_circles_time)/len(noisy_circles_time))
    print("Noisy circles nmi: ", sum(noisy_circles_nmi)/len(noisy_circles_nmi))
    print("Noisy moons time: ", sum(noisy_moons_time)/len(noisy_moons_time))
    print("Noisy moons nmi: ", sum(noisy_moons_nmi)/len(noisy_moons_nmi))
    print("Blobs time: ", sum(blobs_time)/len(blobs_time))
    print("Blobs nmi: ", sum(blobs_nmi)/len(blobs_nmi))
    print("No structure time: ", sum(no_structure_time)/len(no_structure_time))
    print("Aniso time: ", sum(aniso_time)/len(aniso_time))
    print("Aniso nmi: ", sum(aniso_nmi)/len(aniso_nmi))
    print("Varied time: ", sum(varied_time)/len(varied_time))
    print("Varied nmi: ", sum(varied_nmi)/len(varied_nmi))

def tangles_tester(noisy_circles, noisy_moons, blobs, no_structure, aniso, varied):
    test = Test()
    noisy_circles_time = []
    noisy_circles_nmi = []
    noisy_moons_time = []
    noisy_moons_nmi = []
    blobs_time = []
    blobs_nmi = []
    no_structure_time = []
    no_structure_nmi = []
    aniso_time = []
    aniso_nmi = []
    varied_time = []
    varied_nmi = []

    numb = 1

    data_points = [noisy_circles[0], noisy_moons[0], blobs[0], no_structure[0], aniso[0], varied[0]]
    plot_names = ["Noisy circles", "Noisy moons", "Blobs", "No structure", "Aniso", "Varied"]

    #data_points = [varied[0]]
    for i in range(numb):
        t, nmi_score, labels_circle = test.tangles(noisy_circles[0], noisy_circles[1], int(750*0.90))
        noisy_circles_time.append(t)
        noisy_circles_nmi.append(nmi_score)
       
        t, nmi_score, labels_moon = test.tangles(noisy_moons[0], noisy_moons[1], int(750*0.90))
        noisy_moons_time.append(t)
        noisy_moons_nmi.append(nmi_score)

        t, nmi_score, labels_blobs = test.tangles(blobs[0], blobs[1], int(500*0.95))
        blobs_time.append(t)
        blobs_nmi.append(nmi_score)

        t, nmi_score, labels_no = test.tangles(no_structure[0], no_structure[1], int(500*0.95))
        no_structure_time.append(t)

        t, nmi_score, labels_aniso = test.tangles(aniso[0], aniso[1], int(500*0.95))
        aniso_time.append(t)
        aniso_nmi.append(nmi_score)

        t, nmi_score, labels_varied = test.tangles(varied[0], varied[1], int(500*0.95))
        varied_time.append(t)
        varied_nmi.append(nmi_score)

        if i == numb-1: 
            
            ground_truth = [labels_circle, labels_moon, labels_blobs, labels_no, labels_aniso, labels_varied]

            fig, axs = plt.subplots(1, 6, figsize=(18, 3))
            for i, (ax, points, labels) in enumerate(zip(axs, data_points, ground_truth), 1):
                ax.scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis')  # Assuming points is an array-like object with x and y values
                ax.set_title(f'{plot_names[i-1]}')
            
            plt.suptitle("Tangle")  
            plt.tight_layout()  # Adjust subplot parameters to give specified padding
            plt.show()

    print("-----Tangles-----")
    print("Noisy circles time: ", sum(noisy_circles_time)/len(noisy_circles_time))
    print("Noisy circles nmi: ", sum(noisy_circles_nmi)/len(noisy_circles_nmi))
    print("Noisy moons time: ", sum(noisy_moons_time)/len(noisy_moons_time))
    print("Noisy moons nmi: ", sum(noisy_moons_nmi)/len(noisy_moons_nmi))
    print("Blobs time: ", sum(blobs_time)/len(blobs_time))
    print("Blobs nmi: ", sum(blobs_nmi)/len(blobs_nmi))
    print("No structure time: ", sum(no_structure_time)/len(no_structure_time))
    print("Aniso time: ", sum(aniso_time)/len(aniso_time))
    print("Aniso nmi: ", sum(aniso_nmi)/len(aniso_nmi))
    print("Varied time: ", sum(varied_time)/len(varied_time))
    print("Varied nmi: ", sum(varied_nmi)/len(varied_nmi))



def iris_tester():
    import pandas as pd
    from sklearn.manifold import TSNE

    df = pd.read_csv('iris.csv')
    X = df.values
    y = X[:, -1]
    X = X[:, :-1]
    X = X.astype(float)

    instances = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

    # Create a dictionary to map each instance to a numerical value
    instance_to_number = {instance: i for i, instance in enumerate(instances)}

    # Convert the original list to numerical values
    numerical_list = [instance_to_number[instance] for instance in y]
    

    test = Test()
    time, nmi, hard = test.tangles(X, numerical_list, 50)
    
    time_kmeans, nmi_kmeans, hard_kmeans = test.kmeans(X, numerical_list, 3)

    time_spectral, nmi_spectral, hard_spectral = test.spectral(X, numerical_list, 3)


    print(f"Time: {time}")
    print(f"NMI: {nmi}")
    print(f"time_kmeans: {time_kmeans}")
    print(f"nmi_kmeans: {nmi_kmeans}")
    print(f"time_spectral: {time_spectral}")
    print(f"nmi_spectral: {nmi_spectral}")

    perplexity = min(20, len(X) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity)
    data = tsne.fit_transform(X)


    x = [point[0] for point in data]
    y = [point[1] for point in data]

    fig, axs = plt.subplots(1, 4, figsize=(10, 5))

    # Plot the first graph
    axs[0].scatter(x, y, c=numerical_list, cmap='viridis')
    axs[0].set_title('Ground truth')

    # Plot the second graph
    axs[1].scatter(x, y, c=hard, cmap='viridis')
    axs[1].set_title('Tangles')

    axs[2].scatter(x, y, c=hard_kmeans, cmap='viridis')
    axs[2].set_title('kmeans')

    axs[3].scatter(x, y, c=hard_spectral, cmap='viridis')
    axs[3].set_title('spectral')

    # Show the plot
    plt.tight_layout()
    plt.show()

def cost_tester():
    j = 4
    a = [50, 100, 200, 400, 800, 1600 ]
    t_mean_overall = []
    t_pair_overall = []
    t_cure_overall = []
    nmi_mean_overall = []
    nmi_pair_overall = []
    nmi_cure_overall = []
    k = 0
    while k < 6:
        t_mean = []
        t_pair = []
        t_cure = []
        score_mean = []
        score_pair = []
        score_cure = []
        for i in range(10):
            test = Test()
            data = test.random_data(j, 1.5, a[k], 2, 0.7)
            time_mean, nmi_mean, _ = test.tangles(data.points, data.ground_truth, a[k])
            time_pair, nmi_pair, _ = test.tangles2(data.points, data.ground_truth, a[k])
            time_cure, nmi_cure, _ = test.tangles3(data.points, data.ground_truth, a[k])
            t_mean.append(time_mean)
            t_pair.append(time_pair)
            t_cure.append(time_cure)
            score_mean.append(nmi_mean)
            score_pair.append(nmi_pair)
            score_cure.append(nmi_cure)

        t_mean_overall.append(sum(t_mean)/len(t_mean))
        t_pair_overall.append(sum(t_pair)/len(t_pair))
        t_cure_overall.append(sum(t_cure)/len(t_cure))
        nmi_mean_overall.append(sum(score_mean)/len(score_mean))
        nmi_pair_overall.append(sum(score_pair)/len(score_pair))
        nmi_cure_overall.append(sum(score_cure)/len(score_cure))

        k += 1
        print("t_mean_overall: ", t_mean_overall)
        print("t_pair_overall: ", t_pair_overall)
        print("t_cure_overall: ", t_cure_overall)
        print("nmi_mean_overall: ", nmi_mean_overall)
        print("nmi_pair_overall: ", nmi_pair_overall)
        print("nmi_cure_overall: ", nmi_cure_overall)
        print(a[k])

    

def toyTesting():
    # test = Test()
    # test.tester(4, 1, 100, 2, 0.5, 100, 4)
    
    import time
    import warnings
    from itertools import cycle, islice

    import matplotlib.pyplot as plt
    import numpy as np

    from sklearn import cluster, datasets
    from sklearn.preprocessing import StandardScaler


    n_samples = 1500
    noisy_circles = datasets.make_circles(
        n_samples=n_samples, factor=0.5, noise=0.05, random_state=170
    )
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=170)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=170)
    rng = np.random.RandomState(170)
    no_structure = rng.rand(n_samples, 2), None

    # Anisotropicly distributed data
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=170)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = datasets.make_blobs(
        n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=170
    )


    #k_mean_tester(noisy_circles, noisy_moons, blobs, no_structure, aniso, varied)
    #spectral_tester(noisy_circles, noisy_moons, blobs, no_structure, aniso, varied)
    tangles_tester(noisy_circles, noisy_moons, blobs, no_structure, aniso, varied)


def numb_point_tester():
    k = 4
    a = 25
    t_tangle_overall = []
    t_kmean_overall = []
    t_spectral_overall = []
    nmi_tangle_overall = []
    nmi_kmean_overall = []
    nmi_spectral_overall = []

    while a < 1000000:
        t_tangle = []
        t_kmean = []
        t_spectral = []
        score_tangle = []
        score_kmean = []
        score_spectral = []
        for i in range(10):
            test = Test()
            data = test.random_data(k, 1.5, a, 2, 0.7)
            time_tangle, nmi_tangle, _ = test.tangles(data.points, data.ground_truth, int(a*0.95))
            time_kmean, nmi_kmean, _ = test.kmeans(data.points, data.ground_truth, k)
            if a < 10000:
                time_spectral, nmi_spectral, _ = test.spectral(data.points, data.ground_truth, k)
                t_spectral.append(time_spectral)
                score_spectral.append(nmi_spectral)
            t_tangle.append(time_tangle)
            t_kmean.append(time_kmean) 
            score_tangle.append(nmi_tangle)
            score_kmean.append(nmi_kmean)
            
        t_tangle_overall.append(sum(t_tangle)/len(t_tangle))
        t_kmean_overall.append(sum(t_kmean)/len(t_kmean))
        if a < 10000:
            t_spectral_overall.append(sum(t_spectral)/len(t_spectral))
            nmi_spectral_overall.append(sum(score_spectral)/len(score_spectral))
        nmi_tangle_overall.append(sum(score_tangle)/len(score_tangle))
        nmi_kmean_overall.append(sum(score_kmean)/len(score_kmean))
        
        a *= 10
        print("t_tangle_overall: ", t_tangle_overall)
        print("t_kmean_overall: ", t_kmean_overall)
        print("t_spectral_overall: ", t_spectral_overall)
        print("nmi_tangle_overall: ", nmi_tangle_overall)
        print("nmi_kmean_overall: ", nmi_kmean_overall)
        print("nmi_spectral_overall: ", nmi_spectral_overall)
        print(a)

def create_bar_plot():
    import matplotlib.pyplot as plt
    import numpy as np

    # Sample data
    x = np.array([1, 2, 3, 4, 5])  # X-axis points
    k_mean_time = np.array([0.025053787231445312, 0.018529987335205077, 0.028957438468933106, 25, 30])  # Y-axis data for first bar
    tangles_time = np.array([0.0053331851959228516, 0.026862406730651857, 0.21382060050964355, 27, 32])  # Y-axis data for second bar
    spectral_time = np.array([0.022859859466552734, 0.04334642887115479, 1.2422450065612793, 23, 28])   # Y-axis data for third bar
    
    k_mean_nmi = []
    tangles_nmi = []
    spectral_nmi = []

    # Width of each bar
    bar_width = 0.2

    # Plotting
    plt.bar(x - bar_width, k_mean_time, width=bar_width, color='b', align='center', label='Bar 1')
    plt.bar(x, tangles_time, width=bar_width, color='g', align='center', label='Bar 2')
    plt.bar(x + bar_width, spectral_time, width=bar_width, color='r', align='center', label='Bar 3')

    # Adding labels and title
    plt.xlabel('number of datapoints')
    plt.ylabel('runtime (s)')
    plt.xticks(x, ['4*10^1', '4*10^2', '4*10^3', '4*10^4', '4*10^5'])  # Customizing x-axis labels
    plt.legend()

    # Display plot  
    plt.show()
    
def cut_tester():
    k = 4
    numb_points = 100
    
    t_mean_overall = []
    t_pair_overall = []
    t_cure_overall = []
    nmi_mean_overall = []
    nmi_pair_overall = []
    nmi_cure_overall = []

    while numb_points < 1000000:
        t_mean = []
        t_pair = []
        t_cure = []
        score_mean = []
        score_pair = []
        score_cure = []
        a  = numb_points
        for i in range(10):
            test = Test()
            data = test.random_data(k, 1.5, numb_points, 2, 0.7)
            time_mean, nmi_mean, _ = test.tangles(data.points, data.ground_truth, int(numb_points * 0.95))
            time_pair, nmi_pair, _ = test.tangles2(data.points, data.ground_truth, a)
            time_cure, nmi_cure, _ = test.tangles3(data.points, data.ground_truth, int(numb_points * 0.95))
            t_mean.append(time_mean)
            t_pair.append(time_pair)
            t_cure.append(time_cure)
            score_mean.append(nmi_mean)
            score_pair.append(nmi_pair)
            score_cure.append(nmi_cure)

        t_mean_overall.append(sum(t_mean)/len(t_mean))
        t_pair_overall.append(sum(t_pair)/len(t_pair))
        t_cure_overall.append(sum(t_cure)/len(t_cure))
        nmi_mean_overall.append(sum(score_mean)/len(score_mean))
        nmi_pair_overall.append(sum(score_pair)/len(score_pair))
        nmi_cure_overall.append(sum(score_cure)/len(score_cure))

        numb_points *= 10
        print("t_mean_overall: ", t_mean_overall)
        print("t_axis_overall: ", t_pair_overall)
        print("t_adjusted_overall: ", t_cure_overall)
        print("nmi_mean_overall: ", nmi_mean_overall)
        print("nmi_axis_overall: ", nmi_pair_overall)
        print("nmi_adjusted_overall: ", nmi_cure_overall)
    
    # Define the x and y coordinates of the points for Line 1
    x1 = [1, 2, 3, 4]
    x2 = [1, 2, 3, 4]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the first set of data on the first subplot
    axs[0].plot(x1, t_mean_overall, marker='o', linestyle='-', label='mean-cut')
    axs[0].plot(x1, t_pair, marker='o', linestyle='--', label='parallel-cut')
    axs[0].plot(x2, t_cure_overall, marker='o', linestyle='--', label='adjusted-cut')
    axs[0].set_xticks([1, 2, 3, 4])
    axs[0].set_xticklabels(['4*10^2', '4*10^3', '4*10^4', '4*10^5'])
    axs[0].set_yscale('log')

    axs[0].set_xlabel('number of points')
    axs[0].set_ylabel('log run-time (s)')
    axs[0].set_title('log run-time by number of points')
    axs[0].legend()

    axs[1].plot(x1, nmi_mean_overall, marker='o', linestyle='-', label='mean-cut')
    axs[1].plot(x1, nmi_pair_overall, marker='o', linestyle='--', label='parallel-cut')
    axs[1].plot(x2, nmi_cure_overall, marker='o', linestyle='--', label='adjusted-cut')
    axs[1].set_xticks([1, 2, 3, 4])
    axs[1].set_xticklabels(['4*10^2', '4*10^3', '4*10^3', '4*10^4'])
    axs[1].set_xlabel('number of points')
    axs[1].set_ylabel('NMI-score')
    axs[1].set_title('NMI-score by number of points')
    axs[1].legend()

    # Adjust layout to prevent overlapping of labels
    plt.tight_layout()

    # Display the plots
    plt.show()


def plots_cost_tester():
    import matplotlib.pyplot as plt

    t_mean_overall =  [0.013997101783752441, 0.03039367198944092, 0.07438118457794189, 0.11946399211883545, 0.2937695741653442, 0.5482940673828125]
    t_pair_overall =  [0.31872589588165284, 1.2426138877868653, 6.385974025726318, 18.74116370677948, 107.8408296585083, 433.0894425153732]
    t_cure_overall =  [0.051869869232177734, 0.10114574432373047, 0.24244928359985352, 0.4070738315582275, 0.9947033643722534, 2.0437520027160643]
    nmi_mean_overall =  [0.8727331141749997, 0.8165809720029363, 0.825610183152793, 0.8466369033156473, 0.9067854513948352, 0.8134456450144235]
    nmi_pair_overall =  [0.8923614548687876, 0.7323161401843006, 0.7428383543227939, 0.6745691462972153, 0.8413488680357822, 0.6881073383699136]
    nmi_cure_overall =  [0.7499341023280648, 0.7040499406565852, 0.6298378339347501, 0.6204230142951535, 0.7273275249169012, 0.6861498027466044]

    # Define the x and y coordinates of the points for Line 1
    x1 = [1, 2, 3, 4, 5, 6]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the first set of data on the first subplot
    axs[0].plot(x1, t_mean_overall, marker='o', linestyle='-', label='mean-cost')
    axs[0].plot(x1, t_cure_overall, marker='o', linestyle='--', label='cure-cost')
    axs[0].plot(x1, t_pair_overall, marker='o', linestyle='--', label='pairwise-cost')
    axs[0].set_xticks(x1)
    axs[0].set_xticklabels(['50', '100', '200', '400', '800','1600'])
    axs[0].set_yscale('log')

    axs[0].set_xlabel('number of points')
    axs[0].set_ylabel('log run-time (s)')
    axs[0].set_title('log run-time by number of points')
    axs[0].legend()

    axs[1].plot(x1, nmi_mean_overall, marker='o', linestyle='-', label='mean-cost')
    axs[1].plot(x1, nmi_cure_overall, marker='o', linestyle='--', label='cure-cost')
    axs[1].plot(x1, nmi_pair_overall, marker='o', linestyle='--', label='pairwise-cost')
    axs[1].set_xticks(x1)
    axs[1].set_xticklabels(['50', '100', '200', '400', '800','1600'])
    axs[1].set_xlabel('number of points')
    axs[1].set_ylabel('NMI-score')
    axs[1].set_title('NMI-score by number of points')
    axs[1].legend()

    # Adjust layout to prevent overlapping of labels
    plt.tight_layout()

    # Display the plots
    plt.show()

def plots_cut_tester():
    import matplotlib.pyplot as plt
    t_mean_overall =  [0.03646299839019775, 0.3784020900726318, 4.746135187149048, 120.4710624217987]
    t_axis_overall =  [0.014738941192626953, 0.13544008731842042, 1.9432506322860719, 30.23695206642151]
    t_adjusted_overall =  [0.019073796272277833, 0.16777973175048827, 2.341724419593811, 34.63127956390381]
    nmi_mean_overall =  [0.9043758631408301, 0.8751553544830175, 0.9026162972192804, 0.8596547371810068]
    nmi_axis_overall =  [0.9106196711353982, 0.8432934946211829, 0.805781267236061, 0.7897898730489521]
    nmi_adjusted_overall =  [0.9226486847518605, 0.912507782678962, 0.9316958888124841, 0.9105074614171489]
 

    # Define the x and y coordinates of the points for Line 1
    x1 = [1, 2, 3, 4]
    x2 = [1, 2, 3, 4]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the first set of data on the first subplot
    axs[0].plot(x1, t_mean_overall, marker='o', linestyle='-', label='mean-cut')
    axs[0].plot(x1, t_axis_overall, marker='o', linestyle='--', label='parallel-cut')
    axs[0].plot(x2, t_adjusted_overall, marker='o', linestyle='--', label='adjusted-cut')
    axs[0].set_xticks([1, 2, 3, 4])
    axs[0].set_xticklabels(['4*10^2', '4*10^3', '4*10^4', '4*10^5'])
    axs[0].set_yscale('log')

    axs[0].set_xlabel('number of points')
    axs[0].set_ylabel('log run-time (s)')
    axs[0].set_title('log run-time by number of points')
    axs[0].legend()

    axs[1].plot(x1, nmi_mean_overall, marker='o', linestyle='-', label='mean-cut')
    axs[1].plot(x1, nmi_axis_overall, marker='o', linestyle='--', label='parallel-cut')
    axs[1].plot(x2, nmi_adjusted_overall, marker='o', linestyle='--', label='adjusted-cut')
    axs[1].set_xticks([1, 2, 3, 4])
    axs[1].set_xticklabels(['4*10^2', '4*10^3', '4*10^3', '4*10^4'])
    axs[1].set_xlabel('number of points')
    axs[1].set_ylabel('NMI-score')
    axs[1].set_title('NMI-score by number of points')
    axs[1].legend()

    # Adjust layout to prevent overlapping of labels
    plt.tight_layout()

    # Display the plots
    plt.show()


def plot_numb_point_tester():
    import matplotlib.pyplot as plt

    t_tangle_overall =  [0.02759058475494385, 0.21849963665008545, 2.5372827291488647, 26.792141366004945, 249.438623046875]
    t_kmean_overall =  [0.035118699073791504, 0.0036871910095214845, 0.017798399925231932, 0.157812762260437, 1.1240960836410523]
    t_spectral_overall =  [0.13620738983154296, 0.2970494031906128, 29.911492156982423]
    nmi_tangle_overall =  [0.9134645913923481, 0.8757313391493675, 0.9314911390345866, 0.8863141382650348, 0.8907776070361095]
    nmi_kmean_overall =  [0.9601823321256425, 0.9714661562471392, 0.956904636391435, 0.9534829742647764, 0.975295834654258]
    nmi_spectral_overall =  [0.9514771896606309, 0.8918129608707746, 0.9789978367072788]

    # Define the x and y coordinates of the points for Line 1
    x1 = [1, 2, 3, 4, 5]
    x2 = [1, 2, 3]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the first set of data on the first subplot
    axs[0].plot(x1, t_tangle_overall, marker='o', linestyle='-', label='tangle')
    axs[0].plot(x1, t_kmean_overall, marker='o', linestyle='--', label='kmean')
    axs[0].plot(x2, t_spectral_overall, marker='o', linestyle='--', label='spectral')
    axs[0].set_xticks([1, 2, 3, 4, 5])
    axs[0].set_xticklabels(['10^2', '10^3', '10^4', '10^5', '10^6'])
    axs[0].set_yscale('log')

    axs[0].set_xlabel('number of points')
    axs[0].set_ylabel('log run-time (s)')
    axs[0].set_title('log run-time by number of points')
    axs[0].legend()

    axs[1].plot(x1, nmi_tangle_overall, marker='o', linestyle='-', label='tangle')
    axs[1].plot(x1, nmi_kmean_overall, marker='o', linestyle='--', label='kmean')
    axs[1].plot(x2, nmi_spectral_overall, marker='o', linestyle='--', label='spectral')
    axs[1].set_xticks([1, 2, 3, 4, 5])
    axs[1].set_xticklabels(['10^2', '10^3', '10^4', '10^5', '10^6'])
    axs[1].set_xlabel('number of points')
    axs[1].set_ylabel('NMI-score')
    axs[1].set_title('NMI-score by number of points')
    axs[1].legend()
    axs[1].set_ylim(0, 1)

    # Adjust layout to prevent overlapping of labels
    plt.tight_layout()

    # Display the plots
    plt.show()



def plot_tangle():
    test = Test()
    data = test.fixed_data(4,0.7,100, [(5,5), (6,9)])

    time, nmi_score, hard = test.tangles(data.points, data.ground_truth, 100)
    time2, nmi_score2, hard2 = test.tangles2(data.points, data.ground_truth, 100)
    
    print(nmi_score)
    print(nmi_score2)

    x = [point[0] for point in data.points]
    y = [point[1] for point in data.points]

    fig, axs = plt.subplots(1, 3, figsize=(12, 5))

    axs[0].scatter(x, y, c=data.ground_truth, cmap='viridis')
    axs[0].set_title('Ground truth')
    axs[1].scatter(x, y, c=hard, cmap='viridis')
    axs[1].set_title('Tangles mean')
    axs[2].scatter(x, y, c=hard2, cmap='viridis')
    axs[2].set_title('Tangles axis')
  

    plt.tight_layout()
    plt.show()


def different_cluster_sizes():
    k = 3
    a = 1000
    value_range = 0
    t_tangle_overall = []
    t_kmean_overall = []
    t_spectral_overall = []
    nmi_tangle_overall = []
    nmi_kmean_overall = []
    nmi_spectral_overall = []

    while a < 100000:
        t_tangle = []
        t_kmean = []
        t_spectral = []
        score_tangle = []
        score_kmean = []
        score_spectral = []
        for i in range(10):
            test = Test()
            data = test.random_data_different(k, 1.5, a, 2, value_range, 0.7)
            time_tangle, nmi_tangle, _ = test.tangles(data.points, data.ground_truth, int(a*0.95))
            time_kmean, nmi_kmean, _ = test.kmeans(data.points, data.ground_truth, k)
            time_spectral, nmi_spectral, _ = test.spectral(data.points, data.ground_truth, k)
            t_spectral.append(time_spectral)
            score_spectral.append(nmi_spectral)
            t_tangle.append(time_tangle)
            t_kmean.append(time_kmean)
            score_tangle.append(nmi_tangle)
            score_kmean.append(nmi_kmean)
            

        t_tangle_overall.append(sum(t_tangle)/len(t_tangle))
        t_kmean_overall.append(sum(t_kmean)/len(t_kmean))
        t_spectral_overall.append(sum(t_spectral)/len(t_spectral))
        nmi_spectral_overall.append(sum(score_spectral)/len(score_spectral))
        nmi_tangle_overall.append(sum(score_tangle)/len(score_tangle))
        nmi_kmean_overall.append(sum(score_kmean)/len(score_kmean))
        
        value_range += 100
        a+= 100
        print("t_tangle_overall: ", t_tangle_overall)
        print("t_kmean_overall: ", t_kmean_overall)
        print("t_spectral_overall: ", t_spectral_overall)
        print("nmi_tangle_overall: ", nmi_tangle_overall)
        print("nmi_kmean_overall: ", nmi_kmean_overall)
        print("nmi_spectral_overall: ", nmi_spectral_overall)
        print(a)

def plot_different_cluster_sizes():
    import matplotlib.pyplot as plt

    nmi_tangle_overall =  [0.9778982494338925, 0.9725638420977507, 0.9129780581275272, 0.723729302817415, 0.7569458047026212, 0.7937455971795002, 0.81325454654, 0.7849385609167621]
    nmi_kmean_overall =  [0.9746283805220651, 0.9935055107571589, 0.9829807266850918, 0.9792286450458253, 0.9266921660774751, 0.9574962719790999, 0.98615044051116, 0.9897709049592407]
    nmi_spectral_overall =  [0.9488243545704236, 0.9952054141980284, 0.9949351598445235, 0.8968186474528999, 0.9678899509863879, 0.9734946969338407, 0.9602759615760428, 0.9974195964722379]

    
    # Define the x and y coordinates of the points for Line 1
    x1 = [1, 2, 3, 4, 5, 6, 7, 8]

    plt.plot(x1, nmi_tangle_overall, marker='o', linestyle='-', label='tangle')
    plt.plot(x1, nmi_kmean_overall, marker='o', linestyle='--', label='kmean')
    plt.plot(x1, nmi_spectral_overall, marker='o', linestyle='--', label='spectral')
    plt.xticks(x1,['0', '100', '200', '300', '400', '500', '600', '700'])
    plt.xlabel('difference in cluster size (points)')
    plt.ylabel('NMI-score')
    plt.title('NMI-score by number of points')
    plt.legend()

    # Adjust layout to prevent overlapping of labels
    plt.tight_layout()

    # Display the plots
    plt.show()


def test_covert():
    from ucimlrepo import fetch_ucirepo 
    import pandas as pd
    from sklearn.manifold import TSNE
    
    
    from ucimlrepo import fetch_ucirepo 
    
    # fetch dataset 
    breast_cancer_wisconsin_original = fetch_ucirepo(id=15) 
    
    # data (as pandas dataframes) 
    X = breast_cancer_wisconsin_original.data.features 
    y = breast_cancer_wisconsin_original.data.targets 


    X = np.array(X)
    # Define your thresholds
    test = Test()
    print("hep")
    X = np.delete(X, 5, axis=1)
    y = y["Class"].values
    time, nmi, hard = test.tangles(X, y, 250)
    
    print(nmi)
    
    perplexity = min(20, len(X) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity)
    data = tsne.fit_transform(X)
    
    print(data[0,0])
    print(data[0,1])
    xv = [point[0] for point in data]
    yv = [point[1] for point in data]

    fig, axs = plt.subplots(1, 3, figsize=(12, 5))


    color_mapping = {0: 'blue', 1: 'red', 2: 'blue', 3: 'green', 4: 'orange', 5: 'purple', 6: 'brown', 7: 'pink'}

    colors = [color_mapping[val] for val in y]

    axs[0].scatter(xv, yv, c=colors)
    axs[0].set_title('Ground truth')
    axs[1].scatter(xv, yv, c=hard)
    axs[1].set_title('Tangles mean')

  

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_covert()


    


    




