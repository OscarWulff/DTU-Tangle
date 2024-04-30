
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

        start_time = time.time()
        spectral = SpectralClustering(n_clusters=k, eigen_solver="arpack", affinity="nearest_neighbors")
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
        
        tangle.cut_spectral()
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
        tangle.cut_spectral()
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


    def tester(self, numb_clusters, std, numb_points, dimension, overlap, a, k):
        tangles_nmi = []
        tangles_time = []
        spectral_nmi = []
        spectral_time = []
        kmeans_nmi = []
        kmeans_time = []

        for _ in range(10):        
            data = self.random_data(numb_clusters, std, numb_points, dimension, overlap)

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

    numb = 10

    data_points = [noisy_circles[0], noisy_moons[0], blobs[0], no_structure[0], aniso[0], varied[0]]
    plot_names = ["Noisy circles", "Noisy moons", "Blobs", "No structure", "Aniso", "Varied"]


    for i in range(numb):
        t, nmi_score, labels_circle = test.tangles(noisy_circles[0], noisy_circles[1], 750)
        noisy_circles_time.append(t)
        noisy_circles_nmi.append(nmi_score)
       
        t, nmi_score, labels_moon = test.tangles(noisy_moons[0], noisy_moons[1], 750)
        noisy_moons_time.append(t)
        noisy_moons_nmi.append(nmi_score)

        t, nmi_score, labels_blobs = test.tangles(blobs[0], blobs[1], 400)
        blobs_time.append(t)
        blobs_nmi.append(nmi_score)

        t, nmi_score, labels_no = test.tangles(no_structure[0], no_structure[1], 400)
        no_structure_time.append(t)

        t, nmi_score, labels_aniso = test.tangles(aniso[0], aniso[1], 400)
        aniso_time.append(t)
        aniso_nmi.append(nmi_score)

        t, nmi_score, labels_varied = test.tangles(varied[0], varied[1], 400)
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


if __name__ == "__main__":
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
    spectral_tester(noisy_circles, noisy_moons, blobs, no_structure, aniso, varied)
    #tangles_tester(noisy_circles, noisy_moons, blobs, no_structure, aniso, varied)


    


    




