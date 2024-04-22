
from Model.GenerateTestData import GenerateDataFeatureBased
from sklearn.metrics.cluster import normalized_mutual_info_score
import time
from sklearn.cluster import KMeans, SpectralClustering
from Model.DataSetFeatureBased import DataSetFeatureBased
from Model.TangleCluster import *
from Model.SearchTree import *


class Test:
    def __init__(self):
        print("Test class constructor")


    def spectral(self, data, k):
        
        start_time = time.time()
        spectral = SpectralClustering(n_clusters=k)
        spectral.fit(data.points)
        
        end_time = time.time()
        
        total_time = end_time - start_time

        nmi_score = normalized_mutual_info_score(data.ground_truth, spectral.labels_)

        return total_time, nmi_score

    def kmeans(self, data, k):
        
        
        start_time = time.time()

        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data.points)
        
        end_time = time.time()

        total_time = end_time - start_time

        nmi_score = normalized_mutual_info_score(data.ground_truth, kmeans.labels_)

        return total_time, nmi_score
    



    def tangles(self, data, a):
         # Creating the tangles
        tangle = DataSetFeatureBased(a)

        tangle.points = data.points

        start_time = time.time()
        
        tangle.cut_generator_axis_dimensions()
        tangle.pairwise_cost()
    
        root = create_searchtree(tangle)
        tangle_root = condense_tree(root)
        contracting_search_tree(tangle_root)
        soft = soft_clustering(tangle_root)
        hard = hard_clustering(soft)

        end_time = time.time()

        if tangle_root.left_node is None and tangle_root.right_node is None:
            print("No tangles found")


        
        total_time = end_time - start_time
        nmi_score = normalized_mutual_info_score(data.ground_truth, hard)


        return total_time, nmi_score




    def random_tester(self, numb_clusters, std, numb_points, dimension, overlap, a, k):
        tangles_nmi = []
        tangles_time = []
        spectral_nmi = []
        spectral_time = []
        kmeans_nmi = []
        kmeans_time = []

        for _ in range(10):        
            data = GenerateDataFeatureBased(numb_clusters, std)
            data.random_clusters(numb_points, dimension, overlap)

            # Tangles
            time, nmi_score = self.tangles(data, a)
            tangles_nmi.append(nmi_score)
            tangles_time.append(time)


            # Spectral clustering
            time, nmi_score = self.spectral(data, k)
            spectral_nmi.append(nmi_score)
            spectral_time.append(time)

            # Kmeans clustering
            time, nmi_score = self.kmeans(data, k)
            kmeans_nmi.append(nmi_score)
            kmeans_time.append(time)

        print(f"tangles_nmi: {sum(tangles_nmi)/len(tangles_nmi)}")
        print(f"tangles_time: {sum(tangles_time)/len(tangles_time)}")
        print(f"spectral_nmi: {sum(spectral_nmi)/len(spectral_nmi)}")
        print(f"spectral_time: {sum(spectral_time)/len(spectral_time)}")
        print(f"kmeans_nmi: {sum(kmeans_nmi)/len(kmeans_nmi)}")
        print(f"kmeans_time: {sum(kmeans_time)/len(kmeans_time)}")



test = Test()
test.random_tester(4, 1, 100, 2, 0.5, 100, 4)


