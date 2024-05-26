import matplotlib.pyplot as plt
import time
import networkx as nx
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import normalized_mutual_info_score
from Model.GenerateTestData import GenerateRandomGraph
from Model.DataSetGraph import DataSetGraph
from Model.TangleCluster import *
import community as community_louvain

# class GraphTest:
#     def __init__(self):
#         print("Graph test class constructor")

#     def tangles_test(self, generated_graph, generated_ground_truth, agreement_parameter, k, initial_partition_method="K-Means"):
#         try:
#             # Perform tangles on the generated graph
#             data = DataSetGraph(agreement_param=agreement_parameter, k=k)
#             data.G = generated_graph
#             start_time = time.time()
#             start_time_kernighan = time.time()
#             data.generate_multiple_cuts(data.G, initial_partition_method=initial_partition_method)
#             end_time_kernighan = time.time()
#             data.cost_function_Graph()
#             root = create_searchtree(data)
            
#             tangle_root = condense_tree(root)
#             contracting_search_tree(tangle_root)

#             if tangle_root.numb_tangles == 0:
#                 print("No tangles found")
            
#             soft = soft_clustering(tangle_root)
#             hard = hard_clustering(soft)
#             end_time = time.time()

#             total_time_kernighan = end_time_kernighan - start_time_kernighan
#             total_time = end_time - start_time

#             # Calculate NMI score only if ground truth is available
#             if generated_ground_truth:
#                 nmi_score = normalized_mutual_info_score(generated_ground_truth, hard)  # Assuming tangles contain the predicted tangles
#                 return total_time, total_time_kernighan, round(nmi_score, 2)
#             else:
#                 return total_time, None

#         except Exception as e:
#             print("Error:", e)
#             return None, None

#     def spectral_test(self, generated_graph, generated_ground_truth, k):
#         try:
#             start_time = time.time()
#             G = generated_graph
#             # Get adjacency matrix as numpy array
#             adj_mat = nx.convert_matrix.to_numpy_array(G)

#             # Cluster
#             sc = SpectralClustering(k, affinity='precomputed')  # Specify affinity as precomputed
#             sc.fit(adj_mat)

#             end_time = time.time()

#             total_time = end_time - start_time

#             # Calculate NMI score only if ground truth is available
#             if generated_ground_truth:
#                 nmi_score = normalized_mutual_info_score(generated_ground_truth, sc.labels_)
#                 return total_time, round(nmi_score, 2)
#             else:
#                 return total_time, None

#         except Exception as e:
#             print("Error in spectral clustering:", e)
#             return None, None

#     def louvain_test(self, generated_graph, generated_ground_truth):
#         try:
#             G = generated_graph
#             start_time = time.time()

#             # Perform Louvain clustering
#             partition = community_louvain.best_partition(G)

#             end_time = time.time()

#             total_time = end_time - start_time

#             # Calculate NMI score only if ground truth is available
#             if generated_ground_truth:
#                 labels = [partition[node] for node in G.nodes()]
#                 nmi_score = normalized_mutual_info_score(generated_ground_truth, labels)
#                 return total_time, round(nmi_score, 2)
#             else:
#                 return total_time, None

#         except Exception as e:
#             print("Error in Louvain clustering:", e)
#             return None, None

#     def nmi_score(self, ground_truth, predicted_tangles):
#         """
#         Calculates the NMI score of the predicted tangles
#         """
#         nmi_score = normalized_mutual_info_score(ground_truth, predicted_tangles)
#         return nmi_score
    
# if __name__ == "__main__":
#     test = GraphTest()
#     num_of_clusters = 10  # Increase the number of clusters
#     agreement_parameter = [14, 28, 42, 56, 70]  # Adjusted agreement parameter values
#     avg_edges_to_same_cluster = 0.6
#     avg_edges_to_other_clusters = 0.4
#     k = num_of_clusters
#     num_iterations = 10

#     # Increase the number of nodes for testing
#     num_of_nodes_list = [400, 800, 1200, 1600, 2000]  # Adjusted number of nodes

#     avg_nmi_scores_list = []
#     avg_running_times_list = []

#     for num_of_nodes in num_of_nodes_list:
#         avg_nmi_scores = []
#         avg_running_times = []

#         for _ in range(num_iterations):
#             random_graph_generator = GenerateRandomGraph(num_of_nodes, num_of_clusters, avg_edges_to_same_cluster,
#                                                          avg_edges_to_other_clusters)

#             # Generate a random graph using the ground truth
#             generated_graph, generated_ground_truth = random_graph_generator.generate_random_graph()

#             # Lists to store the results
#             algorithms = ['Tangles (Kernighan-Lin)', 'Tangles (K-Means)', 'Louvain Clustering', 'Spectral Clustering']
#             running_times = []
#             nmi_scores = []

#             # Run tests for each algorithm
#             for algorithm in algorithms:
#                 if algorithm == 'Tangles (Kernighan-Lin)':
#                     tangles_time, tangles_kernighan_time, tangles_nmi = test.tangles_test(generated_graph, generated_ground_truth,
#                                                                      agreement_parameter[num_of_nodes_list.index(num_of_nodes)],k, initial_partition_method="Kernighan-Lin")
#                     nmi_scores.append(tangles_nmi)
#                     running_times.append(tangles_time)
#                 elif algorithm == 'Tangles (K-Means)':
#                     tangles_time, tangles_kmeans_time, tangles_nmi = test.tangles_test(generated_graph, generated_ground_truth,
#                                                                      agreement_parameter[num_of_nodes_list.index(num_of_nodes)],k, initial_partition_method="K-Means")
#                     nmi_scores.append(tangles_nmi)
#                     running_times.append(tangles_time)
#                 elif algorithm == 'Spectral Clustering':
#                     spectral_time, spectral_nmi = test.spectral_test(generated_graph, generated_ground_truth, k)
#                     nmi_scores.append(spectral_nmi)
#                     running_times.append(spectral_time)
#                 elif algorithm == 'Louvain Clustering':
#                     louvain_time, louvain_nmi = test.louvain_test(generated_graph, generated_ground_truth)
#                     nmi_scores.append(louvain_nmi)
#                     running_times.append(louvain_time)

#             avg_nmi_scores.append(nmi_scores)
#             avg_running_times.append(running_times)

#         # Calculate average NMI scores and running times
#         avg_nmi_scores = [sum(scores) / num_iterations for scores in zip(*avg_nmi_scores)]
#         avg_running_times = [sum(times) / num_iterations for times in zip(*avg_running_times)]

#         avg_nmi_scores_list.append(avg_nmi_scores)
#         avg_running_times_list.append(avg_running_times)

#     # Plotting average NMI scores against number of nodes
#     plt.figure(figsize=(10, 5))
#     for i in range(len(algorithms)):  # Include all algorithms
#         plt.plot(num_of_nodes_list, [score[i] if score[i] is not None else 0 for score in avg_nmi_scores_list], label=algorithms[i])
#     plt.xlabel('Number of Nodes')
#     plt.ylabel('Average NMI Score')
#     plt.title('Average NMI Scores of Different Clustering Algorithms ({} iterations)'.format(num_iterations))
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#     # Plotting average running times against number of nodes
#     plt.figure(figsize=(10, 5))
#     for i in range(len(algorithms)):
#         plt.plot(num_of_nodes_list, [time[i] if time[i] is not None else 0 for time in avg_running_times_list], label=algorithms[i])
#     plt.xlabel('Number of Nodes')
#     plt.ylabel('Average Running Time (s)')
#     plt.title('Average Running Times of Different Clustering Algorithms ({} iterations)'.format(num_iterations))
#     plt.legend()
#     plt.grid(True)
#     plt.show()


class CompareKMeansInitialPartitioning:
    def __init__(self):
        print("Graph test class constructor")

    def tangles_test(self, generated_graph, generated_ground_truth, agreement_parameter, k, initial_partition_method="K-Means"):
        try:
            data = DataSetGraph(agreement_param=agreement_parameter, k=k)
            data.G = generated_graph
            start_time = time.time()
            start_time_kernighan = time.time()
            data.generate_multiple_cuts(data.G, initial_partition_method=initial_partition_method)
            end_time_kernighan = time.time()
            data.cost_function_Graph()
            root = create_searchtree(data)
            
            tangle_root = condense_tree(root)
            contracting_search_tree(tangle_root)

            if tangle_root.numb_tangles == 0:
                print("No tangles found")
            
            soft = soft_clustering(tangle_root)
            hard = hard_clustering(soft)
            end_time = time.time()

            total_time_kernighan = end_time_kernighan - start_time_kernighan
            total_time = end_time - start_time

            if generated_ground_truth:
                nmi_score = normalized_mutual_info_score(generated_ground_truth, hard)
                return total_time, total_time_kernighan, round(nmi_score, 2)
            else:
                return total_time, None

        except Exception as e:
            print("Error:", e)
            return None, None

    def spectral_test(self, generated_graph, generated_ground_truth, k):
        try:
            start_time = time.time()
            G = generated_graph
            adj_mat = nx.convert_matrix.to_numpy_array(G)

            sc = SpectralClustering(k, affinity='precomputed')
            sc.fit(adj_mat)

            end_time = time.time()
            total_time = end_time - start_time

            if generated_ground_truth:
                nmi_score = normalized_mutual_info_score(generated_ground_truth, sc.labels_)
                return total_time, round(nmi_score, 2)
            else:
                return total_time, None

        except Exception as e:
            print("Error in spectral clustering:", e)
            return None, None

    def louvain_test(self, generated_graph, generated_ground_truth):
        try:
            G = generated_graph
            start_time = time.time()

            partition = community_louvain.best_partition(G)

            end_time = time.time()
            total_time = end_time - start_time

            if generated_ground_truth:
                labels = [partition[node] for node in G.nodes()]
                nmi_score = normalized_mutual_info_score(generated_ground_truth, labels)
                return total_time, round(nmi_score, 2)
            else:
                return total_time, None

        except Exception as e:
            print("Error in Louvain clustering:", e)
            return None, None

    def nmi_score(self, ground_truth, predicted_tangles):
        nmi_score = normalized_mutual_info_score(ground_truth, predicted_tangles)
        return nmi_score
    
if __name__ == "__main__":
    test = CompareKMeansInitialPartitioning()
    num_of_clusters = 10
    agreement_parameter = [14, 28, 42, 56, 70]
    avg_edges_to_same_cluster = 0.6
    avg_edges_to_other_clusters = 0.4
    k = num_of_clusters
    num_iterations = 1

    num_of_nodes_list = [400, 800, 1200, 1600, 2000]

    avg_nmi_scores_list = []
    avg_running_times_list = []

    for num_of_nodes in num_of_nodes_list:
        avg_nmi_scores = []
        avg_running_times = []

        for _ in range(num_iterations):
            random_graph_generator = GenerateRandomGraph(num_of_nodes, num_of_clusters, avg_edges_to_same_cluster,
                                                         avg_edges_to_other_clusters)

            generated_graph, generated_ground_truth = random_graph_generator.generate_random_graph()

            algorithms = ['Tangles (K-Means-centroids-cuts)', 'Tangles (K-Means-Half)', 'Tangles (K-Means-Both)', 'Louvain Clustering', 'Spectral Clustering']
            running_times = []
            nmi_scores = []

            for algorithm in algorithms:
                if algorithm == 'Tangles (K-Means-centroids-cuts)':
                    tangles_time, tangles_kmeans_time, tangles_nmi = test.tangles_test(generated_graph, generated_ground_truth,
                                                                     agreement_parameter[num_of_nodes_list.index(num_of_nodes)], k, initial_partition_method="K-Means")
                    nmi_scores.append(tangles_nmi)
                    running_times.append(tangles_time)
                elif algorithm == 'Tangles (K-Means-Half)':
                    tangles_time, tangles_kmeans_half_time, tangles_nmi = test.tangles_test(generated_graph, generated_ground_truth,
                                                                        agreement_parameter[num_of_nodes_list.index(num_of_nodes)], k, initial_partition_method="K-Means-Half")
                    nmi_scores.append(tangles_nmi)
                    running_times.append(tangles_time)
                elif algorithm == 'Tangles (K-Means-Both)':
                    tangles_time, tangles_kmeans_both_time, tangles_nmi = test.tangles_test(generated_graph, generated_ground_truth,
                                                                     agreement_parameter[num_of_nodes_list.index(num_of_nodes)], k, initial_partition_method="K-Means-Both")
                    nmi_scores.append(tangles_nmi)
                    running_times.append(tangles_time)
                elif algorithm == 'Spectral Clustering':
                    spectral_time, spectral_nmi = test.spectral_test(generated_graph, generated_ground_truth, k)
                    nmi_scores.append(spectral_nmi)
                    running_times.append(spectral_time)
                elif algorithm == 'Louvain Clustering':
                    louvain_time, louvain_nmi = test.louvain_test(generated_graph, generated_ground_truth)
                    nmi_scores.append(louvain_nmi)
                    running_times.append(louvain_time)

            avg_nmi_scores.append(nmi_scores)
            avg_running_times.append(running_times)

        avg_nmi_scores = [sum(scores) / num_iterations for scores in zip(*avg_nmi_scores)]
        avg_running_times = [sum(times) / num_iterations for times in zip(*avg_running_times)]

        avg_nmi_scores_list.append(avg_nmi_scores)
        avg_running_times_list.append(avg_running_times)

    plt.figure(figsize=(10, 5))
    for i in range(len(algorithms)):
        plt.plot(num_of_nodes_list, [score[i] if score[i] is not None else 0 for score in avg_nmi_scores_list], label=algorithms[i])
    plt.xlabel('Number of Nodes')
    plt.ylabel('Average NMI Score')
    plt.title('Average NMI Scores of Different Clustering Algorithms ({} iterations)'.format(num_iterations))
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    for i in range(len(algorithms)):
        plt.plot(num_of_nodes_list, [time[i] if time[i] is not None else 0 for time in avg_running_times_list], label=algorithms[i])
    plt.xlabel('Number of Nodes')
    plt.ylabel('Average Running Time (s)')
    plt.title('Average Running Times of Different Clustering Algorithms ({} iterations)'.format(num_iterations))
    plt.legend()
    plt.grid(True)
    plt.show()
