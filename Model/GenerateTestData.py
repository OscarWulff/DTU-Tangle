import numpy as np
import matplotlib.pyplot as plt


class GenerateDataFeatureBased():
    
    # [(x, y, x)]

    def __init__(self, numb_clusters, std_deviation, centroids = [(1,1), (10,10)]):
        self.numb_clusters = numb_clusters
        self.std_deviation = std_deviation
        self.centroids = centroids
        self.points = []
        self.ground_truth = []

        # The parameters for the encirclement 
        self.box_low_x = 0
        self.box_high_x = 20
        self.box_low_y = 0
        self.box_high_y = 20

    def fixed_clusters(self, cluster_points):
        """
        Creating two clusters in 2 dimension for two fixed centroids.
        The points is created from Gaussian Distribution. 
        """
        points_x = []
        points_y = []

        for truth, (center_x, center_y) in enumerate(self.centroids):
            # Generate points using Gaussian distribution
            points_x.extend(np.random.normal(loc=center_x, scale=self.std_deviation, size=cluster_points))
            points_y.extend(np.random.normal(loc=center_y, scale=self.std_deviation, size=cluster_points))
            
            for _ in range(cluster_points):
                self.ground_truth.append(truth)

        self.points = [(x, y, z) for z, (x, y) in enumerate(zip(points_x, points_y))]

        print(self.points)
        print(self.ground_truth)
           

    def random_clusters(self, cluster_points):
        """
        Create a choosen number of clusters from Gaussian Distribution. 
        Standard deviation and centroids are choosen random. 
        """
        # Parameter that controls how much overlap is allowed
        overlap = 0.2

        std_low = 0.1
        std_high = 2

        tries = 0
        while(tries < 1200):
            tries += 1
            start_over = False
            centroids = []
            std_deviations = [] 
            for i in range(self.numb_clusters): 
                std_deviation_x = np.random.uniform(std_low, std_high)
                std_deviation_y = np.random.uniform(std_low, std_high)
                center_x = np.random.uniform(self.box_low_x, self.box_high_x)
                center_y = np.random.uniform(self.box_low_y, self.box_high_y)
                
                for i in range(len(centroids)): 
                    x1, y1 = centroids[i]
                    std_x, std_y = std_deviations[i]

                    # Check if it is not too close to another cluster
                    if np.abs(center_x-x1) < (std_deviation_x + std_x) * overlap:
                        start_over = True
                    if np.abs(center_y-y1) < (std_deviation_y + std_y) * overlap:
                        start_over = True
                    # Check clusters are inside the box
                    if center_y - std_deviation_y*3 < self.box_low_y and center_y + std_deviation_y*3 > self.box_high_y:
                        start_over = True
                    if center_x - std_deviation_x*3 < self.box_low_x and center_x + std_deviation_x*3 > self.box_high_x:
                        start_over = True

                if start_over:
                    break

                centroids.append((center_x, center_y))
                std_deviations.append((std_deviation_x, std_deviation_y))


            if not start_over:
                points_x = []
                points_y = []

                for truth, (center_x, center_y) in enumerate(centroids):
                    # Generate points using Gaussian distribution
                    points_x.extend(np.random.normal(loc=center_x, scale=std_deviations[truth][0], size=cluster_points))
                    points_y.extend(np.random.normal(loc=center_y, scale=std_deviations[truth][1], size=cluster_points))
                    
                    for _ in range(cluster_points):
                        self.ground_truth.append(truth)

                self.points = [(x, y, z) for z, (x, y) in enumerate(zip(points_x, points_y))]
                print(self.points)
                print(self.ground_truth)
                print(tries)
                break
        
        if tries == 300: 
            print("to many tries")

    def plot_points(self):
        clusters = sorted(set(self.ground_truth))
        colors = plt.cm.viridis(np.linspace(0, 1, len(clusters)))
        color_map = {cluster: color for cluster, color in zip(clusters, colors)}

        # Plot the points with color
        for point, truth in zip(self.points, self.ground_truth):
            plt.scatter(point[0], point[1], color=color_map[truth])

        plt.xlim(self.box_low_x-1, self.box_high_x+1)  # Setting x-axis limits from 0 to 10
        plt.ylim(self.box_low_y-1, self.box_high_y+1) 
        # Add labels and title
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Colorized Clusters')

        # Display the plot
        plt.show()
            
                
                
             





