#This class doesn't standardize a dataset, it must be standardized before being clustered.
import math
import os 
import numpy as np

class Clustering:
    def __init__(self,  centroids: list, dataset: list, max_radius: int, min_in_cluster: int, dataset_dimensions: int ) -> None:
        """Receives a dataset, the centroids, the maximum value for the centroids' cluster
           and creates an empty list to store the clusters.
        """
        self.__dataset = dataset
        self.__original_dataset = dataset #used for accuracy test in case the dataset is changed.
        self.__max_radius = max_radius
        self.__dataset_dimensions = dataset_dimensions
        self.__min_in_cluster = min_in_cluster
        self.__accuracy = -1

        self.__centroids = []
        if centroids == []:
            self.__centroids.append(self.__dataset[0])
        else:
            for index in centroids:
                self.__centroids.append(self.__dataset[index])

        self.__clusters = []
        for centroid in self.__centroids:
            new_centroid = [centroid]
            self.__clusters.append(new_centroid)

    #getters and setters
    @property    
    def dataset(self):
        return self.__dataset

    @dataset.setter
    def dataset(self, dataset: list):
        self.__dataset = dataset

    @property    
    def centroids(self):
        return self.__centroids 

    @centroids.setter
    def centroids(self, centroids: list):
        self.__centroids = centroids   

    @property
    def clusters(self):
        return self.__clusters

    @clusters.setter
    def clusters(self, clusters: list):
        self.__clusters = clusters

    @property    
    def max_radius(self):
        return self.__max_radius

    @max_radius.setter
    def max_radius(self, max_radius: int):
        self.__max_radius = max_radius

    @property
    def dataset_dimensions(self):
        return self.__dataset_dimensions

    @dataset_dimensions.setter
    def dataset_dimensions(self, dataset_dimensions: int):
        self.__dataset_dimensions = dataset_dimensions

    @property
    def min_in_cluster(self):
        return self.__min_in_cluster

    @min_in_cluster.setter
    def max_in_cluster(self, min_in_cluster):
        self.__min_in_cluster = min_in_cluster

    @property
    def accuracy(self):
        return self.__accuracy


    #essential methods for clustering
    def new_data_point(self, data_point: list) -> None:
        """Creates a new data point in the data set."""
        self.__dataset.append(data_point)

    def new_cluster(self, centroid: list) -> None:
        """Creates a new cluster and defines its centroid."""
        new_cluster = [centroid]
        self.centroids.append(centroid)
        self.clusters.append(new_cluster)

    def euclidean_distance(self, data_point1: list, data_point2: list) -> float:
        """Calculates and returns the euclidean distance between two 
        given data points
        """
        num_coordinates = self.dataset_dimensions

        distance = 0
        for i in range(num_coordinates):
            
            distance += ((data_point1[i] - data_point2[i]) ** 2)
        distance = math.sqrt(distance)

        return distance

    def datapoint_in_cluster(self, data_point: list, centroid: list) -> bool:
        """Determines if a data point belongs to a cluster, given its centroid."""
        distance = self.euclidean_distance(data_point, centroid)

        if distance <= self.max_radius:
            return True
        else:
            return False

    def show_clusters(self) -> None:
        """Shows all the clusters in the terminal."""
        n = 1
        for cluster in self.clusters:
            print(f'cluster {n}: {cluster}')
            n += 1

    def show_number_of_clusters(self) -> int:
        """Returns the numbers of clusters calculated by the algorithm."""
        number_of_clusters = len(self.clusters)
        print(number_of_clusters)

        return number_of_clusters


    #methods to remove outliers
    def get_data_dimensions(self) -> list:
        """Gets the data points dimensions and separates them in lists.
           It returns an array with all the data dimensions."""
        dataset = self.dataset

        data_0 = []
        data_1 = []
        data_2 = []
        data_3 = []
        data_4 = []
        data_5 = []
        data_ID = []

        for data_point in dataset:
            data_0.append(data_point[0])
            data_1.append(data_point[1])
            data_2.append(data_point[2])
            data_3.append(data_point[3])
            data_4.append(data_point[4])
            data_5.append(data_point[5])
            data_ID.append(data_point[6])

        data_dimensions = [data_0, data_1, data_2, data_3, data_4, data_5,data_ID]

        return data_dimensions

    def to_dataset(self, data_dimensions) -> list:
        """Receives a data_dimensions array and converts it to dataset."""
        dataset = []

        for index_datapoint in range(len(data_dimensions[0])):

            datapoint = [data_dimensions[0][index_datapoint],
                         data_dimensions[1][index_datapoint],
                         data_dimensions[2][index_datapoint],
                         data_dimensions[3][index_datapoint],
                         data_dimensions[4][index_datapoint],
                         data_dimensions[5][index_datapoint],
                         data_dimensions[6][index_datapoint]]
            dataset.append(datapoint)

        return dataset

    def remove_outliers(self) -> list:
        """Removes outliers in the dataset. And returns the dataset without outliers"""
        data_dimensions = self.get_data_dimensions()
        data_dimensions = np.array(data_dimensions)

        indexes_to_remove = []
        for index_axis, axis in enumerate(data_dimensions[0:6]):

            data_std = np.std(data_dimensions[index_axis])
            data_mean = np.mean(data_dimensions[index_axis])

            for index_value, value in enumerate(axis):
                if not (value <= data_mean + 3*data_std):
                    indexes_to_remove.append(index_value)

        data_dimensions = data_dimensions.tolist()
        aux_data_dimensions = self.get_data_dimensions()
        indexes_to_remove = list(set(indexes_to_remove))

        for index in indexes_to_remove:
            data_dimensions[0].remove(aux_data_dimensions[0][index])
            data_dimensions[1].remove(aux_data_dimensions[1][index])
            data_dimensions[2].remove(aux_data_dimensions[2][index])
            data_dimensions[3].remove(aux_data_dimensions[3][index])
            data_dimensions[4].remove(aux_data_dimensions[4][index])
            data_dimensions[5].remove(aux_data_dimensions[5][index])
            data_dimensions[6].remove(aux_data_dimensions[6][index])
            

        dataset = self.to_dataset(data_dimensions)
        self.dataset = dataset 
        
        return self.dataset


    #methods for running the algorithm
    def start(self) -> list:
        """Starts the clustering algorithm and returns the clusters."""
        dataset = self.dataset
        centroids = self.centroids

        data_in_cluster = [False]*len(dataset)

        for index_data_point, data_point in enumerate(dataset):
            for index_centroid, centroid in enumerate(centroids):
                if (self.datapoint_in_cluster(data_point, centroid)) and (not data_in_cluster[index_data_point]):
                    self.clusters[index_centroid].append(data_point)
                    data_in_cluster[index_data_point] = True

            if not data_in_cluster[index_data_point]:
                self.new_cluster(data_point)

        return self.clusters

    def start_min(self) -> list:
        """Starts the clustering algorithm and returns the clusters
        that have less or an equal number of members in comparison to
        the maximum number in a cluster."""
        self.start()

        clusters = self.clusters
        min = self.min_in_cluster
        new_clusters = []
        new_centroids = []

        for cluster in clusters:
            if len(cluster) >= min:
                new_clusters.append(cluster)
                new_centroids.append(cluster[0])

        self.__clusters = new_clusters
        self.__centroids = new_centroids

        return self.__clusters

    def start_without_outliers(self) -> None:
        """Removes the outliers and runs the algorithm."""
        self.remove_outliers()
        self.start()
        
    def run(self) -> None:
        """Runs the full clustering algorithm, without outliers and with a minimum
            number of members for a cluster."""
        self.remove_outliers()
        self.start_min()
            

    #methods to calculate the accuracy of the algorithm (correctlyClustered-wronglyClustered). Discarted.
    def accuracy(self) -> float:
        """Checks if the clustering is right and returns its accuracy."""
        sum_accuracy_clusters = 0
        clusters = self.clusters
        num_clusters = len(clusters)

        for cluster in clusters:
            cluster_accuracy = self.cluster_accuracy(cluster)

            sum_accuracy_clusters += cluster_accuracy
        
        accuracy = sum_accuracy_clusters / num_clusters

        return accuracy * 100

    def cluster_accuracy(self, cluster) -> float:
        """Returns the accuracy of a cluster."""
        num_datapoints = len(cluster)
        cluster_ID = self.get_cluster_ID(cluster)

        correctly_clustered_datapoint = 0
        for datapoint in cluster:
            id = datapoint[-1]
            if id == cluster_ID:
                correctly_clustered_datapoint += 1
        
        cluster_accuracy = correctly_clustered_datapoint / num_datapoints

        return cluster_accuracy * 100

    def get_cluster_ID(self, cluster) -> int:
        """Returns the ID which represents a certain cluster."""
        IDs = self.get_IDs(cluster)

        counter = 0
        cluster_ID = 0 #The most frequent element in IDs' list.

        for id in IDs: #Perhaps will have a high cost.
            curr_frequency = IDs.count(id)

            if curr_frequency > len(IDs)/2:
                return id 

            if(curr_frequency > counter):
                counter = curr_frequency
                cluster_ID = id

        return cluster_ID

    def get_IDs(self, cluster) -> list:
        """Returns a list of all the IDs of the datapoints in a cluster."""
        IDs = []
        
        for datapoint in cluster:
            id = datapoint[-1] #or index == 6
            IDs.append(id)

        return IDs


    #methods to calculate the accuracy of the algorithm (clustered-nonClustered).
    def capture_accuracy(self) -> float:
        """Checks how many datapoints would be clustered given the centroids after the clustization and returns the accuracy in percentage. Is 100% accurate only if no datapoint has been removed."""
        dataset = self.__original_dataset
        centroids = self.__centroids
        max_radius = self.__max_radius

        captured = 0
        for datapoint in dataset:
            for centroid in centroids:
                if self.bind(datapoint, centroid, max_radius):
                    captured += 1
                    break
        
        num_datapoints = len(dataset)

        self.__accuracy = (float(captured / num_datapoints)) * 100

        return self.__accuracy

    def bind(self, datapoint_x, datapoint_y, threshold) -> bool:
        """Checks if the distance between the datapoints x and y exceeds a binding threshold."""
        threshold = self.max_radius
        dist = self.euclidean_distance(datapoint_x, datapoint_y)

        if dist <= threshold:
            return True
        else:
            return False


    #method that calculates the % of clustered dps by dividing the number of clustered datapoints by the number of datapoints in the dataset.
    def rate_of_clusterization(self) -> float:
        total = len(self.__original_dataset)
        clustered = 0
        
        for cluster in self.__clusters:
            clustered += len(cluster)
            
        return ((clustered / total * 100))


    #method that generates a report of the algorithm's performance
    def generate_report(self, txt_name:str, folder_path:str) -> None:
        """Creates a txt file and saves the clusters in it."""
        path = f"{folder_path}/{txt_name}.txt"
        path = os.path.join(path)
        
        txt_cluster = open(path, "w+")
        number_of_clusters = len(self.clusters)

        txt_cluster.write(f'{txt_name}: \n'
                          '\n'
                          f'centroids: {self.centroids} \n' 
                          '\n'
                          f'radius: {self.max_radius} \n'
                          '\n'
                          f'minimum number of members in a cluster: {self.min_in_cluster} \n'
                          '\n'
                          f'accuracy (with the final set of centroids): {self.capture_accuracy()} \n'
                          '\n'
                          f'rate of clusterization (clustered datapoints): {self.rate_of_clusterization()} \n'
                          '\n'
                          f'number of clusters: {number_of_clusters} \n'
                          '\n')

        n = 1
        for cluster in self.clusters: 
            txt_cluster.write(f'cluster {n}: {cluster}')
            n += 1

            txt_cluster.write('\n') #Blank line.
            txt_cluster.write('\n')

        txt_cluster.close()
