#imports
from algorithms.clustering_without_standardization.Clustering import Clustering
import pandas as pd
from path import Path

#paths
path = Path()
dataset_path = path.dataset_path("dataset.csv") #already standardized.
report_path = path.report_path()

#get dataset
dataset = pd.read_csv(dataset_path)
dataset = dataset.iloc[:,0:7].values.tolist()

#clustering attributes
centroids = []
max_radius = 0.10 
min_members_cluster = 3

#clustering
clustering = Clustering(centroids, dataset, max_radius, min_members_cluster, 6)
clustering.start()

#generates a report on the clustering
clustering.generate_report("start_report", report_path)
