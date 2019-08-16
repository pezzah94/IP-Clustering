from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.rock import rock
from pyclustering.samples.definitions import FCPS_SAMPLES
from pyclustering.utils import read_sample
# Read sample for clustering from file.
sample = read_sample(FCPS_SAMPLES.SAMPLE_HEPTA)


# Create instance of ROCK algorithm for cluster analysis. Seven clusters should be allocated.
rock_instance = rock(sample, 1.0, 7)
# Run cluster analysis.
rock_instance.process()
# Obtain results of clustering.
clusters = rock_instance.get_clusters()
# Visualize clustering results.
visualizer = cluster_visualizer()
visualizer.append_clusters(clusters, sample)
visualizer.show()