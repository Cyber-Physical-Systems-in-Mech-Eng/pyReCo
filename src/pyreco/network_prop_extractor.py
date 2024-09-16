from .utils_networks import (
    extract_density, extract_spectral_radius, extract_in_degree,
    extract_out_degree, extract_clustering_coefficient
)

class  NetworkQuantifier:
    def __init__(self, quantities=None):
        self.quantities = quantities or ['density', 'spectral_radius', 'in_degree', 'out_degree', 'clustering_coefficient']
        self.extractors = {
            'density': extract_density,
            'spectral_radius': extract_spectral_radius,
            'in_degree_av': extract_in_degree,  
            'out_degree_av': extract_out_degree,
            'clustering_coefficient': extract_clustering_coefficient
        }

    def extract(self, adjacency_matrix):
        network_props = {}
        for quantity in self.quantities:
            if quantity in self.extractors:
                network_props[quantity] = self.extractors[quantity](adjacency_matrix)
            else:
                print(f"Warning: {quantity} is not a recognized network property.")
        return network_props