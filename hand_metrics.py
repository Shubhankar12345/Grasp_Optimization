import numpy as np
def orientation_similarity_metric(R1, R2):
    """ Computes the similarity between two rotation matrices """
    
    d1 = R1.flatten()
    d2 = R2.flatten() 
    sim_metric = np.sqrt(np.sum((d1-d2)**2))
    return sim_metric