import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

def scale(x):
    """Scale vector x using min max technique, in this way it's values are
    always between 0 and 1.

    :argument
        x -> values to be scaled

    :return
        the vector x scaled and the scaler used so it can be used to unscale
        results
    """

    scaler = MinMaxScaler()
    x = np.reshape(x, (-1, 1))
    scaler.fit(x)
    return scaler.transform(x), scaler

def kmeans(data, n_clusters, init='kmeans++', n_init=10, max_iter=300, tol=1e-04):
    """Used to init and fit KMeans using data.

    :argument
        data -> values used in training
        n_clusters = integer -> number of clusters
        init -> how to initialize the centroids
        n_init = integer -> number of multi-start
        max_iter = integer -> max number of iteration per start
        tol -> tolerance for stopping criteria

    :return
        the model trained
    """
    km = KMeans(n_clusters=n_clusters,
                init=init,
                n_init=n_init,
                max_iter=max_iter,
                tol=tol,
                random_state=0)
    km.fit(data)
    return km
