1. Randomly initialize k cluster centroids (u1, u2,...,uk)

Each uk vector is of shape (1, num_of_features)

Repeat till centroids are converged (or) max_iters has finished {

    2. Assign points to cluster centroids, for each training sample (i=1tom)
       c(i) = min|| x(i) - uk ||**2 (The cluster centroid should have minimum distance from the training sample)

    3. Move cluster centroids to the mean of all the points assigned to that cluster
       uk = Mean of points assigned to cluster k

}
