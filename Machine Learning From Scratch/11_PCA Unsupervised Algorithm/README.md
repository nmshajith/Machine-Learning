
1. Data Preprocessing:

Mean u_j = 1/m (i=1tom)Σ xj(i)
X = X-U (U is the Mean Vector)

2. Compute CoVariance Matrix
Σ = 1/m * (i=1ton)Σ x(i) (x(i)).T

3. Compute eighen vector of Covariance matrix
eighen_values, eighen_vector = np.linalg.eig(Covariance-matrix)
U = eighen_vector (should be of shape (n,n) )
k -> no.of new dimension
n -> number of original dimension

4. Choose the first k dimensions of eighen vector U
U_reduce = U[k,:]

4. Transform the data
z = U_reduce.X