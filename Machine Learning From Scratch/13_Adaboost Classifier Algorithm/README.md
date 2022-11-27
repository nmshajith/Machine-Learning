Get the number of stumps to be created (n_stumps)

    Initialize weights for each training sample = 1/num_of_samples

for stump in n_stumps:

    (i) Train weak classifier/stump (greedy search to find the best feature and threshold) (Whichever (feature,threshold) has the least error is choosen as the best feature and threshold)

    (ii) Calculate the error of that stump E_stump = (miss)Σweights -->(Add the weights of misclassified samples as the error of that stump)

    (iii) Flip error and decision if error >0.5

    (iv) Calculate Amount of say alpha = 0.5*log( (1-E_stump)/(E_stump) )

    (v) Update weights w = w*exp( -alpha*y*h(X) / sum(w) ) ; h(X) -> Prediction of X by that stump
