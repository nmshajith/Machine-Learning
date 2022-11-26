Entropy H
  
    (i=1tonum_label) -Σ P(xi)*log2(P(x1))
  
Information Gain 

     H(root)-( (W_left*H(left)) + (W_right*H(right)) )


For each node, loop through the possible features and its thresholds,

calculate the Information gain for that (feature, threshold) set.

For whichever (feature, threshold) set, information gain is high, choose that as the splitting 
feature of that node


Do the above step, till the stopping criteria is met

Stopping criteria:

1. All of the node has the same label

2. Max depth has been reached

3. Num_of_sample in a node is < input_min_samples provided


While predicting, for each sample of X_test, traverse through the root and find the appropriate leaf node
