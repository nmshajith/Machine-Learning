Naive Bayes Formula:

        P(A|B) = P(B|A).P(A) / P(B)

For Classification Problem:

        P(y|X) = P(X|y).P(y) / P(X)
    
where 

X -> Input vector containing n features
X = (x1, x2, x3, ..., xn)

y -> Output class (0 or 1)

For Naive Bayes, we assume that each feature is independent of other. So,

P(X) = P(x1).P(x2).P(x3)...P(xn)

P(y|X) = P(y).P(x1|y).P(x2|y).P(x3|y)...P(xn|y) / P(x1).P(x2).P(x3)...P(xn)

If y has two possible class - 0 and 1

P(0|X) = value1

P(1|X) = value 2

Whichever has the greater probability, that is the predicted class y, for the input X

        y = argmax_y P(y) (i=1ton)∏P(xi|y)

        y = argmax_y log(y) + log(P(x1|y)) + log(P(x2|y)) + ... + log(P(xn|y))

Each feature is assumed to be distributed by gaussian distribution

    P(xi|y) = (1/sqrt(2*pi*sigma_y**2)) * exp ( -(xi-u_y)**2 / (2*sigma_y**2))
