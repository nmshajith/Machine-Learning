
Prediction:
f(x) = w.x-b
Output 
yi = 1; if f(x)>=1
yi = -1; if f(x)<=-1

Cost Function:
J = (1/n)(i=1ton)Σmax(0, 1-yi(w.xi-b)) + lambda* ||w||**2
i.e
If yi*f(x) >= 1 (If for the vector, output predicted is correct):
Then Ji = lambda*||w||**2
If yi*f(x) < 1 (If for that input vector output predicted is wrong):
Then Ji = lambda*||w||**2 +(1-yi*(w.xi-b))   -> This is Ji (i.e) Cost function for one single input vector

Gradient:
If yi.f(x) >= 1
    dJi/dwk = 2*lambda*wk
    dJi/db  = 0
Else:
    dJi/dwk = 2*lambda*wk - yi*xi
    dJi/db = yi

Gradient Descent:
    Repeat for number of iterations{
        For each training sample xi:
        w = w-alpha*dw
        b = b-alpha*db
    }