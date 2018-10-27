def aperceptron_sgd(X, Y,epochs):    
    # initialize weights
    w = np.zeros(X.shape[1] )
    u = np.zeros(X.shape[1] )
    b = 0
    beta = 0

    # counters    
    final_iter = epochs
    c = 1
    converged = False

    # main average perceptron algorithm
    for epoch in range(epochs):
        # initialize misclassified
        misclassified = 0

        # go through all training examples
        for  x,y in zip(X,Y):
            h = y * (np.dot(x, w) + b)

            if h <= 0:
                if beta+c > 0:
                    w_hat = (beta*w_hat + c *w)/(beta+c)
                beta = beta +c
                w = w + y*x
                c=0
            else:c+=1
    if c>0
        w_hat = (beta*w_hat + c *w)/(beta+c)


                u = u+ y*c*x
                beta = beta + y*c
                misclassified += 1

        # update counter regardless of good or bad classification        
        c = c + 1

        # break loop if w converges
        if misclassified == 0:
            final_iter = epoch
            converged = True
            print("Averaged Perceptron converged after: {} iterations".format(final_iter))
            break

    if converged == False:
        print("Averaged Perceptron DID NOT converged.")

    # prints
    # print("final_iter = {}".format(final_iter))
    # print("b, beta, c , (b-beta/c)= {} {} {} {}".format(b, beta, c, (b-beta/c)))
    # print("w, u, (w-u/c) {} {} {}".format(w, u, (w-u/c)) )


    # return w and final_iter
    w = w - u/c
    b = np.array([b- beta/c])
    w = np.append(b, w)

    return w, final_iter

