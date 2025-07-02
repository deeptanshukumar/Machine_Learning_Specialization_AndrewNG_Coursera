import numpy as np
import copy
'''
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

w_init = 0
b_init = 0
alpha = 0.001
num_iters = 100000
def costfn(x_train, y_train, w,b):
    j_wb = 0
    m = x_train.shape[0]
    for i in range(m):
        f_wb = w*x_train[i]+b
        j_wb = j_wb + (f_wb-y_train[i])**2
    j_wb = j_wb/(2*m)  

    return j_wb    
 
def gradientfn(x_train, y_train, w,b):
    m = x_train.shape[0]
    d_jwb_w = 0
    d_jwb_b = 0

    for i in range(m):
        f_wb = w*x_train[i] + b
        d_jwb_w = d_jwb_w + (f_wb-y_train[i])*x_train[i]
        d_jwb_b = d_jwb_b + (f_wb-y_train[i])
    d_jwb_w = d_jwb_w/m
    d_jwb_b = d_jwb_b/m

    return d_jwb_w, d_jwb_b



def gradient_descent(x_train, y_train, alpha, w_init, b_init, num_iters, costfn, gradientfn):
    w,b = w_init, b_init
    dj_dw , dj_db = gradientfn(x_train, y_train, w,b)
    jwb = []

    for i in range(num_iters):
        dj_dw , dj_db = gradientfn(x_train, y_train, w,b)
        
        w = w - alpha*dj_dw
        b = b - alpha*dj_db
        if (i%1000)==0:
            jwb.append(costfn(x_train, y_train, w,b))
    return w,b,jwb

w_final, b_final,j_wb = gradient_descent(x_train, y_train, alpha, w_init,b_init, num_iters, costfn, gradientfn)
print(f'w_final:{w_final:.2f}, b_final:{b_final:.2f}')
print(j_wb)


'''

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
alpha = 5.0e-7
num_iters = 1000

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

'''def predict_single_loop(x, y, w, b):

    n = x.shape[0]
    f_wb = 0
    for i in range(n):
        f_wb += w[i]*x[i]
    f_wb +=b
    
    return f_wb

print(predict_single_loop(X_train[0], y_train, w_init, b_init))'''
'''
def predict(x,w,b):
    f_wb = np.dot(x,w) + b
    print(f_wb)
    #here f_wb is a list of predictions, number of f_wb is equal to x.shape[0] (the number of rows)
predict(X_train, w_init,b_init)
'''


def compute_cost(X, y, w,b):
    m = X.shape[0]
    cost = 0
    for i in range(m):
        f_wb_i = np.dot(X[i],w)+b
        cost += (f_wb_i-y[i])**2

    cost = cost/(2*m)

    return cost

def compute_gradient(X, y, w,b):
    m,n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0
    for i in range(m):
        error = np.dot(X[i],w)+b  - y[i]
        for j in range(n):
            dj_dw[j] += error*X[i,j]
        dj_db += error
    dj_db /= m
    dj_dw /= m

    return dj_dw, dj_db


def gradient_descent(X_train, y_train, alpha, num_iters, w_init, b_init, compute_cost, compute_gradient):
    jwb = []
    w = copy.deepcopy(w_init)
    b = b_init
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X_train, y_train, w, b)
        w = w - alpha*dj_dw
        b = b - alpha*dj_db

        if i%100==0:
            jwb.append(compute_cost(X_train, y_train, w_init, b_init))
    
    return w,b,jwb


b_init = 0
w_init = np.zeros(X_train.shape[1])


w_final,b_final,j_wb = gradient_descent(X_train, y_train, alpha, num_iters, w_init, b_init, compute_cost, compute_gradient)
print(w_final, b_final)

print(j_wb)


f_wb = np.dot(X_train,w_final) + b_final
print(f_wb)



