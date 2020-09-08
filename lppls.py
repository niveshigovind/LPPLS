from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import least_squares, minimize
from sklearn import linear_model
import random
import cma
import datetime
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import multiprocessing as mp



def f(tc, t, beta):
    return (abs(tc-t))**beta
def g(tc, t, beta, omega):
    return f(tc, t, beta)*np.cos(omega*np.log(abs(tc-t)))
def h(tc, t, beta, omega):
    return f(tc, t, beta)*np.sin(omega*np.log(abs(tc-t)))

def LPPL(t, A, B, C1, C2, beta, omega, tc):
    return A + B*f(tc, t, beta) + C1*g(tc, t, beta, omega) + C2*h(tc, t, beta, omega)

def XOLS(beta, omega, tc, log_price):
    one_col = np.ones(len(log_price))

    t = np.arange(1,len(log_price)+1)
    f_col = f(tc=tc, t=t, beta=beta)
    g_col = g(tc=tc, t=t, beta=beta, omega=omega)
    h_col = h(tc=tc, t=t, beta=beta, omega=omega)

    X = np.array([one_col,f_col,g_col,h_col]).T

    return np.linalg.matrix_rank(X), X

def fit_ABC(beta, omega, tc, log_price):
    r, X = XOLS(beta, omega, tc, log_price)
    if r > 0:
        A, B, C1, C2 = np.linalg.lstsq(X, log_price)[0]
    else:
        A = np.nan
        B = np.nan
        C1 = np.nan
        C2 = np.nan
    return {"A":A, "B":B, "C1":C1, "C2":C2}

def wrapper_scaling(x,log_price):
    tc_s = x[0]
    beta_s = x[1]
    omega_s = x[2]
    return [len(log_price) + len(log_price)*0.2*np.cos(np.pi*tc_s/10), 1-np.cos(np.pi*beta_s/10), 1 + 49*(1-np.cos(np.pi*omega_s/10))/2]

def fit_cma(log_price):
    init_limits = [
        (4, 6), 
        (0, 5),                
        (1, 3),                     
    ]
    non_lin_vals = [random.uniform(a[0], a[1]) for a in init_limits]
    tc_0 = non_lin_vals[0]
    beta_0 = non_lin_vals[1]
    omega_0 = non_lin_vals[2]
    seed = np.array([tc_0, beta_0, omega_0])
        
    opti_sol = cma.fmin(lambda x: func(x,log_price),sigma0=2,x0=seed, options={ 'popsize':100})
    tc, beta, omega = wrapper_scaling(opti_sol[0],log_price)
    fit_res = fit_ABC(beta, omega, tc, log_price)
    normed_residual = func([tc, beta, omega],log_price)/len(log_price)
    sol = {'tc':tc, 'beta':beta, 'omega':omega, 'A':fit_res['A'], 'B':fit_res['B'], 'C1':fit_res['C1'], 'C2':fit_res['C2'], 'Res':normed_residual}
    return sol

def func(x, log_price):
    '''
    finds the least square difference
    '''
    tc, beta, omega = wrapper_scaling(x,log_price)
    fit = fit_ABC(beta, omega, tc, log_price)

    t = np.arange(len(log_price))
    return np.sum((LPPL(t=t, A=fit['A'], B=fit['B'], C1=fit['C1'], C2=fit['C2'], beta=beta, omega=omega, tc=tc)-log_price)**2)

def total_return(log_price):
    price = np.array(np.exp(log_price))
    return (price[-1]-price[0])/price[0]

def get_dt_max(sols, time_windows):
    residuals = [sol['Res'] for sol in sols]
    residuals = pd.DataFrame(residuals).ffill().bfill().values.flatten()
    residuals_lgrn = obtainLagrangeRegularizedNormedCost(residuals, time_windows)
    return time_windows[np.argmin(residuals_lgrn)]


def obtainLagrangeRegularizedNormedCost(residuals, time_windows):
    slope = LagrangeMethod(residuals, time_windows)
    residuals_lgrn = residuals - slope*np.array(list(time_windows))
    return residuals_lgrn

def LagrangeMethod(residuals, time_windows):
    slope = calculate_slope_of_normed_cost(residuals, time_windows)
    return slope[0]

def calculate_slope_of_normed_cost(residuals, time_windows):
    regr =linear_model.LinearRegression(fit_intercept=False)
    x_residuals = np.array(list(time_windows))
    x_residuals = x_residuals.reshape(len(residuals),1)
    res = regr.fit(x_residuals, residuals)
    return res.coef_

def conditions_satisfied(beta, omega, tc, A, B, C1, C2, dt):
    c1 = 0.01 < beta < 1.2
    c2 = 2 < np.abs(omega) < 25
    #c3 = 0.95*dt < tc < 1.11*dt
    #print('tc in interval: ' + str(c3))
    C = np.sqrt(C1**2+C2**2) 
    c3 = abs(C) < 1
    print('Amplitude of log-oscillations: ' + str(c3))
    #c4 = 2.5 < np.abs(omega)/(2*np.pi)*np.log(abs(tc/(tc-dt)))
    #print('number oscillation: ' + str(c4))
    c5 = 0.2 < beta * abs(B) / (omega * abs(C))
    print('damping: ' + str(c5) )
    # return c1 and c2 and c3 and c4 and c5 
    return c1 and c2 and c3 and c5 

def LPPL_confidence(log_price, time_windows):
    
    pool = mp.Pool(mp.cpu_count())
    sols = pool.map(fit_cma, [log_price[-dt:] for dt in time_windows])
    pool.close()

    dt_max = get_dt_max(sols, time_windows)
    print('dt_max:' + str(dt_max))
    LPPL_confidences = []
    total_returns = []
    for dt in time_windows:
        sol = sols.pop(0)
        if dt <= dt_max: 
            if conditions_satisfied(beta=sol['beta'], omega=sol['omega'], tc=sol['tc'], A=sol['A'], B=sol['B'], C1=sol['C1'], C2=sol['C2'], dt=dt):
                LPPL_confidences.append(1)
                total_returns.append(total_return(log_price[-dt:]))
            else:
                LPPL_confidences.append(0)
    print(LPPL_confidences)
    return np.mean(LPPL_confidences)*np.sign(np.median(total_returns))

def LPPL_confidence_signal(log_price, time, time_windows):
    
    LPPL_confidence_ts = []
    for t2 in time:
        print('t2:' + str(t2))
        LPPL_confidence_ts.append(LPPL_confidence(log_price=log_price[:t2], time_windows=time_windows))
    return pd.DataFrame(LPPL_confidence_ts,index=time).fillna(0)


def plot_lppls(log_price,sig,dates):

    df=pd.concat([pd.DataFrame(log_price), sig], axis=1, sort=False).set_index(dates)
    df.columns= ['log_price','sig']


    host = host_subplot(111, axes_class=AA.Axes)
    plt.subplots_adjust(right=0.75)

    par2 = host.twinx()

    offset = 0
    new_fixed_axis = par2.get_grid_helper().new_fixed_axis
    par2.axis["right"] = new_fixed_axis(loc="right",
                                        axes=par2,
                                        offset=(offset, 0))

    par2.axis["right"].toggle(all=True)


    host.set_xlabel("time")
    host.set_ylabel("price")
    par2.set_ylabel("lppls")

    p1, = host.plot(np.exp(df['log_price']), label="price")
    p3, = par2.plot(df['sig'].dropna(), label="lppls")


    host.legend()

    host.axis["left"].label.set_color(p1.get_color())
    par2.axis["right"].label.set_color(p3.get_color())
    plt.draw()
    plt.savefig('lppls.eps')
    plt.show()
    return True

start = datetime.datetime(2016, 1, 1)
end = datetime.datetime(2020, 8, 30)

price = data.DataReader(['sp500'], 'fred', start, end).ffill().bfill()
log_price = np.log(price)
log_price = log_price.values


sig=LPPL_confidence_signal(log_price=log_price, time=[700], time_windows=range(300,745,5))
np.savetxt('sig.csv', sig, delimiter=',')
np.savetxt('log_price.csv', log_price, delimiter=',')

plot_lppls(log_price=log_price,sig=sig,dates=price.index)