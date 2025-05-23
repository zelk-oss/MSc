# integration of a coupled bidimensional linear oscillator simulating ocean and atmosphere 

from scipy.integrate import odeint 
import numpy as np 
import matplotlib.pyplot as plt 

def simulate_and_save(mu):
    def odes(z,t): 
        # constants 
        m_o = 10000 # ocean mass 
        m_a = 10
        k_o = 1000
        k_a = 100

        # the odes 
        # assign each ODE to a vector element 
        x1 = z[0]
        x2 = z[1] 
        y1 = z[2]
        y2 = z[3]

        # real equations 
        # m_o x'' = - k_o*x + mu*(y-x) 
        # m_a y'' = - k_a*y + mu*(x-y)

        dx1dt = x2
        dx2dt = ( -k_o * x1 + mu*(y1 - x1) ) / m_o
        dy1dt = y2
        dy2dt = ( -k_a * y1 + mu*(x1 - y1) ) / m_a

        return [dx1dt, dx2dt, dy1dt, dy2dt]

    
    # initial conditions 
    z0 = [0,1,0,10]

    # test defined ODEs
    print(odes(z = z0, t = 0))

    # declare a time vector 
    # high sampling to prevent aliasing 
    t = np.linspace(0,1000,100000) 
    z = odeint(odes, z0,t)

    x = z[:,0] # oceano 
    xdot = z[:,1]
    y = z[:,2] # atmosfera
    ydot = z[:,3]


    #plot 
    plt.plot(t[0:10000],x[0:10000], label = "ocean")
    plt.plot(t[0:10000],y[0:10000], label = "atmosphere")
    plt.legend()

    plt.show()

    plt.scatter(x,xdot, s = 0.2, label = "ocean phase space")
    
    plt.scatter(y,ydot, s = 0.2, label = "atmosphere phase space")
    plt.legend()

    plt.show()

    # save time series 
    #with open(f'data_mu{mu}.txt', 'w') as file: 
        #file.write("time ocean atmosphere\n")  # optional: add header
        #for i in range(len(t)):
            #file.write(f"{t[i]} {x[i]} {y[i]}\n")



simulate_and_save(0)
simulate_and_save(1)
simulate_and_save(10)
simulate_and_save(100)
