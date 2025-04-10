# integration of a coupled bidimensional linear oscillator simulating ocean and atmosphere 

from scipy.integrate import odeint 
import numpy as np 
import matplotlib.pyplot as plt 

def integrate(mu):
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
    z0 = [0,0.2,0,1]

    # test defined ODEs
    #print(odes(z = z0, t = 0))

    # declare a time vector 
    t = np.linspace(0,10000,10000) 
    z = odeint(odes, z0,t)

    x = z[:,0] # oceano 
    xdot = z[:,1]
    y = z[:,2] # atmosfera
    ydot = z[:,3]


    #plot 
    plt.plot(t,x, label = "ocean")
    plt.show()
    plt.plot(t,y, label = "atmosphere")
    plt.legend()

    plt.show()

    plt.scatter(x,xdot, s = 0.1, label = "ocean phase space")
    
    plt.scatter(y,ydot, s = 0.1, label = "atmosphere phase space")
    plt.legend()

    plt.show()


integrate(0)
integrate(0.5)
integrate(10)
integrate(100)
