from scipy.integrate import odeint 
import numpy as np 
import matplotlib.pyplot as plt 
import os

def simulate_and_save(mu):
    def odes(z, t): 
        # constants 
        m_o = 1  # ocean mass 
        m_a = 0.01     # atmosphere mass
        k_o = 1   # ocean spring constant
        k_a = 1    # atmosphere spring constant

        # parameters for modified model
        mu0 = 1  # transition sharpness for alpha(mu)

        # unpack state vector
        x1 = z[0]
        x2 = z[1] 
        y1 = z[2]
        y2 = z[3]

        dx1dt = x2
        dx2dt = ( -k_o * x1 + mu * (y1 - x1) ) / m_o
        dy1dt = y2
        dy2dt = (-k_a * y1 + mu * (x1 - y1)) / m_a

        return [dx1dt, dx2dt, dy1dt, dy2dt]

    # initial conditions 
    z0 = [0, 2, 0, 1]

    # time vector
    dt = 0.01
    tmax = 600
    nt = int(tmax / dt)
    t = np.linspace(0, tmax, nt)
    z = odeint(odes, z0, t)

    x = z[:, 0]   # ocean position
    xdot = z[:, 1]
    y = z[:, 2]   # atmosphere position
    ydot = z[:, 3]

    # Plot time series
    plt.figure(figsize=(10, 4))
    plt.plot(t, x, label="ocean")
    plt.plot(t, y, label="atmosphere")
    plt.title(f"Time Series (mu={mu})")
    plt.xlabel("Time")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Phase space
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(x, xdot, s=0.2, label="ocean phase space", alpha=0.6)
    plt.scatter(y, ydot, s=0.2, label="atmosphere phase space", alpha=0.6)
    plt.title(f"Phase Space (mu={mu}, modified={modified_coupling})")
    plt.legend()
    plt.tight_layout()
    #plt.show()
    """

    
    # save time series 
    output_dir = os.path.expanduser("~/data_thesis/lin_oscillator")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'data_mu{mu}.txt'), 'w') as file: 
        file.write("time ocean atmosphere\n")  # optional: add header
        for i in range(len(t)):
            file.write(f"{t[i]} {x[i]} {y[i]}\n")


# Run both original and modified models for various mu values
for mu_val in [10]:
    #simulate_and_save(mu_val, modified_coupling=False)
    simulate_and_save(mu_val)