from scipy.integrate import odeint 
import numpy as np 
import matplotlib.pyplot as plt 
import os

def simulate_and_save(mu, num_points, modified_coupling=True):
    def odes(z, t): 
        m_o = 1  
        m_a = 0.01    
        k_o = 1   
        k_a = 0.01    
        mu0 = 1  

        x1, x2, y1, y2 = z

        dx1dt = x2
        dx2dt = (-k_o * x1 + mu * (y1 - x1)) / m_o

        if modified_coupling:
            alpha = mu / (mu + mu0)
            k_eff = (1 - alpha) * k_a + alpha * k_o
            dy2dt = (-k_eff * y1 + mu * (x1 - y1)) / m_a
        else:
            dy2dt = (-k_a * y1 + mu * (x1 - y1)) / m_a

        dy1dt = y2

        return [dx1dt, dx2dt, dy1dt, dy2dt]

    z0 = [0, 10, 0, 1]
    t = np.linspace(0, 100, num_points)
    z = odeint(odes, z0, t)

    x = z[:, 0]
    y = z[:, 2]

    # Save to custom directory
    output_dir = os.path.expanduser("~/data_thesis/lin_oscillator")
    os.makedirs(output_dir, exist_ok=True)
    filename = f"data_mu{mu}_n{num_points}.txt"
    with open(os.path.join(output_dir, filename), 'w') as file: 
        file.write("time ocean atmosphere\n")
        for i in range(len(t)):
            file.write(f"{t[i]} {x[i]} {y[i]}\n")

# Generate 30 datasets for mu=0 with increasing number of time points
point_counts = np.logspace(3, 5, 20, dtype=int)  # from 1e3 to 1e6
for n in point_counts:
    simulate_and_save(mu=0.1, num_points=n, modified_coupling=False)
