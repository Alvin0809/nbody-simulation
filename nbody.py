from vpython import sphere, vector, rate, canvas
import numpy as np
import time
from random import random, uniform
import math

# Function to generate a random direction

def randomdirection():
    ra = 2.0 * math.pi * random()
    dec = math.acos(2.0 * random() - 1.0) - 0.5 * math.pi
    return ra, dec

# Function to generate a random unit vector
def ranvec():
    theta, phi = randomdirection()
    z = math.sin(phi)
    x = math.cos(phi) * math.sin(theta)
    y = math.cos(phi) * math.cos(theta)
    return np.array([x, y, z])

# Constants
Nstars = 200
G = 6.7e-11  # Gravitational constant
Msun = 1.5E30
Rsun = 3E8
L = 4e10
vsun = 0.9 * math.sqrt(G * Msun / Rsun)
h0 = 1.0e-5  # Hubble's constant

display_scene = canvas(title="Stars", width=1320, height=830)
Stars = []
poslist, plist, mlist, rlist = [], [], [], []
p0 = Msun * 100000.0

# Initialize stars
for i in range(Nstars):
    vec = L * ranvec() * (random() ** 0.3333)
    x, y, z = vec
    r = Rsun
    from vpython import vector  # Ensure vector is imported
    col0 = vector(uniform(0.7, 1.0), uniform(0.7, 1.0), uniform(0.7, 1.0))
    Stars.append(sphere(pos=vector(x, y, z), radius=r, color=col0))
    mass = Msun
    px, py, pz = [p0 * uniform(-1, 1) for _ in range(3)]
    poslist.append([x, y, z])
    plist.append([px, py, pz])
    mlist.append(mass)
    rlist.append(r)

pos = np.array(poslist)
p = np.array(plist)
m = np.array(mlist).reshape(Nstars, 1)
radius = np.array(rlist)

# Zero out center-of-mass velocity
vcm = np.sum(p, axis=0) / np.sum(m)
p -= m * vcm

dt = 50.0
Nsteps = 0
time_start = time.perf_counter()
Nhits = 0

import keyboard  # Install with: pip install keyboard

while True:
    rate(50)

    if keyboard.is_pressed("q"):  # Press 'Q' to quit
        print("Exiting simulation...")
        break

    L *= 1.0 + h0 * dt  # Expand universe
    con = (G * Nstars * Msun) / (L ** 3)
    
    # Compute forces
    r = pos - pos[:, np.newaxis]
    for n in range(Nstars):
        r[n, n] = np.array([1e6, 1e6, 1e6])  # Set diagonal elements to large values
    rmag = np.linalg.norm(r, axis=-1)
    hit = (rmag <= (radius + radius[:, np.newaxis])) - np.eye(Nstars)
    hitlist = np.sort(np.nonzero(hit.flat)[0]).tolist()
    F = G * m * m[:, np.newaxis] * r / rmag[:, :, np.newaxis] ** 3
    for n in range(Nstars):
        F[n, n] = np.array([0.0, 0.0, 0.0])  # Set self-force to zero
    p += np.sum(F, axis=1) * dt + pos * con * dt * m

    # Update positions
    pos += (p / m) * dt
    pos *= 1.0 + h0 * dt  # Expand with universe
    p *= 1.0 + h0 * dt  # Scale momentum accordingly

    # Update visual positions
    for i in range(Nstars):
        Stars[i].pos = vector(*pos[i])

    # Handle collisions
    for ij in hitlist:
        i, j = divmod(ij, Nstars)
        if not Stars[i].visible or not Stars[j].visible:
            continue
        newpos = (pos[i] * m[i, 0] + pos[j] * m[j, 0]) / (m[i, 0] + m[j, 0])
        newmass = m[i, 0] + m[j, 0]
        newp = p[i] + p[j]
        newradius = Rsun * ((newmass / Msun) ** (1 / 3))
        iset, jset = (i, j) if radius[i] >= radius[j] else (j, i)
        Stars[iset].radius = newradius
        m[iset, 0] = newmass
        pos[iset] = newpos
        p[iset] = newp
        Stars[jset].visible = False
        p[jset] = np.array([0, 0, 0])
        m[jset, 0] = Msun * 1E-30
        Nhits += 1
        pos[jset] = [10 * L * Nhits, 0, 0]  # Move merged object far away
    
    Nsteps += 1