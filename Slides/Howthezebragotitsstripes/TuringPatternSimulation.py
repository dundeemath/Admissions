"""
Particle-based 2D Gierer–Meinhardt-like model with:
 - different movement rates for activator/inhibitor
 - real-time smoothed density heatmaps (u = activator density, v = inhibitor density)
 - periodic domain; Gaussian smoothing via FFT-based convolution

Run: python particle_gm_heatmap.py
Requires: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# -------------------- PARAMETERS --------------------

#######################################
use_param_Set=43
save_filename = None #'Demo1.mp4'  # e.g. 'particle_gm_heatmap.mp4'

if use_param_Set==1: # spots
    L = 240
    density = 0.25
    N_particles = int(density * L * L)   # ≈ 3600
    steps = 180000
    plot_every = 1000
    dt =0.001

    mov_Fact=.5
    move_prob_A = dt*mov_Fact
    move_prob_I = 10.0*dt*mov_Fact
    move_prob_N = 100.0*dt*mov_Fact

    smooth_sigma = 2.0

    reac_kin_fact=50.0
    p_base_to_A = 2*dt*reac_kin_fact
    p_base_to_I = dt*reac_kin_fact
    k_inhibition = 1.00

    decay_A = 1.0*dt
    decay_I = 0.1*dt
    seed = 7
elif use_param_Set==2: # spots
    L = 120
    density = 0.35
    N_particles = int(density * L * L)   # ≈ 3600
    steps = 18000
    plot_every = 10000
    dt =0.001

    mov_Fact=0.5
    move_prob_A = dt*mov_Fact
    move_prob_I = 10.0*dt*mov_Fact
    move_prob_N = 100.0*dt*mov_Fact

    smooth_sigma = 2.5

    reac_kin_fact=50.0
    p_base_to_A = 2.5*dt*reac_kin_fact
    p_base_to_I = dt*reac_kin_fact
    k_inhibition = 1.00

    decay_A = 1.0*dt
    decay_I = 0.1*dt
    seed = 7
elif use_param_Set==3: # nice spots - make video
    L = 240
    density = 0.25
    N_particles = int(density * L * L)   # ≈ 3600
    steps = 180000
    plot_every = 1000
    dt =0.001

    mov_Fact=.5
    move_prob_A = dt*mov_Fact
    move_prob_I = 10.0*dt*mov_Fact
    move_prob_N = 100.0*dt*mov_Fact

    smooth_sigma = 2.0

    reac_kin_fact=50.0
    p_base_to_A = 2*dt*reac_kin_fact
    p_base_to_I = dt*reac_kin_fact
    k_inhibition = 1.00

    decay_A = 1.0*dt
    decay_I = 0.1*dt
    seed = 7
elif use_param_Set==4: # nice spots
    L = 240
    density = 0.25
    N_particles = int(density * L * L)   # ≈ 3600
    steps = 80000
    plot_every = 4000
    dt =0.001

    mov_Fact=.5
    move_prob_A = dt*mov_Fact
    move_prob_I = 4.0*dt*mov_Fact
    move_prob_N = 200.0*dt*mov_Fact

    smooth_sigma = 4.5

    reac_kin_fact=50.0
    p_base_to_A = 2.5*dt*reac_kin_fact
    p_base_to_I = dt*reac_kin_fact
    k_inhibition = 1.00

    decay_A = 1.0*dt
    decay_I = 1.0*dt
    seed = 7
    save_filename='NiceSpots.mp4'
elif use_param_Set==41: # nice spots, low density
    L = 240
    density = 0.0025
    N_particles = int(density * L * L)   # ≈ 3600
    steps = 1000
    plot_every = 1
    dt =0.001

    mov_Fact=.5
    move_prob_A = dt*mov_Fact
    move_prob_I = 4.0*dt*mov_Fact
    move_prob_N = 200.0*dt*mov_Fact

    smooth_sigma = 4.5

    reac_kin_fact=50.0
    p_base_to_A = 2.5*dt*reac_kin_fact
    p_base_to_I = dt*reac_kin_fact
    k_inhibition = 1.00

    decay_A = 1.0*dt
    decay_I = 1.0*dt
    seed = 17
    save_filename='NiceSpotsLowDensity.mp4'
elif use_param_Set==42: # nice spots
    L = 240
    density = 0.25
    N_particles = int(density * L * L)   # ≈ 3600
    steps = 160000
    plot_every = 4000
    dt =0.001

    mov_Fact=.5
    move_prob_A = dt*mov_Fact
    move_prob_I = 0.5*dt*mov_Fact
    move_prob_N = 0.5*dt*mov_Fact

    smooth_sigma = 2.5

    reac_kin_fact=50.0
    p_base_to_A = 2.5*dt*reac_kin_fact
    p_base_to_I = dt*reac_kin_fact
    k_inhibition = 1.00

    decay_A = 1.0*dt
    decay_I = 1.0*dt
    seed = 7
    save_filename='NiceSpots_DiffusionEqual.mp4'
elif use_param_Set==43: # nice spots
    L = 240
    density = 0.25
    N_particles = int(density * L * L)   # ≈ 3600
    steps = 80000
    plot_every = 4000
    dt =0.001

    mov_Fact=.5
    move_prob_A = dt*mov_Fact
    move_prob_I = 4.0*dt*mov_Fact
    move_prob_N = 1.0*dt*mov_Fact

    smooth_sigma = 2.5

    reac_kin_fact=50.0
    p_base_to_A = 2.5*dt*reac_kin_fact
    p_base_to_I = dt*reac_kin_fact
    k_inhibition = 1.00

    decay_A = 1.0*dt
    decay_I = 1.0*dt
    seed = 7
    save_filename='NiceSpots_DiffusionEqualHmm.mp4'

elif use_param_Set==5: # spots
    L = 240
    density = 0.25
    N_particles = int(density * L * L)   # ≈ 3600
    steps = 180000
    plot_every = 1000
    dt =0.001

    mov_Fact=.5
    move_prob_A = dt*mov_Fact
    move_prob_I = 0.1*dt*mov_Fact
    move_prob_N = 200.0*dt*mov_Fact

    smooth_sigma = 3.0
    reac_kin_fact=50.0
    p_base_to_A = 7.5*dt*reac_kin_fact
    p_base_to_I = dt*reac_kin_fact
    k_inhibition = 1.00

    decay_A = 1.0*dt
    decay_I = 1.0*dt
    seed = 7    
else:
    L = 120
    density = 0.25
    N_particles = int(density * L * L)   # ≈ 3600
    steps = 180000
    plot_every = 1000
    dt =0.0001

    mov_Fact=0.715
    move_prob_A = dt*mov_Fact
    move_prob_I = 30.0*dt*mov_Fact
    move_prob_N = 0.0*dt*mov_Fact

    smooth_sigma = 4.0

    reac_kin_fact=50.0
    p_base_to_A = 4*dt*reac_kin_fact
    p_base_to_I = dt*reac_kin_fact
    k_inhibition = 0.1

    decay_A = 1.0*dt
    decay_I = 1.0*dt
    seed = 7


    #######################################
   
#######################################

# Smoothing (Gaussian kernel) for density fields u, v

# Plotting parameters
cmap_u = "hot"          # colormap for activator density
cmap_v = "viridis"      # colormap for inhibitor density
show_colorbars = True

# Optional: save animation to file (set filename or None to skip saving)
# ----------------------------------------------------

np.random.seed(seed)

# Initialize particles randomly on the lattice
xs = np.random.randint(0, L, size=N_particles)
ys = np.random.randint(0, L, size=N_particles)
states = np.zeros(N_particles, dtype=np.int8)  # 0 neutral, +1 activator, -1 inhibitor

# Seed a small set of activators and inhibitors
nA0 = int(0.3 * N_particles)
nI0 = int(0.3 * N_particles)
states[:nA0] = 1
states[nA0:nA0+nI0] = -1

# Precompute movement offsets (4-neighbor)
move_offsets = np.array([[1,0],[-1,0],[0,1],[0,-1]])

# Utility: occupancy counts for activator/inhibitor per lattice site
def occupancy_counts(xs, ys, states, L):
    idx = ys * L + xs
    n_sites = L * L
    nA = np.bincount(idx[states == 1], minlength=n_sites).reshape((L, L))
    nI = np.bincount(idx[states == -1], minlength=n_sites).reshape((L, L))
    return nA, nI

# Utility: periodic Gaussian kernel in Fourier domain
def gaussian_kernel_fft(L, sigma):
    """Return FFT(kernel) for periodic Gaussian kernel on LxL grid."""
    # coordinates with periodic wrap (centered at 0)
    x = np.arange(L)
    x = np.where(x <= L//2, x, x - L)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    kernel = np.exp(-(X**2 + Y**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    K_fft = np.fft.fft2(kernel)
    return K_fft

# Precompute kernel FFT
K_fft = gaussian_kernel_fft(L, smooth_sigma)

# Function to smooth integer field via periodic convolution (FFT)
def smooth_field(field, K_fft):
    F = np.fft.fft2(field)
    conv = np.fft.ifft2(F * K_fft).real
    return conv

# Prepare figure: 1 row x 3 cols -> u heatmap, v heatmap, scatter
fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
ax_u, ax_v, ax_sc = axes

# Initial occupancy and fields
nA, nI = occupancy_counts(xs, ys, states, L)
u = smooth_field(nA, K_fft)
v = smooth_field(nI, K_fft)

# Create imshow artists (set vmin/vmax adaptively later)
im_u = ax_u.imshow(u, origin='lower', interpolation='nearest', cmap=cmap_u, extent=[0, L, 0, L])
ax_u.set_title("Activator density (u)")
ax_u.set_xticks([])
ax_u.set_yticks([])
if show_colorbars:
    plt.colorbar(im_u, ax=ax_u, fraction=0.046, pad=0.04)

im_v = ax_v.imshow(v, origin='lower', interpolation='nearest', cmap=cmap_v, extent=[0, L, 0, L])
ax_v.set_title("Inhibitor density (v)")
ax_v.set_xticks([])
ax_v.set_yticks([])
if show_colorbars:
    plt.colorbar(im_v, ax=ax_v, fraction=0.046, pad=0.04)

# Scatter (overlay particle positions)
colors = np.empty((N_particles, 3))
colors[states == 0] = [0.7, 0.7, 0.7]
colors[states == 1] = [1.0, 0.25, 0.25]
colors[states == -1] = [0.2, 0.4, 1.0]
sc = ax_sc.scatter(xs, ys, s=6, c=colors, edgecolors='face', alpha=0.9)
ax_sc.set_xlim(-0.5, L-0.5)
ax_sc.set_ylim(-0.5, L-0.5)
ax_sc.set_aspect('equal')
ax_sc.set_title("Particles (A red, I blue)")
ax_sc.set_xticks([])
ax_sc.set_yticks([])

# For better visualization you might want to normalize/impose vmin/vmax for heatmaps:
u_vmax = max(1.0, u.max())
v_vmax = max(v.min(), 0.25*v.max())
im_u.set_clim(0, max(1.0, u_vmax))
im_v.set_clim(0, max(1.0, v_vmax))

# Update function for animation
frame_counter = 0
def update(frame):
    global xs, ys, states, u_vmax, v_vmax, frame_counter
    
    
    for substep in range(plot_every):
        # ---- Movement (state-dependent rates) ----
        rv = np.random.rand(N_particles)
        moving_A = (states == 1) & (rv < move_prob_A)
        moving_I = (states == -1) & (rv < move_prob_I)
        moving_N = (states == 0) & (rv < move_prob_N)
        moving = moving_A | moving_I | moving_N
        n_moving = moving.sum()
        if n_moving > 0:
            choices = np.random.randint(0, 4, size=n_moving)
            offs = move_offsets[choices]
            xs[moving] = (xs[moving] + offs[:, 0]) % L
            ys[moving] = (ys[moving] + offs[:, 1]) % L

        # ---- Reactions ----
        nA, nI = occupancy_counts(xs, ys, states, L)
        # Local sums can be taken as small neighborhood counts (r_neigh)
        # We'll use the smoothed fields for reaction propensities (hybrid approach)
        # Compute smoothed local fields (cheap here because size moderate)
        u_field = smooth_field(nA, K_fft)
        v_field = smooth_field(nI, K_fft)

        # Sample reaction probabilities at particle positions (using smoothed fields)
        locA_p = u_field[ys, xs]
        locI_p = v_field[ys, xs]

        prob_to_A = p_base_to_A * (locA_p**2) / (1.0 +locI_p/k_inhibition)
        prob_to_A = np.clip(prob_to_A, 0.0, 1.0)

        prob_to_I = p_base_to_I * (locA_p**2)
        prob_to_I = np.clip(prob_to_I, 0.0, 1.0)

        r1 = np.random.rand(N_particles)
        become_A = (r1 < prob_to_A)
        # priority: becoming A overrides becoming I in same step
        states[become_A] = 1

        r2 = np.random.rand(N_particles)
        become_I = (r2 < prob_to_I) & (~become_A)
        states[become_I] = -1

        # spontaneous decay
        r3 = np.random.rand(N_particles)
        decayA_mask = (states == 1) & (r3 < decay_A)
        decayI_mask = (states == -1) & (np.random.rand(N_particles) < decay_I)
        states[decayA_mask] = 0
        states[decayI_mask] = 0

        frame_counter += 1

    # After plot_every substeps, update visualizations
    nA, nI = occupancy_counts(xs, ys, states, L)
    u = smooth_field(nA, K_fft)
    v = smooth_field(nI, K_fft)

    # adapt color scale a little if maxima increase (keeps contrast)
    u_vmax = max(u_vmax, u.max())
    v_vmax = max(v_vmax, v.max())
    im_u.set_data(u)
    im_v.set_data(v)
    im_u.set_clim(0, u_vmax)
    im_v.set_clim(0, v_vmax)

    # update scatter colors and positions
    colors = np.empty((N_particles, 3))
    colors[states == 0] = [0.7, 0.7, 0.7]
    colors[states == 1] = [1.0, 0.25, 0.25]
    colors[states == -1] = [0.2, 0.4, 1.0]
    sc.set_offsets(np.column_stack((xs, ys)))
    sc.set_facecolor(colors)
    sc.set_edgecolors(colors)

    # update titles
    ax_u.set_title(f"Activator density (u) — step {frame_counter}")
    ax_v.set_title(f"Inhibitor density (v) — step {frame_counter}")
    ax_sc.set_title(f"Particles (A red, I blue) — step {frame_counter}")
    return im_u, im_v, sc

# Create animation
nframes = max(1, int(steps /plot_every))

print(nframes)


anim = animation.FuncAnimation(fig, update, frames=nframes, interval=50, blit=False, repeat=False)

#plt.show()

# Optional saving (uncomment and set save_filename to use)
if save_filename is not None:
    print("Saving animation (this may take a while)...")

    anim.save(save_filename, fps=20, dpi=150)
    print("Saved to", save_filename)
