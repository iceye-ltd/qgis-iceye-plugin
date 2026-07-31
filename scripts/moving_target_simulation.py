import numpy as np
import matplotlib.pyplot as plt

lambda_0 = 0.03
R0 = 600e3
incidence_angle = 20*np.pi/180
H = R0*np.sin(incidence_angle)
Y = R0*np.cos(incidence_angle)

aperture_time = 20.0
PRF = 5000
speed_of_light = 3e8
B = 300e6

dr = speed_of_light / (4 * B)
scene_length = 1000
r_scene = np.arange(-scene_length/2, scene_length/2, dr)
t_scene = 2*(R0+r_scene)/speed_of_light

azimuth_time = np.arange(-aperture_time/2, aperture_time/2, 1/PRF)
v_a = 7000
x_target = [0,5,-5,0]
y_target = [0,5,7.5,20]
v_y = [0.0,0.0,0.0,5.0]
v_x = [0.0,5.0,-5.0,0.0]
a_x = [0.0,0.0,0.0,0.0]
a_y = [0.0,0.0,0.0,0.0]
s = np.zeros((len(t_scene), len(azimuth_time)), dtype=np.complex64)
for k in range(len(x_target)):
    R = np.sqrt(
        H**2
        + (Y - y_target[k] - v_y[k]*azimuth_time - 0.5*a_y[k]*azimuth_time**2)**2
        + (v_a*azimuth_time - x_target[k] - v_x[k]*azimuth_time - 0.5*a_x[k]*azimuth_time**2)**2
    )
    for j in range(len(azimuth_time)):
        rc_term = np.sinc(B * (t_scene - 2*R[j]/speed_of_light))
        azimuth_phase_term = np.exp(-1j*4*np.pi/lambda_0*R[j])
        s[:, j] += rc_term * azimuth_phase_term

# Back-projection focusing onto a small image grid centered on the target.
scene_length = 30
x_img = np.arange(-200/2, 200/2, 0.25)
r_img = np.arange(-500/2, 500/2, 0.25)

img = np.zeros((len(r_img), len(x_img)), dtype=np.complex64)

t_min = t_scene[0]
dt = t_scene[1] - t_scene[0]
N_r = len(t_scene)
N_az = len(azimuth_time)
az_idx = np.arange(N_az)

for i, x in enumerate(x_img):
    R_az = np.sqrt(H**2 + Y**2 + (v_a*azimuth_time - x)**2)
    R_full = R_az[None, :] + r_img[:, None]
    idx = np.round((2*R_full/speed_of_light - t_min) / dt).astype(np.int64)
    valid = (idx >= 0) & (idx < N_r)
    idx_clipped = np.clip(idx, 0, N_r - 1)
    contrib = s[idx_clipped, az_idx[None, :]] * np.exp(1j*4*np.pi/lambda_0*R_full)
    contrib[~valid] = 0
    img[:, i] = contrib.sum(axis=1)

img_db = 20*np.log10(np.abs(img) / np.abs(img).max() + 1e-6)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
im = axes[0].imshow(
    img_db, aspect='auto', vmin=-40, vmax=0, cmap='viridis',
    extent=[x_img[0], x_img[-1], r_img[-1], r_img[0]],
)
axes[0].set_title('Focused image (back-projection)')
axes[0].set_xlabel('Azimuth (m)')
axes[0].set_ylabel('Slant-range offset (m)')
fig.colorbar(im, ax=axes[0], label='dB')

# Ground cross-range y maps to slant-range offset by -y*cos(theta) (first-order, at broadside).
colors = plt.get_cmap('tab10').colors
n_sub = 25
idx_sub = np.linspace(0, len(azimuth_time) - 1, n_sub, dtype=int)
i_center = len(azimuth_time) // 2
for k in range(len(x_target)):
    xp = x_target[k] + v_x[k]*azimuth_time + 0.5*a_x[k]*azimuth_time**2
    yp = y_target[k] + v_y[k]*azimuth_time + 0.5*a_y[k]*azimuth_time**2
    slant = -yp * np.cos(incidence_angle)
    c = colors[k % 10]
    axes[0].plot(xp[idx_sub], slant[idx_sub], 'o', color=c, ms=4, alpha=0.55, label=f'v_x={v_x[k]:g}, v_y={v_y[k]:g}')
    axes[0].plot(xp[i_center], slant[i_center], 'o', color=c, ms=10, mec='k', mew=1.0)

axes[0].set_xlim(x_img[0], x_img[-1])
axes[0].set_ylim(r_img[-1], r_img[0])
axes[0].set_title('True target positions (big dot = t=0)')
axes[0].set_xlabel('Azimuth (m)')
axes[0].set_ylabel('Slant-range offset (m)')
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc='best', fontsize=8)

im = axes[1].imshow(
    img_db, aspect='auto', vmin=-40, vmax=0, cmap='viridis',
    extent=[x_img[0], x_img[-1], r_img[-1], r_img[0]],
)
axes[1].set_title('Focused image (back-projection)')
axes[1].set_xlabel('Azimuth (m)')
axes[1].set_ylabel('Slant-range offset (m)')
fig.colorbar(im, ax=axes[1], label='dB')

plt.tight_layout()
plt.show()
