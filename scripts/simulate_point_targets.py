import numpy as np
import matplotlib.pyplot as plt

lambda_0 = 0.03
R0 = 600e3
incidence_angle = 20*np.pi/180
H = R0*np.sin(incidence_angle)
Y = R0*np.cos(incidence_angle)

aperture_time = 20.0
PRF = 50
speed_of_light = 3e8
B = 300e6

dr = speed_of_light / (4 * B)
scene_length = 1000
r_scene = np.arange(-scene_length/2, scene_length/2, dr)
t_scene = 2*(R0+r_scene)/speed_of_light

azimuth_time = np.arange(-aperture_time/2, aperture_time/2, 1/PRF)
v_a = 7000

x_target = 0
v_x_list = np.arange(-40, 46, 5)

n_pairs = 20
idx = np.linspace(0, len(azimuth_time) - 1, n_pairs).astype(int)
colors = plt.cm.viridis(np.linspace(0, 1, n_pairs))
theta = np.arctan(v_a * azimuth_time / R0)

v_x = 5.0
x_target_true_position = x_target + v_x * azimuth_time
y_target_true_position = np.zeros_like(azimuth_time)
v_x_slantrange_component = v_x * v_a * azimuth_time / R0
displacement_azimuth = (Y / v_a) * v_x_slantrange_component
disp_x = displacement_azimuth * np.cos(theta)
disp_y = displacement_azimuth * np.sin(theta)
x_target_observed_position = x_target_true_position + disp_x
y_target_observed_position = y_target_true_position + disp_y

plt.figure()
plt.plot(azimuth_time, v_x_slantrange_component)
plt.xlabel('Azimuth time (s)')
plt.ylabel('v_y_r (m/sn)')
plt.title('v_y_r vs. azimuth time')

plt.figure()
plt.plot(azimuth_time, x_target_true_position)
plt.xlabel('Azimuth time (s)')
plt.ylabel('Azimuth (m)')
plt.title(f'True azimuth position vs. time (v_y = {v_x} m/s)')

plt.figure()
for k, i in enumerate(idx):
    plt.plot(
        [x_target_true_position[i], x_target_observed_position[i]],
        [y_target_true_position[i], y_target_observed_position[i]],
        linestyle='--', color=colors[k], linewidth=0.8,
    )
plt.scatter(
    x_target_true_position[idx], y_target_true_position[idx],
    c=colors, marker='o', label=f'True position (v_y = {v_x} m/s)',
)
plt.scatter(
    x_target_observed_position[idx], y_target_observed_position[idx],
    c=colors, marker='x', label=f'Observed position (v_y = {v_x} m/s)',
)
plt.xlabel('Azimuth (m)')
plt.ylabel('Slant-range offset (m)')
plt.title(f'Target position vs. azimuth (v_y = {v_x} m/s)')
plt.legend()
plt.grid(True, alpha=0.3)

n_v = len(v_x_list)
n_cols = 6
n_rows = int(np.ceil(n_v / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 10), sharex=True, sharey=True)
axes_flat = axes.ravel()

for ax_i, v_x in enumerate(v_x_list):
    ax = axes_flat[ax_i]
    x_true = x_target + v_x*azimuth_time
    y_true = np.zeros_like(azimuth_time)
    v_x_r = v_x * v_a * azimuth_time / R0
    disp_azi = (Y/v_a) * v_x_r
    disp_x = disp_azi * np.cos(theta)
    disp_y = disp_azi * np.sin(theta)
    x_obs = x_true + disp_x
    y_obs = y_true + disp_y

    for k, i in enumerate(idx):
        ax.plot(
            [x_true[i], x_obs[i]], [y_true[i], y_obs[i]],
            linestyle='--', color=colors[k], linewidth=0.6,
        )
    ax.scatter(x_true[idx], y_true[idx], c=colors, marker='o', s=18)
    ax.scatter(x_obs[idx], y_obs[idx], c=colors, marker='x', s=25)
    ax.set_title(f'Azimuth Velocity = {v_x:g} m/s', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.4, alpha=0.4)
    ax.axvline(0, color='k', linewidth=0.4, alpha=0.4)
    ax.tick_params(labelbottom=True, labelleft=True)
    ax.set_xlabel('Azimuth (m)')
    ax.set_ylabel('Slant-range offset (m)')

for ax_i in range(n_v, len(axes_flat)):
    axes_flat[ax_i].set_visible(False)

handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=7, label='True position'),
    plt.Line2D([0], [0], marker='x', color='gray', linestyle='', markersize=7, label='Observed position'),
]
fig.legend(handles=handles, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.00))
fig.suptitle('Target position vs. azimuth — sweep over along-track velocity v_x', y=1.02, fontsize=13)

plt.tight_layout()
plt.show()
