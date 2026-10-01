# SAR mover relocator: the maths as implemented

This documents the maths of the two-click moving-target relocation in
`core/mover_relocation.py` and the shared geometry in `core/target_finder.py`
(branch `feat/sar-mover-two-click`, built on `feat/sar-mover-relocation` @ `e810a55`).

The earlier Doppler-centroid relocation (the previous revision's sections 5 to 7, 9 and
10) was removed; section 5 below records why.

Numbers quoted are from the WWGTZ2 SLED (Dwell) fixture,
`test/fixtures/ICEYE_WWGTZ2_20251109T141525Z_6987409_X44_SLED_CROP_37a3f6c7.tif`.

---

## 0. Notation

| Symbol | Meaning |
|---|---|
| $\lambda = c / f_c$ | wavelength (0.03107 m) |
| $\mathbf S(t), \mathbf v(t), \mathbf a(t)$ | satellite ECEF position, velocity, acceleration |
| $\mathbf P$ | Earth-fixed target position (ECEF) |
| $R = \lVert \mathbf P - \mathbf S \rVert$ | slant range |
| $t$ | zero-Doppler (azimuth) time, seconds from `iceye:zero_doppler_start_datetime` |
| $K_a$ | azimuth FM rate (Hz/s) |
| $f_{dc}(t)$ | metadata Doppler centroid |
| $v_r$ | radial velocity, **positive towards the radar** |
| $v_{gr}$ | ground-range velocity, $v_r / \sin\theta_{inc}$ |
| $v_a$ | along-track velocity, positive along the platform velocity |
| $v_t$ | ground speed of the target |
| $\hat{\mathbf g}$ | horizontal ground-range unit vector, away from the satellite |
| $\hat{\mathbf a}$ | horizontal along-track unit vector (projection of $\mathbf v$) |

---

## 1. Data layout and time axis

The SLC is read with `core.raster.read_slc_layer` as $s = A\,e^{-j\phi}$ and put back
into file layout (rows = range samples, columns = azimuth lines).

- Azimuth sample spacing: $|\Delta t_{col}| = 1 / f_s$ with $f_s$ = `iceye:processing_prf`.
- The **sign** of $\Delta t_{col}$ comes from geolocation (on the fixture, time
  *decreases* with column index). The two-click relocation never uses it: it works on
  geolocated points only (section 6).
- The GCP model spans about 1.3 % more zero-Doppler time across the crop than
  $1/f_s$ per column implies (`image_time_limits`, test on the fixture).

### 1.1 Display surface

QGIS shows the SLC warped through its GCPs (EPSG:4326). The GCP heights are the
surface of that display: -2.65 m on the fixture, against
`iceye:average_scene_height` = -3 m. Clicks are converted to ECEF on the mean GCP
height (`gcp_mean_height`), so they map back to the pixels the user sees, and the
layover of a clicked constraint matches the target's. Ships are finally geocoded on
the sea surface (`iceye:average_scene_height`); other targets stay on the display
surface.

---

## 2. Orbit and range-Doppler geometry

### 2.1 Orbit fit
The 50 `iceye:orbit_states` are fitted per ECEF axis with a degree-7 polynomial in
normalised time $u = (t - \bar t)/s$. Velocity and acceleration are its analytic
derivatives, divided by $s$ and $s^2$. The fit error is about 3 mm in position and
1.4 mm/s in velocity.

### 2.2 Zero-Doppler time of a point (range-Doppler inverse)
Solve $g(t) = \mathbf v(t)\cdot(\mathbf P - \mathbf S(t)) = 0$ by Newton:

$$
g'(t) = \mathbf a(t)\cdot(\mathbf P - \mathbf S) - \lVert\mathbf v\rVert^2,
\qquad t \leftarrow t - g/g'.
$$

Then $R = \lVert \mathbf P - \mathbf S(t)\rVert$.

### 2.3 Geocoding (range-Doppler forward)
Given $(R, t, h)$, find $\mathbf P$ from three equations by 3D Newton, starting at a
guess on the correct look side:

$$
\begin{aligned}
F_1 &= \mathbf v\cdot(\mathbf P-\mathbf S) = 0 \\
F_2 &= \lVert\mathbf P-\mathbf S\rVert^2 - R^2 = 0 \\
F_3 &= \frac{P_x^2+P_y^2}{(a+h)^2} + \frac{P_z^2}{(b+h)^2} - 1 = 0
\end{aligned}
$$

with WGS84 $a, b$. Raising the ellipsoid axes by $h$ approximates a true height-$h$
surface.

### 2.4 FM rate, effective and ground velocity

$$
V_{eff}^2 = \lVert\mathbf v\rVert^2 - \mathbf a\cdot(\mathbf P-\mathbf S),
\qquad
K_a = \frac{2 V_{eff}^2}{\lambda R},
\qquad
V_g = \lVert\mathbf v\rVert \,\lVert\mathbf P\rVert / \lVert\mathbf S\rVert .
$$

Fixture values: $K_a \approx 4920$ Hz/s at $R \approx 703$ km, and
$R V_g / V_{eff}^2 \approx 90$ s, i.e. about 90 m of azimuth shift per 1 m/s radial
velocity.

### 2.5 Local incidence and directions
At a point $\mathbf P$ with local East/North/Up basis, the vector from the satellite
$\mathbf d = \mathbf P - \mathbf S$ gives $\cos\theta_{inc} = -\hat{\mathbf d}\cdot\hat{\mathbf u}$,
$\hat{\mathbf g}$ = horizontal part of $\mathbf d$ (normalised) and $\hat{\mathbf a}$ =
horizontal part of $\mathbf v$ (normalised). The tool uses this local incidence
rather than the scene average (`local_geometry`).

---

## 3. Moving-target model

A constant-velocity target's range history is an exact hyperbola, identical (apart
from the Doppler-rate term of its along-track velocity) to that of a stationary target
at a position displaced along track. A zero-Doppler processor therefore images it at
the minimum of its range history, $(R_{min}, t_{min})$:

$$
R(\tau) \approx R_0 - v_r(\tau - t_0) + \frac{V_{eff}^2}{2R_0}(\tau-t_0)^2
\;\Rightarrow\;
\boxed{\;\Delta t = t_{img} - t_{true} = \frac{R\,v_r}{V_{eff}^2},
\qquad \Delta x = V_g\,\Delta t\;}
$$

with $t_{true} = t_0$ the zero-Doppler time of the true position.

- Displacement is purely along track. Slant range is correct to within
  $R v_r^2 / (2 V_{eff}^2)$ (6 mm at 1 m/s, about 0.9 m at 12.5 m/s); the true range is
  the longer one.
- Every possible true position lies on the target's own range line (constant slant
  range, varying zero-Doppler time): a line, not a circle or cone.
- An approaching target ($v_r > 0$) is imaged **later** in zero-Doppler time than its
  true position. This sign is verified against exact simulated range histories
  (section 8).

---

## 4. Target input: imaged position

### 4.1 Bezier
The curve editor's four control points $\mathbf p_0, \mathbf c_1, \mathbf c_2, \mathbf p_3$
are 0..1 fractions of the chip (x along columns, y along rows), sampled at 256 points
and mapped to pixels as $r = y(N_r-1)$, $c = x(N_c-1)$. A single click is the
degenerate curve with all four points at the click, so its corridor is a disc.

### 4.2 Corridor (distance in ground metres)
Pixel spacings come from a local ENU Jacobian of the GCP geolocation: ground metres per
+1 row, $\Delta_r \approx 0.43$ m, and per +1 column, $\Delta_c \approx 0.044$ m. A pixel is in
the corridor if it lies within the ellipse
$\left(\frac{\delta r}{w/\Delta_r}\right)^2 + \left(\frac{\delta c}{w/\Delta_c}\right)^2 \le 1$
of some point on the densified curve, with $w$ = `corridor_half_width_m` (15 m).

### 4.3 Ring and hull
- **Ring:** corridor of half-width $w + g + W$ minus corridor of half-width $w + g$,
  with $g = 10$ m and $W = 30$ m. Clutter level $\bar I_c$ = median intensity over it.
- **Hull:** corridor pixels with
  $I > \max(\bar I_c\,10^{\mathrm{SNR}/10},\; I_{peak}\,10^{-D/10})$, SNR = 10 dB, D = 25 dB.

Imaged position = intensity-weighted hull centroid, geocoded through the GCPs onto the
display surface and RD-inverted to $(R_{img}, t_{img})$. The target half-extent is half
the hull's spread along $\hat{\mathbf g}$ (`locate_imaged_target`).

---

## 5. Removed: Doppler-centroid radial velocity

The previous revision estimated $v_r$ from the hull's Doppler centroid (spectral
centroid or Madsen correlation), corrected it for truncation by the processing window,
and solved the reference Doppler at the true position by fixed-point iteration. All of
it was removed, because in Spotlight / Dwell the beam is steered on the scene and the
metadata centroid moves at almost exactly the FM rate:

$$
A = \frac{1}{\lvert 1 - \dot f_{dc}/K_a\rvert} \approx 3000
\qquad (\dot f_{dc} \approx 4918 \text{ vs } K_a \approx 4920 \text{ Hz/s on the fixture}).
$$

Physically, section 3 makes the mover's whole phase history that of a stationary target
at the displaced position; its spectrum is centred on the clutter Doppler at the imaged
position and carries no usable information on $v_r$. Sub-apertures do not change this
(each is still beam-steered), a Spotlight mover suffers no truncation, and the
reference-Doppler iteration converges to zero displacement.

`doppler_amplification` still computes $A$ as a diagnostic (test: $A > 500$ on the
fixture). Heading from the hull axis and $v_r$ from map drift plus that heading were
removed with it (the user decided against target-direction logic).

---

## 6. Two-click relocation

### 6.1 Possible-location band (`band_for_target`)
With target class maximum speed $v_{max}$ (car 40, train 90, ship 15 m/s):

$$
\Delta t_{max} = \frac{R_{img}\, v_{max} \sin\theta_{inc}}{V_{eff}^2}.
$$

- Centre line: geocode $(R_{img}, t, h)$ for `band_samples` (41) times over
  $[t_{img} - \Delta t_{max},\ t_{img} + \Delta t_{max}]$, plus $t_{img}$ itself.
- Edges: range lines at $R_{img} \pm w_b \sin\theta_{inc}$, with ground half-width
  $w_b$ = target half-extent + `band_margin_m` (10 m).
- Ticks at $|v_r| = k \cdot$ `tick_step_mps` (5 m/s) up to $v_{max}\sin\theta_{inc}$, on both
  sides: $t = t_{img} - R_{img} v_r / V_{eff}^2$. Label: $|v_r|$ and the implied minimum
  ground speed $|v_r| / \sin\theta_{inc}$. Approaching ticks sit at earlier $t$.
- Clipping: samples and ticks outside the image's zero-Doppler span at the target's
  row are dropped and the band is flagged `band_clipped`. Cars on the 0.05 s fixture
  crop are always clipped.
- Order of magnitude on the fixture (91.6 m per m/s, $\sin\theta_{inc} = 0.568$): ship
  bands $\pm$0.78 km, car bands $\pm$2.1 km, train bands $\pm$4.7 km.

### 6.2 Cursor readout (`cursor_readout`)
The cursor point is RD-inverted to $(R_c, t_c)$. If $|R_c - R_{img}| \le w_b\sin\theta_{inc}$,
the readout is $|v_r| = V_{eff}^2 |t_{img} - t_c| / R_{img}$ (m/s, km/h, kn) and
$|\Delta x| = V_g |t_{img} - t_c|$.

### 6.3 Constraint intersection (`intersect_constraint`)
Clicks A and B on the road / rail / bridge deck / wake axis are converted to ECEF on
the display surface and RD-inverted to $(R_A, t_A)$, $(R_B, t_B)$.

- Reject when $(R_A - R_{img})(R_B - R_{img}) > 0$ ("points must straddle the band"),
  unless `allow_extrapolation`; reject when $R_A = R_B$.
- Start from $s = (R_{img} - R_A)/(R_B - R_A)$ and refine by secant iterations on
  $R(\mathbf P(s)) = R_{img}$, with $\mathbf P(s)$ the point at fraction $s$ of the
  straight segment A-B on the display surface. This keeps long constraints exact;
  the linear formula alone would pick up the range line's curvature.
- $t_{true}$ = zero-Doppler time of $\mathbf P(s)$;
  $\mathbf P_{true}$ = geocode$(R, t_{true}, h_{target})$.
- Optional `range_residual` (default off): intersect at
  $R_{img} + R v_r^2/(2V_{eff}^2)$ instead, two passes.

### 6.4 Derived quantities (`relocate`)
Evaluated with the local geometry at $\mathbf P_{true}$:

$$
v_r = \frac{V_{eff}^2 (t_{img} - t_{true})}{R_{img}},\quad
v_{gr} = \frac{v_r}{\sin\theta_{inc}},\quad
\Delta x = V_g (t_{img} - t_{true}),
$$

$$
\cos\phi = \hat{\mathbf u}_{road}\cdot\hat{\mathbf g},\qquad
v_t = \frac{|v_r|}{|\cos\phi|\,\sin\theta_{inc}},
$$

with $\hat{\mathbf u}_{road}$ the horizontal unit vector from A to B at $\mathbf P_{true}$.
Travel direction $\hat{\mathbf u}_{dir} = \pm\hat{\mathbf u}_{road}$, chosen so that
$\operatorname{sign}(\hat{\mathbf u}_{dir}\cdot(-\hat{\mathbf g})) = \operatorname{sign}(v_r)$.
Heading = $\operatorname{atan2}(u_E, u_N)$. Predicted along-track velocity
$v_{a,pred} = v_t\,(\hat{\mathbf u}_{dir}\cdot\hat{\mathbf a})$. The click order does not
matter (test).

Epoch: $t_{true}$ in UTC from `iceye:zero_doppler_start_datetime`. Track over the
collection window (`start_datetime` .. `end_datetime`, 28.8 s on the fixture):
$\mathbf P(t) = \mathbf P_{true} + v_t\,\hat{\mathbf u}_{dir}(t - t_{true})$.

### 6.5 Uncertainty
With $\psi$ the angle between the constraint and the track and $w_c$ the constraint
width (`constraint_width_m`, 10 m; the panel has no constraint-type choice, since the
type changed nothing but this number):

$$
\sigma_{\Delta x}^2 = \sigma_{centroid}^2 + \sigma_{click}^2
+ \frac{(w_c/\sqrt{12})^2}{\sin^2\psi},\qquad
\sigma_{v_r} = \frac{V_{eff}^2\,\sigma_{\Delta x}}{V_g R},\qquad
\sigma_{v_t} = \frac{\sigma_{v_r}}{|\cos\phi|\sin\theta_{inc}},
$$

with $\sigma_{centroid} = \sigma_{click} = 2$ m by default.

### 6.6 Plausibility (`plausibility`)

| Check | Pass condition | Flag | Kind |
|---|---|---|---|
| Inside band | $\lvert\Delta x\rvert \le V_g \Delta t_{max}$ | `outside_band` | hard |
| Speed | $v_t \le v_{max}$ | `implausible_speed` | hard |
| Moving | $v_t \ge 0.5$ m/s | `probably_stationary` | soft |
| Geometry | $\psi > 15°$ | `constraint_parallel_to_track` | hard |
| $v_a$ (off by default) | same sign and $\lvert v_{a,meas} - v_{a,pred}\rvert \le 3\sigma$ | `v_a_inconsistent` | hard |
| Clipping | band not clipped | `band_clipped` | soft |

Green: no flags. Amber: soft flags only. Red: any hard flag.

### 6.7 Single-click mode (`relocate_single_click`)
The user clicks once where the constraint crosses the band. The click C is
RD-inverted to $(R_C, t_C)$ and must satisfy $|R_C - R_{img}| \le w_b \sin\theta_{inc}$;
then $t_{true} = t_C$ and $\mathbf P_{true}$ = geocode$(R, t_{true}, h_{target})$ with the
optional range residual as in 6.3. $v_r$, $v_{gr}$, $\Delta x$ and the epoch follow as
in 6.4. One point does not give $\hat{\mathbf u}_{road}$, so:

- only the minimum ground speed $v_{t,min} = |v_r| / \sin\theta_{inc}$ is known (exact
  when the motion is along ground range); $v_t$, heading, $\phi$, $\psi$, track and
  $v_{a,pred}$ are left empty;
- $\sigma_{\Delta x}^2 = \sigma_{centroid}^2 + \sigma_{click}^2 + (w_c/\sqrt{12})^2$, without
  the $1/\sin^2\psi$ term it cannot evaluate;
- plausibility uses $v_{t,min}$ for the speed checks, skips the geometry check and
  adds the soft flag `heading_unknown`, so the indicator is amber at best.

Section 6.8 optionally recovers a rough direction from the image.

### 6.8 Single click with an image-estimated axis (`estimate_constraint_axis`)
A user decision reversing the original "no automatic road / wake detection" rule, as
an option on top of 6.7. Around the click (radius `axis_radius_m`, 50 m):

1. $|s|^2$ is block-averaged to cells of about `axis_cell_m` (2 m) of ground, which
   also suppresses speckle, and $f = \log_{10}$ of it is differentiated per block.
2. Gradients are mapped to ground (East, North) with the block Jacobian
   $J = [\mathbf e_{row} b_r,\ \mathbf e_{col} b_c]$: $\nabla_{EN} f = J^{-T}\nabla_{ij} f$.
3. The Gaussian-weighted ($\sigma = 25$ m) structure tensor
   $T = \sum w\, \nabla f\, \nabla f^{\mathsf T}$ has eigenvalues $\lambda_1 \ge \lambda_2$; the
   line axis is the eigenvector of $\lambda_2$ (gradients run across the edges of a
   bright or dark linear feature), and the coherence is
   $(\lambda_1 - \lambda_2)/(\lambda_1 + \lambda_2)$.

If the coherence is at least `min_axis_coherence` (0.5), two points
`axis_half_length_m` (40 m) either side of the click along the axis replace the two
clicks of 6.3 / 6.4; the result is flagged `heading_estimated` (soft, amber at
best). Otherwise, or if the axis runs along the band, the 6.7 result stands.

Measured on synthetic 8 m features in speckle on the fixture geometry: axis errors
of about $\pm$4° (random sign) and coherence 0.82 to 0.94 for bright roads and dark
wakes, 0.07 to 0.12 for pure speckle. On the real fixture a weak edge of a port
structure next to the click gave 0.38, below the threshold. Strong straight edges
(quays, building rows) or sidelobe streaks of bright targets can still be taken for
the road, so the spawned points are drawn for the user to check. A 4° axis error
changes $v_t$ by about $\tan\phi \cdot 7\,\%$.

---

## 7. Map drift along the curve (optional $v_a$ check)

Kept from the previous revision for the $v_a$ consistency check, which is off until its
sign is validated and is not wired into the tool.

The azimuth spectrum of the window around the curve is split into $N = 5$ bands over
$[-B/2, B/2]$; look $k$ has centre Doppler $f_k$ and slow time $\tau_k = -f_k/K_a$. In
each look the hull is re-thresholded (4.3) and its centroid gives
$t_k = (c_k - c_{ref})\,\Delta t_{col}$. Along-track motion changes the FM rate,
$K_{a,t} \approx K_a(1 - 2v_a/V_{eff})$, so $t_k \approx t_0 + \frac{2v_a}{V_{eff}}\tau_k$,
and a weighted linear fit of $t_k$ against $\tau_k$ gives
$v_a = \tfrac12 V_{eff}\,m$ (`along_track_velocity`). This relies on `DOPPLER_SIGN` and the
column-time sign, which the two-click chain does not.

---

## 8. Validation

`test/test_mover_relocation.py::TestSignValidation` simulates a constant-velocity
mover on the fixture orbit: $\mathbf P(\tau) = \mathbf P_0 + \mathbf v(\tau - t_0)$, with
$t_0$ the zero-Doppler time of $\mathbf P_0$. The imaged coordinates are the minimum of
$\lVert\mathbf P(\tau) - \mathbf S(\tau)\rVert$ (Newton on its derivative), the road is the
mover's own straight track, and the two-click chain must recover $t_0$ (to 20 µs),
$\mathbf P_0$ (0.3 m), the sign and size of $v_r = -\hat{\mathbf d}\cdot\mathbf v$ (1 %),
$\Delta x$ (1 %), $v_t$ (1.5 %) and the heading (0.5°). It covers 20 cases: left- and
right-looking geometry (the right-looking point is the fixture point mirrored across
the ground track), 3 and 12.5 m/s, and five headings covering approaching and receding
motion. The same test confirms physically that $t_{img} > t_{true}$ exactly when
$v_r > 0$.

This validates the signs of the geometry chain and its conventions in simulation. It
does not validate them against a real mover with known motion; see section 9.

---

## 9. Open points

1. **Real-scene acceptance.** Not yet done: the WWGTZ2 vessel (a map-drift estimate
   of about 2.5 m/s earlier) and the bridge-cars scene (expected displacement of about
   1 km, $|v_r| \approx 12.5$ m/s, ground speed about 20 m/s, displaced cars south of
   the bridge receding, i.e. eastbound) both need a human to click the constraint.
2. **Deck height.** Land vehicles on bridges or elevated roads are geocoded on the
   display surface; an unknown deck height $\Delta h$ shifts the true position by
   about $\Delta h/\tan\theta_{inc}$ in ground range. Users should click the deck's
   direct-return line, not its water-level line.
3. **Metadata interpretation.** $V_g \approx \lVert\mathbf v\rVert\lVert\mathbf P\rVert/\lVert\mathbf S\rVert$
   and $1/f_s$ as the line spacing are inferred, not documented; neither enters the
   two-click chain except through $V_g$ in $\Delta x$ and $\sigma$.
4. **Map-drift sign** (section 7): unvalidated, so the $v_a$ check stays off.
