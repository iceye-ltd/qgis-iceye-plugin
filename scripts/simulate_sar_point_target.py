"""Simulate a SAR point target and image it with the Range-Doppler Algorithm.

Geometry: broadside, side-looking stripmap. A single ideal point scatterer
sits at slant range ``R0`` and zero azimuth offset. Airborne X-band-ish
parameters are chosen so the raw echo, range-compressed data, and focused
image all fit comfortably in memory and run in ~1 s.

Pipeline:
  1. Generate baseband raw echo  s(eta, tau) for one point target
     (rectangular chirp, sinc^2 antenna pattern in azimuth, hyperbolic
     range history R(eta) = sqrt(R0^2 + (v*eta)^2)).
  2. Range compress with the analytic LFM matched filter (POSP).
  3. FFT along azimuth -> range-Doppler domain.
  4. Range cell migration correction (RCMC): linear range-phase per f_eta.
  5. Azimuth compression: multiply by exp(+j*pi*f_eta^2 / Ka).
  6. IFFT along azimuth -> focused SLC.
  7. Cut through the peak, measure 3 dB resolution and PSLR.

Outputs a single PNG summarising each stage next to the script.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent

C = 299_792_458.0


@dataclass
class SARParams:
    fc: float = 9.65e9         # carrier frequency [Hz] (X-band, ICEYE-ish)
    v: float = 7500.0          # platform velocity [m/s] (LEO)
    R0: float = 700_000.0      # target slant range [m]
    B: float = 100e6           # range bandwidth [Hz]
    Tp: float = 2.5e-6         # transmitted pulse duration [s]
    Fs: float = 240e6          # range sampling rate [Hz]
    Nrg: int = 1024            # samples per pulse
    PRF: float = 100000.0        # pulse repetition frequency [Hz]
    Ta: float = 5.0            # observation (dwell) time [s]
    La: float = 6.0            # real antenna azimuth length [m] -> rho_a ~ La/2

    @property
    def lam(self) -> float:
        return C / self.fc

    @property
    def Kr(self) -> float:
        return self.B / self.Tp

    @property
    def Naz(self) -> int:
        return int(round(self.Ta * self.PRF))

    @property
    def Ka(self) -> float:
        return 2.0 * self.v * self.v / (self.lam * self.R0)


def generate_raw(p: SARParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (raw, eta, tau).  raw has shape (Naz, Nrg), complex64."""
    dtau = 1.0 / p.Fs
    tau_center = 2.0 * p.R0 / C
    tau = tau_center + (np.arange(p.Nrg) - p.Nrg // 2) * dtau

    deta = 1.0 / p.PRF
    eta = (np.arange(p.Naz) - p.Naz // 2) * deta
    vx = 20.0
    vy = 10.0
    x0 = 0.0
    y0 = 0.0
    xt = x0 + vx * eta
    yt = y0 + vy * eta
    incident_angle= 20*np.pi/180.0
    H = p.R0 * np.cos(incident_angle)
    y_sc = p.R0 * np.sin(incident_angle)
    R_eta = np.sqrt(H**2 + (p.v * eta - xt) ** 2 + (y_sc - yt) ** 2) 
    # R_eta = np.sqrt(p.R0**2 + (p.v * eta) ** 2)     # (Naz,)
    theta = np.arctan2(p.v * eta, p.R0)             # squint at each pulse
    theta_bw = p.lam / p.La
    w_ant = np.sinc(theta / theta_bw) ** 2          # 2-way = sinc^2

    dt = tau[None, :] - (2.0 * R_eta[:, None] / C)  # (Naz, Nrg)
    range_env = (np.abs(dt) <= p.Tp / 2).astype(np.float32)
    range_phase = np.pi * p.Kr * dt * dt
    az_phase = -4.0 * np.pi * R_eta[:, None] / p.lam

    raw = (w_ant[:, None] * range_env
           * np.exp(1j * (range_phase + az_phase))).astype(np.complex64)
    return raw, eta, tau


def range_compress(raw: np.ndarray, p: SARParams) -> np.ndarray:
    """Analytic LFM matched filter in the range-frequency domain."""
    Nrg = raw.shape[1]
    f_r = np.fft.fftfreq(Nrg, d=1.0 / p.Fs)
    Href = (np.abs(f_r) <= p.B / 2) * np.exp(1j * np.pi * f_r * f_r / p.Kr)
    RC_f = np.fft.fft(raw, axis=1) * Href[None, :].astype(np.complex64)
    return np.fft.ifft(RC_f, axis=1).astype(np.complex64)


def apply_rcmc(rc: np.ndarray, p: SARParams) -> np.ndarray:
    """RCMC only. Returns the range-compressed data with RCM removed, back in the
    azimuth-time / range-time domain so it is directly comparable to `rc`."""
    Naz, Nrg = rc.shape
    Rc = np.fft.fft(rc, axis=0)                     # -> range-Doppler
    f_eta = np.fft.fftfreq(Naz, d=1.0 / p.PRF)      # (Naz,)
    f_r = np.fft.fftfreq(Nrg, d=1.0 / p.Fs)         # (Nrg,)

    dR = (p.lam ** 2) * (f_eta ** 2) * p.R0 / (8.0 * p.v * p.v)
    Rc_f = np.fft.fft(Rc, axis=1)
    shift_phase = np.exp(1j * 2.0 * np.pi
                         * f_r[None, :] * (2.0 * dR[:, None] / C))
    Rc_f = Rc_f * shift_phase
    Rc = np.fft.ifft(Rc_f, axis=1)                  # back to range-time
    return np.fft.ifft(Rc, axis=0).astype(np.complex64)  # back to azimuth-time


_UNUSED_KEPT_FOR_LATER = '''
def generate_raw(p: SARParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (raw, eta, tau).  raw has shape (Naz, Nrg), complex64."""
    dtau = 1.0 / p.Fs
    tau_center = 2.0 * p.R0 / C
    tau = tau_center + (np.arange(p.Nrg) - p.Nrg // 2) * dtau

    deta = 1.0 / p.PRF
    eta = (np.arange(p.Naz) - p.Naz // 2) * deta
    incident_angle = 20 * np.pi / 180.0
    H = p.R0 * np.cos(incident_angle)
    y_sc = p.R0 * np.sin(incident_angle)
    R_eta = np.sqrt(H ** 2 + (p.v * eta) ** 2 + y_sc ** 2)

    theta = np.arctan2(p.v * eta, p.R0)
    theta_bw = p.lam / p.La
    w_ant = (np.sinc(theta / theta_bw) ** 2).astype(np.float32)

    dt = tau[None, :] - (2.0 * R_eta[:, None] / C)
    range_env = (np.abs(dt) <= p.Tp / 2).astype(np.float32)
    range_phase = np.pi * p.Kr * dt * dt
    az_phase = -4.0 * np.pi * R_eta[:, None] / p.lam
    raw = (w_ant[:, None] * range_env
           * np.exp(1j * (range_phase + az_phase))).astype(np.complex64)
    return raw, eta, tau


def range_compress(raw: np.ndarray, p: SARParams) -> np.ndarray:
    """Analytic LFM matched filter in the range-frequency domain."""
    Nrg = raw.shape[1]
    f_r = np.fft.fftfreq(Nrg, d=1.0 / p.Fs)
    Href = (np.abs(f_r) <= p.B / 2) * np.exp(1j * np.pi * f_r * f_r / p.Kr)
    RC_f = np.fft.fft(raw, axis=1) * Href[None, :].astype(np.complex64)
    return np.fft.ifft(RC_f, axis=1).astype(np.complex64)


def azimuth_compress(rcmc_out: np.ndarray, p: SARParams) -> np.ndarray:
    """Azimuth matched filter: azimuth phase is exp(-j*pi*Ka*eta^2), POSP spectrum is
    exp(+j*pi*f_eta^2/Ka), so the matched filter is its conjugate."""
    Naz = rcmc_out.shape[0]
    f_eta = np.fft.fftfreq(Naz, d=1.0 / p.PRF)
    S = np.fft.fft(rcmc_out, axis=0)
    Haz = np.exp(-1j * np.pi * f_eta * f_eta / p.Ka)
    S = S * Haz[:, None]
    return np.fft.ifft(S, axis=0).astype(np.complex64)


def _upsample(cut: np.ndarray, factor: int = 16) -> np.ndarray:
    """Ideal sinc interpolation via zero-padded FFT."""
    N = cut.size
    F = np.fft.fftshift(np.fft.fft(cut))
    pad = np.zeros(N * factor, dtype=complex)
    pad[N * factor // 2 - N // 2: N * factor // 2 - N // 2 + N] = F
    return np.abs(np.fft.ifft(np.fft.ifftshift(pad))) * factor


def measure_response(img: np.ndarray, p: SARParams) -> dict:
    """Cut through the peak; return azimuth/range 3-dB widths and PSLR."""
    mag = np.abs(img)
    ky, kx = np.unravel_index(np.argmax(mag), mag.shape)
    az_cut = mag[:, kx]
    rg_cut = mag[ky, :]

    OS = 16

    def _res_3db(cut: np.ndarray, pix: float) -> float:
        up = _upsample(cut, OS)
        peak = up.max()
        idx = np.where(up >= peak / np.sqrt(2.0))[0]
        if idx.size < 2:
            return float(pix / OS)
        return float((idx[-1] - idx[0]) * (pix / OS))

    def _pslr_db(cut: np.ndarray) -> float:
        up = _upsample(cut, OS)
        peak_idx = int(np.argmax(up))
        d = np.diff(np.sign(np.diff(up)))
        peaks = np.where(d < 0)[0] + 1
        side = peaks[peaks != peak_idx]
        if side.size == 0:
            return float("nan")
        return float(20.0 * np.log10(up[side].max() / up[peak_idx]))

    return {
        "peak_row": ky,
        "peak_col": kx,
        "az_cut": az_cut,
        "rg_cut": rg_cut,
        "res_az_m": _res_3db(az_cut, p.v / p.PRF),
        "res_rg_m": _res_3db(rg_cut, C / (2.0 * p.Fs)),
        "pslr_az_db": _pslr_db(az_cut),
        "pslr_rg_db": _pslr_db(rg_cut),
    }


def _db(x: np.ndarray) -> np.ndarray:
    m = np.abs(x)
    return 20.0 * np.log10(m / m.max() + 1e-12)
'''


def plot_all(rc, rc_rcmc, p: SARParams) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    extent = [0, p.Nrg, 0, p.Naz]

    rc_mag = np.abs(rc[::16, ::4])
    rcmc_mag = np.abs(rc_rcmc[::16, ::4])
    vmin0, vmax0 = rc_mag.mean() - 2 * rc_mag.std(), rc_mag.mean() + 2 * rc_mag.std()
    vmin1, vmax1 = rcmc_mag.mean() - 2 * rcmc_mag.std(), rcmc_mag.mean() + 2 * rcmc_mag.std()

    axes[0].imshow(
        rc_mag,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=vmin0,
        vmax=vmax0,
    )
    axes[0].set_title("|range compressed|  (decimated ::16, ::4)")
    axes[0].set_xlabel("range sample [0 .. Nrg]")
    axes[0].set_ylabel("azimuth sample [0 .. Naz]")

    axes[1].imshow(
        rcmc_mag,
        aspect="auto",
        origin="lower",
        extent=extent,
        vmin=vmin1,
        vmax=vmax1,
        cmap="viridis",
    )
    axes[1].set_title("|after RCMC|  (decimated ::16, ::4)")
    axes[1].set_xlabel("range sample [0 .. Nrg]")

    fig.tight_layout()
    plt.show()


def main() -> None:
    p = SARParams()
    print(f"lambda    = {p.lam*1e3:.2f} mm")
    print(f"Kr        = {p.Kr:.3e} Hz/s")
    print(f"Ka        = {p.Ka:.3e} Hz/s  (at R0={p.R0} m)")
    print(f"rho_r     = {C/(2*p.B):.2f} m  (theoretical)")
    print(f"rho_a     = {p.La/2:.2f} m  (real-antenna limit)")
    print(f"Nchirp    = {int(round(p.Tp*p.Fs))} samples")
    print(f"(Naz,Nrg) = ({p.Naz}, {p.Nrg})")

    raw, eta, tau = generate_raw(p)
    rc = range_compress(raw, p)
    del raw
    rc_rcmc = apply_rcmc(rc, p)

    plot_all(rc, rc_rcmc, p)


if __name__ == "__main__":
    main()
