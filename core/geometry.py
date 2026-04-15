"""
A Python port of the vect2geom function from the ngageoint MATLAB_SAR package.

https://github.com/ngageoint/MATLAB_SAR/blob/master/Geometry/vect2geom.m

Original released under the MIT license:

MIT License

Copyright (c) 2018 National Geospatial-Intelligence Agency

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import math
import warnings
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.metadata import IceyeMetadata


@dataclass
class _FocusPlaneNormal(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self):
        pass

    @abstractmethod
    def calculate_normal(self):
        pass


@dataclass
class WGS84Norm(_FocusPlaneNormal):
    """WGS84 ellipsoid normal for focus plane calculation."""

    a: int = 6378137  # Semi - major(equatorial) axis of WGS_84 model
    b: float = 6356752.314245179  # Semi - minor(polar) axis of WGS_84 model

    def __call__(self, x: float, y: float, z: float) -> NDArray[np.float64]:
        """Return unit normal to WGS84 ellipsoid at (x, y, z) in ECEF."""
        return self.calculate_normal(x=x, y=y, z=z, a=self.a, b=self.b)

    def calculate_normal(
        self, x: float, y: float, z: float, a: int, b: float
    ) -> NDArray[np.float64]:
        """Compute the normal vector to the WGS84 ellipsoid at a given point in ECEF.

        Parameters
        ----------
        x : float
            ECEF x coordinate in meters.
        y : float
            ECEF y coordinate in meters.
        z : float
            ECEF z coordinate in meters.
        a : int
            Semi-major axis in meters.
        b : float
            Semi-minor axis in meters.

        Returns
        -------
        ndarray of float64
            Unit normal vector to the WGS84 ellipsoid at (x, y, z).
        """
        assert not isinstance(x, np.ndarray), (
            "Expected x to be a scalar, but a numpy array was passed."
        )
        assert not isinstance(y, np.ndarray), (
            "Expected y to be a scalar, but a numpy array was passed"
        )
        assert not isinstance(z, np.ndarray), (
            "Expected z to be a scalar, but a numpy array was passed"
        )

        # Calculate normal vector
        x = x / (a**2)
        y = y / (a**2)
        z = z / (b**2)

        norm_v = np.array((x, y, z))

        # Make into unit vector
        mag = np.linalg.norm(norm_v)
        norm_v = norm_v / mag

        return norm_v


@dataclass
class SARViewGeometry:
    """Represent the relationship between a point in the ground and a point of view of a satellite as a dataclass."""

    azimuth: float  # sensor azimuth
    graze: float  # grazing angle
    slope: float  # slope angle (>= graze)
    squint: float  # squint angle
    layover: float  # layover angle
    multipath: float  # multiplath angle
    dca: float  # doppler cone angle
    tilt: float  # slant plane tilt angle
    track: float  # ground track angle in tangent plane at AIM point
    felev: float  # flight elevation
    shadow: float  # shadow angle
    north: float  # north direction angle

    sense: float  # direction of flight, sense < 0 = right, sense > 0 = left

    orbital_node: str | None = None
    look_side: str | None = None

    units: str = "radians"

    def __post_init__(self) -> None:
        """Validate angle attributes are within bounds for the current units."""
        if self.units == "degrees":
            lower_limit = -360.0
            upper_limit = 360.0
        elif self.units == "radians":
            lower_limit = -2.0 * math.pi
            upper_limit = 2.0 * math.pi
        elif not isinstance(self.units, str):
            raise TypeError(
                "Expected attribute units to be a string, got a %s type input instead."
                % type(self.units)
            )
        else:
            raise ValueError(
                'Expected attribute units to be either "degrees" or "radians", got "%s"'
                % self.units
            )

        for item in (
            "azimuth",
            "graze",
            "slope",
            "squint",
            "layover",
            "multipath",
            "dca",
            "tilt",
            "track",
            "shadow",
            "north",
        ):
            item_value = getattr(self, item)
            assert item_value >= lower_limit and item_value <= upper_limit, (
                'Expected "%s" to be between [%.2f,%.2f] %s)'
                % (item, lower_limit, upper_limit, self.units)
            )

    def deg2radian(self) -> None:
        """Convert all angle attributes from degrees to radians in place."""
        assert self.units in ("degrees", "radians")

        if self.units == "radians":
            warnings.warn(
                "The units are already in radians, ignoring the .deg2radian() call."
            )
            return

        self.azimuth = math.radians(self.azimuth)
        self.graze = math.radians(self.graze)
        self.slope = math.radians(self.slope)
        self.squint = math.radians(self.squint)
        self.layover = math.radians(self.layover)
        self.multipath = math.radians(self.multipath)
        self.dca = math.radians(self.dca)
        self.tilt = math.radians(self.tilt)
        self.track = math.radians(self.track)
        self.felev = math.radians(self.felev)
        self.shadow = math.radians(self.shadow)
        self.north = math.radians(self.north)

        self.units = "radians"

    def radian2deg(self) -> None:
        """Convert all angle attributes from radians to degrees in place."""
        assert self.units in ("degrees", "radians")

        if self.units == "degrees":
            warnings.warn(
                "The units are already in degrees, ignoring the.radian2deg() call."
            )
            return

        self.azimuth = math.degrees(self.azimuth)
        self.graze = math.degrees(self.graze)
        self.slope = math.degrees(self.slope)
        self.squint = math.degrees(self.squint)
        self.layover = math.degrees(self.layover)
        self.multipath = math.degrees(self.multipath)
        self.dca = math.degrees(self.dca)
        self.tilt = math.degrees(self.tilt)
        self.track = math.degrees(self.track)
        self.felev = math.degrees(self.felev)
        self.shadow = math.degrees(self.shadow)
        self.north = math.degrees(self.north)

        self.units = "degrees"

    def deduce_look_side(self) -> str:
        """Return look side from sense: "left", "right", or "unknown"."""
        # Original had self.right as an integer flag
        if self.sense < 0:
            geometry = "right"
        elif self.sense > 0:
            geometry = "left"
        else:
            geometry = "unknown"
        return geometry

    def deduce_orbital_node(self, VEL: NDArray[np.float64]) -> str:
        """Return orbital node from velocity: "ascending", "descending", or "unknown"."""
        ## TODO validate
        if VEL[-1] > 0:
            orbital_node = "ascending"
        elif VEL[-1] < 0:
            orbital_node = "descending"
        else:
            orbital_node = "unknown"
        return orbital_node


def geodetic_to_ecef(
    lat: float,
    lon: float,
    h: float = 0.0,
    a: float = 6378137.0,
    b: float = 6356752.314245,
) -> tuple[float, float, float]:
    """Convert geodetic coordinates to ECEF (x, y, z) in meters.

    Parameters
    ----------
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees.
    h : float, optional
        Height above ellipsoid in meters. Default is 0.
    a : float, optional
        Semi-major axis in meters. Default is WGS84.
    b : float, optional
        Semi-minor axis in meters. Default is WGS84.

    Returns
    -------
    tuple of float
        (x, y, z) in meters (ECEF).
    """
    # Convert latitude and longitude from degrees to radians
    lat = np.radians(lat)
    lon = np.radians(lon)

    # Eccentricity squared
    e2 = (a * a - b * b) / (a * a)

    # Radius of curvature in prime vertical
    N = a / (1 - e2 * (np.sin(lat) ** 2)) ** 0.5

    # Convert
    x = (N + h) * np.cos(lat) * np.cos(lon)
    y = (N + h) * np.cos(lat) * np.sin(lon)
    z = ((b * b / (a * a)) * N + h) * np.sin(lat)

    return x, y, z


def ecef_to_geodetic(
    x: float,
    y: float,
    z: float,
    a: float = 6378137.0,
    b: float = 6356752.314245,
) -> tuple[float, float]:
    """Convert ECEF coordinates to geodetic (latitude, longitude) in radians.

    Parameters
    ----------
    x : float
        ECEF x coordinate in meters.
    y : float
        ECEF y coordinate in meters.
    z : float
        ECEF z coordinate in meters.
    a : float, optional
        Semi-major axis in meters. Default is WGS84.
    b : float, optional
        Semi-minor axis in meters. Default is WGS84.

    Returns
    -------
    tuple of float
        (latitude, longitude) in radians.
    """
    # Longitude calculation is straightforward
    lon = np.arctan2(y, x)

    # For latitude, use iterative method
    p = np.sqrt(x * x + y * y)
    lat = np.arctan2(z, p * (1 - (a * a - b * b) / (a * a)))

    for _ in range(5):  # Usually converges in 2-3 iterations
        N = a / np.sqrt(1 - (a * a - b * b) / (a * a) * np.sin(lat) ** 2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - N * (a * a - b * b) / (a * a + h) / N))

    return lat, lon


def sar_vectors_to_geometry(
    AIM: NDArray[np.float64],
    P: NDArray[np.float64],
    VEL: NDArray[np.float64],
    N: _FocusPlaneNormal = WGS84Norm(),
) -> SARViewGeometry:
    """Compute SAR collection geometry from aim point, platform position and velocity.

    Parameters
    ----------
    AIM : ndarray of float64
        Ground reference point in ECEF (3 elements), meters.
    P : ndarray of float64
        Platform position (center aperture) in ECEF (3 elements), meters.
    VEL : ndarray of float64
        Platform velocity in ECEF (3 elements), m/s.
    N : _FocusPlaneNormal, optional
        Focus plane normal. Defaults to WGS84 tangent plane unit normal.

    Returns
    -------
    SARViewGeometry
        SAR collection geometry dataclass with angles in radians.
    """

    def proj(
        v: NDArray[np.float64], normal: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Project vector v onto the plane defined by normal."""
        n = normal / np.linalg.norm(normal)
        p = v - np.dot(v, n) * n
        return p

    _sanity_check_inputs(AIM, P, VEL, N)

    N = N(x=AIM[0], y=AIM[1], z=AIM[2])  # Norm class -> norm vector

    # range from aim point to platform
    R = P - AIM
    R_ = R / np.linalg.norm(R)
    P_ = P / np.linalg.norm(P)

    # aim point unit vectors
    KDP_ = N / np.linalg.norm(N)

    JDP_ = proj(R_, KDP_)
    JDP_ = JDP_ / np.linalg.norm(JDP_)

    # sensor azimuth
    ProjN_ = proj(np.array([0, 0, 1]), KDP_)
    ProjN_ = ProjN_ / np.linalg.norm(ProjN_)

    azimuth = np.arctan2(np.dot(np.cross(JDP_, ProjN_), KDP_), np.dot(ProjN_, JDP_))

    if azimuth < 0:
        azimuth = azimuth + 2 * math.pi

    # trajectory unit vector
    TRAJ_ = VEL / np.linalg.norm(VEL)

    # slant plane normal with ambiguous sense of up
    SLANT = np.cross(R_, TRAJ_)
    SLANT_ = SLANT / np.linalg.norm(SLANT)

    # direction of flight: sense < 0 - right, sense > 0 - left
    sense = np.dot(SLANT_, KDP_)
    sense = np.sign(sense)

    # corrected sense of up for slant plane normal
    SLANT_ = sense * SLANT_

    # slope angle( >= graze )
    slope = np.arccos(np.dot(SLANT_, KDP_))

    # grazing angle
    graze = np.arcsin(np.dot(R_, KDP_))

    # squint angle
    Vproj_ = proj(VEL, P)
    Vproj_ = Vproj_ / np.linalg.norm(Vproj_)
    Rproj_ = proj(-R, P)
    Rproj_ = Rproj_ / np.linalg.norm(Rproj_)
    squint = np.arctan2(np.dot(np.cross(Vproj_, Rproj_), P_), np.dot(Rproj_, Vproj_))

    # layover angle
    TMP_ = np.cross(SLANT_, KDP_)
    TMP_ = TMP_ / np.linalg.norm(TMP_)
    layover = np.arcsin(np.dot(JDP_, TMP_))

    # Doppler cone angle
    dca = -sense * np.arccos(np.dot(-R_, TRAJ_))

    # ground track angle in tangent plane at AIM point
    TRACK_ = proj(TRAJ_, KDP_)
    TRACK_ = TRACK_ / np.linalg.norm(TRACK_)
    track = -sense * np.arccos(np.dot(-1.0 * JDP_, TRACK_))

    # flight elevation
    felev = np.arcsin(np.dot(KDP_, TRAJ_))

    # slant plane tilt angle
    tilt = -np.arccos(np.cos(slope) / np.cos(graze)) * np.sign(layover)

    # multipath angle
    multipath = -np.arctan(np.tan(tilt) * np.sin(graze))

    # shadow angle calculation
    S = KDP_ - (R_ / np.dot(R_, KDP_))
    z = np.cross(JDP_, np.cross(JDP_, KDP_))  # Output plane normal
    z = z / np.linalg.norm(z)
    S_prime = S - (np.dot(S, z) / np.dot(z, z)) * z

    # Shadow vector
    S = KDP_ - R_ / np.dot(R_, KDP_)

    # Cross range vector
    c_hat = np.cross(JDP_, KDP_)
    c_hat = c_hat / np.linalg.norm(c_hat)
    z = np.cross(JDP_, c_hat)  # Output plane normal

    # Project shadow vector onto output plane
    S_prime = S - (np.dot(S, z) / np.dot(z, z)) * z

    # Calculate shadow angle using atan2
    shadow = np.arctan2(np.dot(c_hat, S_prime), np.dot(JDP_, S_prime))
    if shadow < 0:
        shadow += 2 * np.pi

    # North direction calculation
    lat, lon = ecef_to_geodetic(AIM[0], AIM[1], AIM[2])

    # North vector following spec
    N_vec = np.array(
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)]
    )

    # Cross range vector (already calculated for shadow)
    c_hat = np.cross(JDP_, KDP_)
    c_hat = c_hat / np.linalg.norm(c_hat)
    z = np.cross(JDP_, c_hat)

    # Project onto output plane
    N_prime = N_vec - (np.dot(N_vec, z) / np.dot(z, z)) * z

    # Calculate north direction angle
    north = np.arctan2(np.dot(c_hat, N_prime), np.dot(JDP_, N_prime))
    if north < 0:
        north += 2 * np.pi

    # results
    geometry = SARViewGeometry(
        azimuth=azimuth,
        graze=graze,
        slope=slope,
        squint=squint,
        layover=layover,
        multipath=multipath,
        dca=dca,
        tilt=tilt,
        track=track,
        felev=felev,
        shadow=shadow,
        north=north,
        sense=sense,
    )

    geometry.orbital_node = geometry.deduce_orbital_node(VEL)
    geometry.look_side = geometry.deduce_look_side()

    return geometry


def _sanity_check_inputs(
    aim: NDArray[np.float64],
    pos: NDArray[np.float64],
    vel: NDArray[np.float64],
    normal: _FocusPlaneNormal,
) -> None:
    assert isinstance(normal, _FocusPlaneNormal), (
        "Expected a _FocusPlaneNormal child class, got %s instead" % (type(normal))
    )

    for vector in (aim, pos, vel):
        assert isinstance(vector, np.ndarray), (
            "Expected a numpy array, got %s instead" % (type(vector))
        )
        assert vector.size == 3, (
            "Expected a 3-element vector, got %d elements instead" % (aim.size)
        )
        assert vector.ndim == 1, "Expected 1D array, got %dD array" % (vector.ndim)

    aim_norm = np.linalg.norm(aim)
    pos_norm = np.linalg.norm(pos)
    vel_norm = np.linalg.norm(vel)

    assert aim_norm > 1000, (
        "Expected the input AIM to be in meters, a very low vector norm of %.1f was detected. "
        % aim_norm
    )
    assert pos_norm > 1000, (
        "Expected the input P to be in meters, a very low vector norm of %.1f was detected."
        % pos_norm
    )
    assert vel_norm > 100, (
        "Expected the input VEL to be in meters, a very low vector norm of %.1f was detected."
        % vel_norm
    )


def get_geometry_from_metadata(metadata: IceyeMetadata) -> SARViewGeometry:
    """Build SARViewGeometry from ICEYE metadata.

    Parameters
    ----------
    metadata : IceyeMetadata
        ICEYE layer metadata with proj_centroid and orbit state.

    Returns
    -------
    SARViewGeometry
        SAR collection geometry for the scene.
    """
    centroid_ecef = np.array(
        geodetic_to_ecef(
            metadata.proj_centroid["lat"],
            metadata.proj_centroid["lon"],
            metadata.iceye_average_scene_height,
        )
    )
    return sar_vectors_to_geometry(
        AIM=centroid_ecef,
        P=metadata.center_aperture_position,
        VEL=metadata.center_aperture_velocity,
    )
