import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from PyAstronomy import pyasl
import multiprocessing as mp


class BinaryAnalysis:
    def __init__(self, mist_filename):
        data = np.loadtxt(mist_filename, skiprows=2)

        log_Teff = data[:, 4]
        log_g = data[:, 5]
        log_L = data[:, 6]

        # Deliberate: no extrapolation
        self.g_interp = interp1d(log_Teff, log_g, kind="linear")
        self.L_interp = interp1d(log_Teff, log_L, kind="linear")

        self.binary_results = {}

    def _calculate_radius(self, temperature, log_luminosity):
        sigma = 5.670374419e-8
        L_sun = 3.828e26

        L_watts = 10 ** log_luminosity * L_sun
        R_m = np.sqrt(L_watts / (4 * np.pi * sigma * temperature ** 4))
        return R_m * 100

    def _interpolate_parameters(self, temperature):
        logT = math.log10(temperature)
        return self.g_interp(logT), self.L_interp(logT)

    def generate_random_binaries(self, count=10):
        binaries = [
            (x := np.random.randint(3001, 6199),
             np.random.randint(3000, x))
            for _ in range(count)
        ]

        for i, (T1, T2) in enumerate(binaries, start=1):
            g1, L1 = self._interpolate_parameters(T1)
            g2, L2 = self._interpolate_parameters(T2)

            self.binary_results[f"pair_{i}"] = {
                "primary": {
                    "Teff": T1,
                    "logT": math.log10(T1),
                    "gravity": g1,
                    "luminosity": L1,
                    "radius": self._calculate_radius(T1, L1),
                },
                "secondary": {
                    "Teff": T2,
                    "logT": math.log10(T2),
                    "gravity": g2,
                    "luminosity": L2,
                    "radius": self._calculate_radius(T2, L2),
                }
            }

    def get_binary_results(self):
        return self.binary_results


class Volga(BinaryAnalysis):
    def __init__(self, mist_data, base_dir):
        super().__init__(mist_data)
        self.base_dir = base_dir
        self.generate_random_binaries()

    @staticmethod
    def process_temp_gravity(temp, gravity):
        temp_str = f"{int(round(temp, -2) // 100):03d}"
        gravity_str = f"{round(gravity * 2) / 2:.1f}"
        return temp_str, gravity_str

    def load_flux_file(self, temp, gravity):
        temp_str, gravity_str = self.process_temp_gravity(temp, gravity)
        filename = f"lte{temp_str}-{gravity_str}-0.0a+0.0.BT-Settl.spec.7.txt"

        filepath = os.path.join(
            self.base_dir,
            "BT-Settl_M-0.0a+0.0",
            filename
        )

        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)

        return np.loadtxt(filepath)

    def normalize_and_scale_flux(self, radius, flux_array):
        surface_area = 4 * np.pi * radius ** 2
        return flux_array[:, 1] * surface_area

    def get_high_res_binary_spectrum_data(self, binary_index=0):
        binary = list(self.binary_results.values())[binary_index]
        p, s = binary["primary"], binary["secondary"]

        flux1 = self.load_flux_file(p["Teff"], p["gravity"])
        flux2 = self.load_flux_file(s["Teff"], s["gravity"])

        wavelength1 = flux1[:, 0]
        wavelength2 = flux2[:, 0]

        if not np.allclose(wavelength1, wavelength2):
            interp_func = interp1d(wavelength2, flux2[:, 1], kind="linear")
            flux2_interp = interp_func(wavelength1)
        else:
            flux2_interp = flux2[:, 1]

        norm_flux1 = self.normalize_and_scale_flux(p["radius"], flux1)
        norm_flux2 = self.normalize_and_scale_flux(s["radius"], flux2_interp)

        return wavelength, norm_flux1, norm_flux2, norm_flux1 + norm_flux2


class VolgaInterpolator(Volga):
    def __init__(self, mist_data, base_dir):
        super().__init__(mist_data, base_dir)

        self.teff_grid = np.arange(3000, 6300, 100)
        self.logg_grid = np.arange(4.0, 6.0, 0.5)
        self.wavelength_grid = np.linspace(6400, 8500, 10000)
        self.bt_settl_dir = os.path.join(self.base_dir, 'BT-Settl_M-0.0a+0.0')

    def filename_func(self, teff, logg):
        temp_str = f"{int(teff / 100):03d}"
        gravity_str = f"{round(logg * 2) / 2:.1f}"
        return os.path.join(
            self.bt_settl_dir,
            f"lte{temp_str}-{gravity_str}-0.0a+0.0.BT-Settl.spec.7.txt"
        )

    def find_surrounding_grid_points(self, teff, logg):
        t0 = max([t for t in self.teff_grid if t <= teff], default=self.teff_grid[0])
        t1 = min([t for t in self.teff_grid if t > teff], default=self.teff_grid[-1])

        g0 = max([g for g in self.logg_grid if g <= logg], default=self.logg_grid[0])
        g1 = min([g for g in self.logg_grid if g > logg], default=self.logg_grid[-1])

        return [(t, g) for t in [t0, t1] for g in [g0, g1]]

    def load_and_interp_flux(self, teff, logg):
        filename = self.filename_func(teff, logg)
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)

        data = np.loadtxt(filename)
        interp_flux = interp1d(
            data[:, 0], data[:, 1],
            bounds_error=False,
            fill_value=0
        )
        return interp_flux(self.wavelength_grid)

    def bilinear_interpolation(self, x, y, points):
        points = sorted(points, key=lambda p: (p[0], p[1]))
        (x1, y1, q11), (_, y2, q12), (x2, _, q21), (_, _, q22) = points

        return (
            q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
        ) / ((x2 - x1) * (y2 - y1))

    def broaden(self, wavelength, flux, res):
        broadened = pyasl.instrBroadGaussFast(wavelength, flux, res, maxsig=5)
        return wavelength, broadened

    def interpolate_and_broaden_spectrum(self, teff, logg, resolution=30000):
        points = [
            (t, g, self.load_and_interp_flux(t, g))
            for t, g in self.find_surrounding_grid_points(teff, logg)
        ]

        interp_flux = self.bilinear_interpolation(teff, logg, points)
        _, broadened_flux = self.broaden(self.wavelength_grid, interp_flux, resolution)

        return self.wavelength_grid, broadened_flux

    def normalize(self, flux_primary, flux_secondary):
        norm = np.median(flux_primary)
        return flux_primary / norm, flux_secondary / norm

    def pixelize(self, wavelength, flux, koi_path="data/Koi1422_HET.txt"):
        if not os.path.isabs(koi_path):
            koi_path = os.path.join(self.base_dir, koi_path)

        data = np.loadtxt(koi_path)
        koi_wave = data[:, 0] * 1e4     # koi is in microns, so we need to ocnvert it to Angstroms
        koi_err = data[:, 2]

        interp_func = interp1d(wavelength, flux, bounds_error=False, fill_value=np.nan)
        interp_flux = interp_func(koi_wave)

        #edit 2/2/26 202 - 215
        snr = 25
        noise_std = 1.0 / snr  # 0.04

        # noise = np.random.normal(
        #     loc=0.0,
        #     scale=noise_std,
        #     size=interp_flux.shape
        # )

        # return koi_wave, interp_flux, noise, koi_err

        noise = np.random.normal(
            loc=0.0,
            scale=noise_std,
            size=interp_flux.shape
        )

        noisy_flux = interp_flux #+ noise #generate CLEAN data only

        return koi_wave, interp_flux, noisy_flux, koi_err


        # noise = np.random.uniform(-1, 1, interp_flux.shape) * (1 / 50)
        # noisy_flux = interp_flux + noise
        # return koi_wave, interp_flux, noisy_flux, koi_err

