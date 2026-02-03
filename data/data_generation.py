import multiprocessing as mp
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import logging
import sys

# Import from the merged module created previously
from data_synthesization import BinaryAnalysis, VolgaInterpolator


# =========================
# Configuration
# =========================
mist_data_file = "Mist.txt"
base_dir = "/data/niaycarr/"
resolution = 30000


# =========================
# Logging setup
# =========================
log_filename = "process_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PID %(process)d] %(message)s",
    handlers=[
        logging.FileHandler(log_filename, mode="w"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# =========================
# Plotting
# =========================
def plot_spectrum_stages(
    pair_key,
    wv_p, flux_p_broad, flux_p_norm,
    wv_s, flux_s_broad, flux_s_norm,
    wv_pix, flux_p_noisy, flux_s_noisy,
    output_dir="spectrum_plots"
):
    os.makedirs(output_dir, exist_ok=True)

    flux_comb_broad = flux_p_broad + flux_s_broad
    flux_comb_norm  = flux_p_norm  + flux_s_norm
    flux_comb_noisy = flux_p_noisy + flux_s_noisy

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    plt.subplots_adjust(hspace=0.1)

    axs[0].plot(wv_p, flux_p_broad, alpha=0.7)
    axs[0].plot(wv_s, flux_s_broad, alpha=0.7)
    axs[0].plot(wv_p, flux_comb_broad, linewidth=2)
    axs[0].set_title("Broadened")

    axs[1].plot(wv_p, flux_p_norm, alpha=0.7)
    axs[1].plot(wv_s, flux_s_norm, alpha=0.7)
    axs[1].plot(wv_p, flux_comb_norm, linewidth=2)
    axs[1].set_title("Normalized")

    axs[2].plot(wv_pix, flux_p_noisy, alpha=0.7)
    axs[2].plot(wv_pix, flux_s_noisy, alpha=0.7)
    axs[2].plot(wv_pix, flux_comb_noisy, linewidth=2)
    axs[2].set_title("Pixelized (Noisy)")
    axs[2].set_xlabel("Wavelength [Å]")

    filepath = os.path.join(output_dir, f"{pair_key}_noisy_stages.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

# =========================
# Construction function
# =========================
def process_binary(pair_key, pair_data, mist_file, base_dir, resolution=30000):
    try:
        volga = VolgaInterpolator(mist_file, base_dir)

        p = pair_data["primary"]
        s = pair_data["secondary"]

        p_radius = p["radius"]
        s_radius = s["radius"]

        # Step 1: Interpolate + broaden
        wv_p, flux_p_broad = volga.interpolate_and_broaden_spectrum(
            p["Teff"], p["gravity"], resolution
        )
        wv_s, flux_s_broad = volga.interpolate_and_broaden_spectrum(
            s["Teff"], s["gravity"], resolution
        )

        # Step 2: Normalize
        flux_p_norm, flux_s_norm = volga.normalize(
            flux_p_broad, flux_s_broad
        )

        # Step 3: Pixelize
        wv_p_pix, _, flux_p_noisy, _ = volga.pixelize(
            wv_p, flux_p_norm
        )
        wv_s_pix, _, flux_s_noisy, _ = volga.pixelize(
            wv_s, flux_s_norm
        )

        if not np.allclose(wv_p_pix, wv_s_pix, atol=1e-6):
            raise ValueError("Pixel wavelength grids do not match")

        # Step 4: Combine
        flux_comb_noisy = flux_p_noisy + flux_s_noisy
        flux_comb_noisy /= np.median(flux_comb_noisy)

        # Step 5: Output directories
        noisy_dir = os.path.join(base_dir, "noisy_spectrum")
        os.makedirs(noisy_dir, exist_ok=True)

        # added rounding to the proper decimal not just "cutting" off
        filename_base = (
            f"spec_{int(p['Teff'])}_{int(s['Teff'])}_"
            f"{p['gravity']:.2f}_{s['gravity']:.2f}_"
            f"{p['radius']:.3e}_{s['radius']:.3e}"
        )


        np.savetxt(
            os.path.join(noisy_dir, f"{filename_base}.txt"),
            np.column_stack((wv_p_pix, flux_p_noisy, flux_s_noisy, flux_comb_noisy)),
            header="Wavelength  Flux_Primary  Flux_Secondary  Flux_Combined"
        )

        logger.info(f"[{pair_key}] completed")

    except Exception as e:
        logger.error(f"[{pair_key}] failed: {e}", exc_info=True)


# =========================
# Main
# =========================
if __name__ == "__main__":
    start_time = time.time()
    logger.info("Starting binary processing pipeline")

    analysis = BinaryAnalysis(mist_data_file)

    target_count = 20000
    analysis.generate_random_binaries(target_count)
    binary_results = analysis.get_binary_results()

    args = [
        (pair_key, pair_data, mist_data_file, base_dir, resolution)
        for pair_key, pair_data in binary_results.items()
    ]

    with Pool(processes=os.cpu_count()) as pool:
        pool.starmap(process_binary, args)

    elapsed = time.time() - start_time
    logger.info(f"Processing complete in {elapsed:.2f} s")

