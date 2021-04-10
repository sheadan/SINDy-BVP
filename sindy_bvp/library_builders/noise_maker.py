"""Simple class for applying noise and gaussian filtering to signal data."""

# Third-Party Imports
import numpy as np
from scipy.ndimage import gaussian_filter


class NoiseMaker:
    """Apply Gaussian white noise, average measurements, and smooth data."""

    def __init__(self, noise_magnitude: float,
                 measurements_to_average: int = 1,
                 gaussian_filter_sigma: float = None):
        """Initialize attributes for noise and data filtering.

        Keyword arguments:
        noise_magnitude -- the magnitude of gaussian white noise to apply
        to signal, based on this float (fraction) of the standard dev.
        measurements_to_average -- consider averaging the data over numerous
        measurements (if like a real-life scenario). not implemented in S-BVP.
        gaussian_filter_sigma -- sigma param for SciPy gaussian_filter method.
        """
        self.noise_mag = noise_magnitude
        self.gaussian_sigma = gaussian_filter_sigma
        self.measurements_to_average = measurements_to_average

    def apply_noise(self, signal: np.ndarray):
        """Apply noise and filtering to a signal.

        Keyword arguments:
        signal -- the input signal, typically a clean signal.

        Returns:
        signal -- the signal after noise and filters were applied.
        """
        noise = 0
        for i in range(self.measurements_to_average):
            noise += self.get_noise(signal)
        noise = noise/self.measurements_to_average
        signal = signal + noise
        if self.gaussian_sigma is not None:
            signal = gaussian_filter(input=signal,
                                     sigma=self.gaussian_sigma,
                                     mode='nearest')
        return signal

    def get_noise(self, signal):
        SNR = 1/(self.noise_mag)
        #std_z = np.std(signal)
        noise = np.random.randn(*signal.shape)

        z_norm = np.linalg.norm(signal)
        SNR_correct_norm = np.sqrt((z_norm**2)/(10**(SNR/10)))

        noise_norm = np.linalg.norm(noise)
        noise = (SNR_correct_norm/noise_norm)*noise

        return noise

    def get_noise_old(self, signal):
        noise = self.noise_mag*np.std(signal)
        noise *= np.random.randn(len(signal))
        return noise
