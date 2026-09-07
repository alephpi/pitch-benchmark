import numpy as np
import torch
from torchfcpe import spawn_bundled_infer_model

from algorithms.base import ContinuousPitchAlgorithm


class FCPEPitchAlgorithm(ContinuousPitchAlgorithm):
    _name = "FCPE"

    def __init__(self, device="cuda", **kwargs):
        super().__init__(**kwargs)
        # Set up device
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 16000
        self.hop_size = 160
        self.model = spawn_bundled_infer_model(device=self.device)

    def _extract_raw_pitch_and_periodicity(self, audio: np.ndarray):
        # Load and preprocess audio
        audio_length = len(audio)
        f0_target_length = (audio_length // self.hop_size) + 1
        wav = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(-1).to(self.device)

        # Perform pitch inference
        f0, uv = self.model.infer(
            wav,
            sr=self.sample_rate,
            decoder_mode='local_argmax',  # Recommended mode
            threshold=0.006,  # Threshold for V/UV decision
            f0_min=80,  # Minimum pitch
            f0_max=880,  # Maximum pitch
            interp_uv=False,  # Interpolate unvoiced frames
            output_interp_target_length=f0_target_length,  # Interpolate to target length
            retur_uv=True,  # Return unvoiced frames
        )
        f0 = f0.squeeze().cpu().numpy()
        uv = uv.squeeze().cpu().numpy()
        # the voiced/unvoiced decision is already binarized inside the model, so here we just invert the uv to get periodicity
        periodicity = 1-uv
        n_frames = len(f0)
        times = np.arange(n_frames) * (self.hop_size / self.sample_rate)
        return times, f0, periodicity
