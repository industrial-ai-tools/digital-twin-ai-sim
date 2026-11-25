import time
import numpy as np
import torch

from model import FNO2d
from data import gaussian_random_field, solve_darcy_2d


class FNOInferenceEngine:
    def __init__(self, checkpoint_path, modes1=12, modes2=12, width=32, n_blocks=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FNO2d(modes1, modes2, width, n_blocks)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, coeff_field):
        if coeff_field.ndim == 2:
            coeff_field = coeff_field[np.newaxis, np.newaxis, :, :]
        elif coeff_field.ndim == 3:
            coeff_field = coeff_field[:, np.newaxis, :, :]

        x = torch.tensor(coeff_field, dtype=torch.float32).to(self.device)
        pred = self.model(x)
        return pred.squeeze(1).cpu().numpy()


if __name__ == "__main__":
    engine = FNOInferenceEngine("checkpoint/fno_best.pt")

    a_field = gaussian_random_field(64, seed=999)

    t0 = time.time()
    u_pred = engine.predict(a_field)
    t1 = time.time()
    print(f"Inference time: {1000*(t1-t0):.2f} ms")

    u_true = solve_darcy_2d(a_field)
    err = np.linalg.norm(u_pred[0] - u_true) / np.linalg.norm(u_true)
    print(f"Relative L2 error: {err:.6f}")