import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
from dyn_models import generate_lti_sample, generate_drone_sample
from core import Config
from tqdm import tqdm
import pickle

import threading
import concurrent.futures

lock = threading.Lock()
logger = logging.getLogger(__name__)
config = Config()
config.parse_args()

def generate_sample(_):
    if config.dataset_typ == "drone":
        _, sample = generate_drone_sample(config.n_positions, sigma_w=config.sigma_w, sigma_v=config.sigma_v, dt=1e-1)
    else:
        _, sample = generate_lti_sample(config.dataset_typ, config.n_positions, config.nx, config.ny,
                                       sigma_w=config.sigma_w, sigma_v=config.sigma_v, n_noise=config.n_noise)
    return sample

if __name__ == "__main__":
    for name, num_tasks in zip(["train", "val"], [config.num_tasks, config.num_val_tasks]):
        print("Generating", num_tasks, "samples for", name)
        samples = []

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for sample in tqdm(executor.map(generate_sample, range(num_tasks)),
                               total=num_tasks):
                sample.pop("inputs", None)
                sample.pop("statesCL", None)
                sample.pop("obsCL", None)
                samples.append(sample)

        os.makedirs("../data", exist_ok=True)
        with open(f"../data/{name}_{config.dataset_typ}.pkl", "wb") as f:
            pickle.dump(samples, f)