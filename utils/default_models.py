import urllib.request
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError

from tqdm import tqdm


default_models = {
    "encoder": ("https://drive.usercontent.google.com/download?id=1qbQCsSvO_yBWNd-o5oKof2J9M6i-XSdP&export=download&authuser=0&confirm=t&uuid=732920b9-8b71-4148-9a2f-dfbe02832476&at=APZUnTV3XNYoXWUgAYaD51loQmhb:1698458050756", 17090379),
    "synthesizer": ("https://drive.usercontent.google.com/download?id=1gUsGqzXB0z-CUVx7Gbl45ZDdDKVp8KRb&export=download&authuser=0&confirm=t&uuid=dc16fd1e-f57a-4f14-91ae-0a93303dfd81&at=APZUnTVgNtRAPPGDmxeGHceO8sTI:1698458098268", 370554559),
    "vocoder": ("https://drive.usercontent.google.com/download?id=19Hh9JhdqNtVxz2K-9ZTNfE_6Cj6awSUH&export=download&authuser=0&confirm=t&uuid=9f994e59-d3d5-4361-b820-4718965b8f84&at=APZUnTUb8gAxJa55wnmFiaxMOK0C:1698458132086", 53845290),
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(url: str, target: Path, bar_pos=0):
    # Ensure the directory exists
    target.parent.mkdir(exist_ok=True, parents=True)

    desc = f"Downloading {target.name}"
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=desc, position=bar_pos, leave=False) as t:
        try:
            urllib.request.urlretrieve(url, filename=target, reporthook=t.update_to)
        except HTTPError:
            return


def ensure_default_models(run_id: str, models_dir: Path):
    # Define download tasks
    jobs = []
    for model_name, (url, size) in default_models.items():
        target_path = models_dir / run_id / f"{model_name}.pt"
        if target_path.exists():
            # if target_path.stat().st_size != size:
            #     print(f"File {target_path} is not of expected size, redownloading...")
            # else:
            continue

        thread = Thread(target=download, args=(url, target_path, len(jobs)))
        thread.start()
        jobs.append((thread, target_path, size))

    # Run and join threads
    for thread, target_path, size in jobs:
        thread.join()

        assert target_path.exists(), \
            f"Download for {target_path.name} failed. You may download models manually instead.\n" \
            f"https://drive.google.com/drive/folders/11DFU_JBGet_HEwUoPZGDfe-fDZ42eqiG"
