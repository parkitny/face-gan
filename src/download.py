import urllib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os


def _download(url: str, output_path: Path) -> None:
    fn = output_path / Path(url).name
    _ = urllib.request.urlretrieve(url, fn)


def copy_test_files_to_local(ctx):
    num_test_shards = ctx.data.celeb_a.shards
    urls = []
    for shard in range(num_test_shards):
        urls.append(
            "https://storage.googleapis.com/"
            + "celeb_a_dataset/celeb_a/"
            + f"{ctx.data.celeb_a.version}/"
            + "celeb_a-test."
            + f"tfrecord-0000{shard}-of-0000{num_test_shards}"
        )
    Path(ctx.paths.test).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=min(len(urls), os.cpu_count())) as executor:
        results = [
            executor.submit(_download, url, Path(ctx.paths.test)) for (url) in urls
        ]
        for future in tqdm(as_completed(results), total=len(urls)):
            _ = future.result()
