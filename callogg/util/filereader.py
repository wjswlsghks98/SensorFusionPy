import os

DATA_ENDPOINT = os.getenv("DATA_ENDPOINT", "http://data-raw.internal/")


def FileReader(fn, fs = None, debug=False):
    if fn.startswith("cd:/"):
        fn = fn.replace("cd:/", DATA_ENDPOINT)
    if fn.startswith("http://") or fn.startswith("https://"):
        raise Exception("Cannot Open URLs in current version")
    if fn.startswith("s3://"):
        return fs.open(fn,"rb")

    return open(fn, "rb")