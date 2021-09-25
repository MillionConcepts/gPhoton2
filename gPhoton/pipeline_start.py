from time import time


class Stopwatch:
    def __init__(self, digits=2):
        self.digits = digits
        self.last_time = None

    def peek(self):
        return round(time() - self.last_time, self.digits)

    def start(self):
        print("starting timer")
        self.last_time = time()

    def click(self):
        if self.last_time is None:
            return self.start()
        print(
            f"{self.peek()} elapsed seconds, restarting timer"
        )
        self.last_time = time()


def eclipse_to_files(eclipse, data_directory="data", depth=None):
    zpad = str(eclipse).zfill(5)
    eclipse_path = f"{data_directory}/e{zpad}/"
    eclipse_base = f"{eclipse_path}e{zpad}"
    bands = "NUV", "FUV"
    band_initials = "n", "f"
    file_dict = {}
    for band, initial in zip(bands, band_initials):
        prefix = f"{eclipse_base}-{initial}d"
        band_dict = {
            "raw6": f"{prefix}-raw6.fits.gz",
            "photonfile": f"{prefix}.parquet",
            "image": f"{prefix}-full.fits.gz",
        }
        if depth is not None:
            band_dict |= {
                "movie": f"{prefix}-{depth}s.fits.gz",
                "photomfile": f"{prefix}-{depth}s-photom.csv",
                "expfile": f"{prefix}-{depth}s-exptime.csv",
            }
        file_dict[band] = band_dict
    return file_dict