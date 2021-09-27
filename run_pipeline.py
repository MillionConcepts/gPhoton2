from gPhoton.pipeline import pipeline

if __name__ == "__main__":
    pipeline(
        23863,
        "NUV",
        depth=30,
        threads=None,
        data_root="test_data",
        download=True,
    )
