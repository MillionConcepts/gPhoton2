from gPhoton.pipeline import pipeline

if __name__ == "__main__":
    pipeline(
        22650,
        "NUV",
        depth=30,
        threads=4,
        data_root="test_data",
        download=True
    )
