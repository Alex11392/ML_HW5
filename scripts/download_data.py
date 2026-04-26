from pathlib import Path

import gdown

from src.hw5_common import DATA_URLS


def main():
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    for file_name, url in DATA_URLS.items():
        output = data_dir / file_name
        if output.exists():
            print(f"skip {output}")
            continue
        print(f"download {output}")
        gdown.download(url, str(output), quiet=False)


if __name__ == "__main__":
    main()
