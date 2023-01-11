import re
from datetime import datetime
from pathlib import Path


root = Path(__file__)
data = root.parent.parent / "data/raw/bbb.txt"
date_length = len("06/01/2023, 15:23")
prefix_length = len("06/01/2023, 15:23 -")
dt_pattern = re.compile(r"\d{2}/\d{2}/\d{4}, \d{2}:\d{2}")


def main():
    with data.open("r") as fi:
        for line in fi:
            if not dt_pattern.match(line):
                continue
            date_str = line[:date_length]
            dt = datetime.strptime(date_str, "%d/%m/%Y, %H:%M")
            d = dt.date()
            outpath = root.parent.parent / f"data/preprocessed/{d:%Y%m%d}.txt"
            text = line[prefix_length:].strip()
            with outpath.open("a") as fo:
                fo.write(text)
                fo.write("\n")


if __name__ == "__main__":
    main()
