def print_log(
        title: str,
        content: str,
        sep: str = ": ",
):
    print(f"\033[1m{title}\033[0m", end=sep)
    print(content)
