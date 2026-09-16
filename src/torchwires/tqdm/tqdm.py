from typing import Any


class Tqdm:
    def __init__(self):
        self._n_chars = 0

    def update(self, data: dict[str, Any]):
        line_content = ' | '.join(
            [
                f"{key}:{value}"
                for key, value in data.items()
            ]
        )

        if self._n_chars > len(line_content):
            line_content += ' ' * (self._n_chars - len(line_content))

        self._n_chars = len(line_content)

        print("\r" + line_content, end='')

    def new_line(self):
        print('\n', end='')
