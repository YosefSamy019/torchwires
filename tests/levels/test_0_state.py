from src.torchwires.state.state import BatchState


def test_state():
    state = BatchState()

    print("State:", state)

    state.set("k1", 1)
    state.set("k2", "Hello")

    print("State:", state)

    state.set_all(
        keys=["l1", "l2"],
        values=["l1_val", 54]
    )

    print("State:", state)

    print("Access 'l1':", state.get('l1'))

    print("Access 'l1', 'k1':", state.get_all(['l1', 'k1']))


if __name__ == "__main__":
    test_state()
