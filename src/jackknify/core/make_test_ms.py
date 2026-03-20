from jackknify.core.MSHandler import MSWrapper


def make_ms(ms_file: str, rows: int, chans: int):
    """Creates a simple mock MS filled with 1s for testing."""
    try:
        MSWrapper.create_test_ms(ms_file, n_rows=rows, n_chan=chans)
        print("Test MS created successfully.")
    except Exception as e:
        print(f"Error creating MS: {e}")
