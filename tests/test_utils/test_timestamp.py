from src.utils.timestamp import fmt_ts_millis, fmt_ts_seconds, TS_FORMAT
import time


def test_fmt_ts_seconds_utc():
    # Fixed timestamp for testing (e.g., Jan 1, 2020, 00:00:00 UTC)
    fixed_timestamp = 1577836800  # This is equivalent to 2020-01-01 00:00:00 UTC
    expected_utc = "2020-01-01 00:00:00"

    # Test formatting in UTC
    formatted_utc = fmt_ts_seconds(fixed_timestamp, as_utc=True)
    assert formatted_utc == expected_utc, f"Expected {expected_utc}, got {formatted_utc} in UTC"


def test_fmt_ts_seconds_local():
    # The same fixed timestamp will be used for local testing
    # Note: The expected local time result will depend on the time zone of the machine running the test
    fixed_timestamp = 1577836800  # This is equivalent to 2020-01-01 00:00:00 UTC
    expected_local = time.strftime(TS_FORMAT, time.localtime(fixed_timestamp))

    # Test formatting in local time
    formatted_local = fmt_ts_seconds(fixed_timestamp, as_utc=False)
    assert formatted_local == expected_local, f"Expected {expected_local}, got {formatted_local} in local time"


def test_fmt_ts_millis_utc():
    # Testing with milliseconds, ensuring conversion is accurate
    fixed_timestamp_millis = 1577836800000  # This is equivalent to 2020-01-01 00:00:00 UTC in milliseconds
    expected_utc = "2020-01-01 00:00:00"

    # Convert and format in UTC
    formatted_utc = fmt_ts_millis(fixed_timestamp_millis, as_utc=True)
    assert formatted_utc == expected_utc, f"Expected {expected_utc}, got {formatted_utc} from millis in UTC"


def test_fmt_ts_millis_local():
    # Again, the expected local time result will depend on the time zone of the test environment
    fixed_timestamp_millis = 1577836800000  # Milliseconds
    expected_local = time.strftime(TS_FORMAT, time.localtime(fixed_timestamp_millis / 1000))

    # Convert and format in local time
    formatted_local = fmt_ts_millis(fixed_timestamp_millis, as_utc=False)
    assert formatted_local == expected_local, f"Expected {expected_local}, got {formatted_local} from millis in local time"
