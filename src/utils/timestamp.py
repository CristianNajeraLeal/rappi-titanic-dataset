"""
This module provides utility functions for converting and formatting timestamps.

Functions:
    fmt_ts_millis(millis, as_utc=False): Converts a timestamp in milliseconds to a formatted string.
    fmt_ts_seconds(seconds, as_utc=False): Converts a timestamp in seconds to a formatted string.
"""

import time


TS_FORMAT = "%Y-%m-%d %H:%M:%S"
ts_now_seconds = round(time.time())
ts_now_fmt_utc = time.strftime(TS_FORMAT, time.gmtime(ts_now_seconds))
ts_now_fmt_local = time.strftime(TS_FORMAT, time.localtime(ts_now_seconds))


def fmt_ts_millis(millis, as_utc=False):
    """
    Convert a timestamp in milliseconds to a formatted string.
    :param millis: The timestamp in milliseconds.
    :param as_utc: True to format the timestamp in UTC, False to use local time zone.
    :return: The formatted timestamp as a string.
    """
    return fmt_ts_seconds(round(millis/1000), as_utc)


def fmt_ts_seconds(seconds, as_utc=False):
    """
    Convert a timestamp in seconds to a formatted string.
    :param seconds: The timestamp in seconds.
    :param as_utc: True to format the timestamp in UTC, False to use local time zone.
    :return: The formatted timestamp as a string.
    """
    ts_format = "%Y-%m-%d %H:%M:%S"
    if as_utc:
        ts = time.gmtime(seconds)
    else:
        ts = time.localtime(seconds)
    return time.strftime(ts_format, ts)
