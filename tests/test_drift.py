"""
Unit tests for the DriftMonitor class.

These tests do not touch the API or the model at all. They test the
statistical logic in isolation using fake input data. This means they are
fast and deterministic.
"""

from app.drift import DriftMonitor


def make_monitor():
    """
    Return a DriftMonitor initialised without a train.csv.
    The fallback baseline is mean_length=80, non_ascii_frac=0.01.
    """
    # Pass a path that definitely does not exist so the fallback branch runs.
    return DriftMonitor(train_csv_path="nonexistent_path/train.csv")


def test_report_shows_insufficient_data_with_no_records():
    """An empty window should not pretend to have useful statistics."""
    monitor = make_monitor()
    report = monitor.report()

    assert report["status"] == "insufficient_data"
    assert report["n"] == 0


def test_report_shows_insufficient_data_with_fewer_than_ten_records():
    """Nine requests is not enough to form a reliable picture."""
    monitor = make_monitor()
    for _ in range(9):
        monitor.record("This is a short English sentence.")

    assert monitor.report()["status"] == "insufficient_data"


def test_report_shows_ok_with_normal_english_text():
    """
    Ten or more typical English sentences should produce an ok status because
    they closely match the fallback baseline.
    """
    monitor = make_monitor()
    for _ in range(15):
        monitor.record(
            "This product worked exactly as described and I am happy with it."
        )

    report = monitor.report()
    assert report["status"] == "ok"
    assert report["n"] == 15


def test_report_detects_non_ascii_drift():
    """
    Sending mostly non-ASCII text should trigger a drift alert because the
    fallback baseline has a very low non_ascii_frac of 0.01.
    """
    monitor = make_monitor()
    # Prime the window with normal text first.
    for _ in range(5):
        monitor.record("Plain English text right here.")
    # Then flood with high-unicode content.
    for _ in range(20):
        monitor.record("Çok güzel bir ürün, harika kalite, çok memnunum. Teşekkürler!")

    report = monitor.report()
    assert report["status"] == "alert"
    assert "non_ascii_frac" in report["alerts"]


def test_report_detects_length_drift():
    """
    Sending only very short snippets should trigger a drift alert on mean_length
    because the fallback baseline expects around 80 characters.
    """
    monitor = make_monitor()
    for _ in range(15):
        monitor.record("ok")  # 2 characters — far below the 80-char baseline

    report = monitor.report()
    assert report["status"] == "alert"
    assert "mean_length" in report["alerts"]


def test_report_detects_oov_rate_above_threshold():
    """
    Text made up entirely of made-up words should push the OOV rate above 0.30
    when we have a real training vocabulary to compare against.
    """
    monitor = DriftMonitor()
    for _ in range(15):
        monitor.record(
            "zxqwerty flibbertigibbet wumboozle glarptastic snorkelwump"
            " plorbazine snarflequark"
        )

    report = monitor.report()
    assert report["status"] == "alert"
    assert "oov_rate" in report["alerts"]


def test_window_rolls_over_at_max_size():
    """
    After sending more than WINDOW_SIZE requests the oldest ones should be
    discarded and n should equal WINDOW_SIZE not the total sent.
    """
    from app.drift import WINDOW_SIZE

    monitor = make_monitor()

    for i in range(WINDOW_SIZE + 20):
        monitor.record(f"Request number {i} with some padding text to be realistic.")

    assert monitor.report()["n"] == WINDOW_SIZE


def test_record_does_not_crash_on_empty_string():
    """Edge case: the schema prevents empty strings but the monitor should be robust."""
    monitor = make_monitor()
    # Should not raise even with an empty string.
    monitor.record("")


def test_alerts_dict_is_empty_when_status_is_ok():
    """When status is ok the alerts dict should exist but be empty."""
    monitor = make_monitor()
    for _ in range(15):
        monitor.record("Completely ordinary English sentence about a product review.")

    report = monitor.report()
    if report["status"] == "ok":
        assert report["alerts"] == {}
