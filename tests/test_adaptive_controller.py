"""Scheduler measurements must distinguish GPU limits from missing client work."""
import pytest

from adaptive_controller import Controller, apply_tuning


def point(t, *, running=64, waiting=0, cache=.4, rate=1000, preemptions=0):
    return dict(time=t, running=running, waiting=waiting, cache=cache,
                tokens=t * rate, preemptions=preemptions)


def raised_controller():
    controller = Controller(32, 384)
    event = None
    for t in range(0, 181, 15):
        event = controller.observe(point(t, running=32))
    assert event['new_cap'] == 64
    return controller


@pytest.mark.parametrize('running', [0, 20, 40])
def test_underfed_window_never_lowers_cap_or_ceiling(running):
    controller = raised_controller()
    for t in range(195, 601, 15):
        controller.observe(point(t, running=running, rate=600))
    assert controller.cap == 64 and controller.upper == 384
    assert controller.previous_rate is None
    assert not controller.throughput_samples


def test_idle_mid_probe_cannot_contaminate_recovery_comparison():
    controller = raised_controller()
    for t in range(195, 331, 15):
        controller.observe(point(t))
    controller.observe(point(345, running=0))
    for t in range(360, 541, 15):
        sample = point(t)
        sample['tokens'] = 345000 + (t-345)*600
        controller.observe(sample)
        assert controller.cap >= 64 and controller.upper == 384


def test_real_throughput_regression_under_sustained_load_still_backs_off():
    controller = raised_controller()
    event = None
    for t in range(195, 376, 15):
        sample = point(t)
        sample['tokens'] = 180000 + (t-180)*700
        event = controller.observe(sample)
    assert event['new_cap'] == 32
    assert controller.upper == 32
    assert 'throughput regressed' in event['reason']


@pytest.mark.parametrize('running', [10, 128])
def test_cache_pressure_still_backs_off_even_when_underfed(running):
    controller = Controller(128, 384)
    for t in [0, 15, 30]:
        event = controller.observe(point(t, running=running, cache=.95))
    assert event['new_cap'] == 96 and controller.upper == 96
    assert 'pressure' in event['reason']


def test_preemptions_still_trigger_pressure_backoff():
    controller = Controller(128, 384)
    controller.observe(point(0, running=128))
    for t in [15, 30, 45]:
        event = controller.observe(point(t, running=128, preemptions=1))
    assert event['new_cap'] == 96


def test_stable_scheduler_queue_still_limits_concurrency():
    controller = Controller(128, 384)
    for t in range(0, 181, 15):
        event = controller.observe(point(t, running=32, waiting=96))
    assert event['new_cap'] == 32


@pytest.mark.parametrize('disruption', ['gap', 'restart', 'clock'])
def test_missing_metrics_or_server_reset_discard_old_comparison(disruption):
    controller = raised_controller()
    sample = point(300 if disruption == 'gap' else 195)
    if disruption == 'restart':
        sample['tokens'] = 0
    elif disruption == 'clock':
        sample['time'] = 179
    assert controller.observe(sample) is None
    assert controller.cap == 64 and controller.upper == 384
    assert controller.previous_rate is None
    assert controller.throughput_samples == [sample]


def test_operator_tuning_discards_pre_change_windows():
    controller = raised_controller()
    apply_tuning({'gpu': controller}, {'gpu': 384},
                 dict(revision='repair', endpoints={'gpu':dict(cap=64, upper=384)}), None)
    assert not controller.samples and not controller.throughput_samples
    assert controller.previous_rate is None
