"""Synthetic supplementary metrics and opt-in timeout contracts."""
import json

import numpy as np
import pytest

from . import boxinggym_core as bg


def test_location_log_nmse_uses_fixed_truth_variance_without_clipping():
    """Location log-nMSE uses the fixed truth variance and no clipping."""
    truth=np.array([.2,1.,8.,10000.])
    prediction=np.array([.3,2.,9.,3.])
    result=bg.mda_score_row(4,prediction,truth,'',location=True)
    expected=np.mean((np.log1p(prediction)-np.log1p(truth))**2)/np.var(np.log1p(truth))
    assert result['nmse_log_truth']==pytest.approx(expected)
    assert result['nmse_log_truth_undefined_points']==0
    assert result['nmse_truth']==bg.mda_score_row(4,prediction,truth,'')['nmse_truth']


def test_location_undefined_points_count_and_best_so_far():
    """Location counts undefined points and tracks the best-so-far log-nMSE."""
    truth=np.array([.2,1.,8.,20.]);prediction=np.array([-1.,-2.,np.nan,3.])
    result=bg.mda_score_row(4,prediction,truth,'',location=True,previous_best_log=.1)
    expected=(np.log1p(3.)-np.log1p(20.))**2/np.var(np.log1p(truth))
    assert result['nmse_log_truth']==pytest.approx(expected)
    assert result['nmse_log_truth_undefined_points']==3
    assert result['best_so_far_nmse_log_truth']==min(.1,expected)
    empty=bg.mda_score_row(4,[-1.,np.nan],[1.,2.],'',location=True,previous_best_log=.1)
    assert empty['nmse_log_truth']=='not_available'
    assert empty['nmse_log_truth_undefined_points']==2
    assert empty['best_so_far_nmse_log_truth']==.1


def test_other_task_sidecar_retains_exact_columns_and_bytes(tmp_path):
    """Non-location tasks keep the exact sidecar columns and bytes."""
    row=bg.mda_score_row(4,[1.,3.],[1.,2.],.5)
    path=tmp_path/'points.csv';bg.write_mda_points(path,[row])
    assert path.read_bytes()==b'x,nmse_truth,brier,best_so_far_nmse_truth\n4,2.0,0.5,2.0\n'


def test_location_sidecar_only_adds_declared_columns(tmp_path):
    """The location sidecar only adds the declared columns."""
    path=tmp_path/'points.csv'
    bg.write_mda_points(path,[bg.mda_score_row(2,[1.,2.],[1.,2.],'',location=True)])
    assert path.read_text().splitlines()[0].split(',')==['x','nmse_truth','brier','best_so_far_nmse_truth','nmse_log_truth','best_so_far_nmse_log_truth','nmse_log_truth_undefined_points']


@pytest.mark.parametrize("scope", ["active", "candidate"])
@pytest.mark.usefixtures("require_boxinggym_source")
def test_timeout_retires_once_and_cell_and_next_fits_continue(tmp_path, monkeypatch, scope):
    """A fit timeout retires the candidate once while the cell and later fits continue."""
    from .test_named_inputs import bg_run_args, bg_stub_fits
    from .test_prewave_fit_timeout import _assert_reaped, _rows, _slow_workers
    adapter = bg_stub_fits(monkeypatch)
    expression = (adapter.TASKS["death"].initial_structures()[0] if scope == "active"
                  else "b0+b1*square(t01)")
    monkeypatch.setattr(adapter.TaskSpec, "mock_templates", lambda self: (expression,))
    original_fit = adapter.count_fit
    attempts = tmp_path / "attempts.jsonl"
    processes = tmp_path / "processes.jsonl"
    def injected(hypothesis, *args, **kwargs):
        """Record the attempt, make the target expression slow, then run the original fit."""
        with attempts.open("a") as stream:
            stream.write(json.dumps({"expression": hypothesis.expression,
                                     "model_id": hypothesis.model_id}) + "\n")
        if hypothesis.expression == expression:
            _slow_workers(processes)
        return original_fit(hypothesis, *args, **kwargs)
    monkeypatch.setattr(adapter, "count_fit", injected)
    out = tmp_path / "run"
    adapter.run(bg_run_args(out, cell="100", rounds=2, fit_timeout_seconds=.3))
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "completed"
    failures = [row for row in summary["candidate_failures"] if row["status"] == "fit_timeout"]
    assert len(failures) == 1
    failure = failures[0]
    assert failure["timeout_seconds"] == .3 and failure["stage"] == "sampling"
    assert failure["cause_code"] == "wall_clock_timeout"
    assert len([row for row in _rows(attempts) if row["expression"] == expression]) == 1
    events = [event for path in sorted((out / "proposals").glob("*.json"))
              for event in json.loads(path.read_text())["events"]]
    assert any(row["status"] == "duplicate_not_refitted" for row in events)
    assert any(row["status"] == "completed" and row["round"] == 2
               for row in _rows(out / "fit_timings.jsonl"))
    assert not list((out / "fits").glob("*" + failure["model_id"] + ".npz"))
    _assert_reaped(processes)


@pytest.mark.usefixtures("require_boxinggym_source")
def test_disabled_programmatic_timeout_does_not_add_config_key(tmp_path, monkeypatch):
    """A disabled timeout adds no config key and no timing file."""
    from .test_named_inputs import bg_run_args, bg_stub_fits
    adapter = bg_stub_fits(monkeypatch)
    out = tmp_path / "disabled"
    adapter.run(bg_run_args(out, cell="000", rounds=0, fit_timeout_seconds=None))
    assert "fit_timeout_seconds" not in json.loads((out / "config.json").read_text())
    assert not (out / "fit_timings.jsonl").exists()


@pytest.mark.usefixtures("require_boxinggym_source")
def test_empty_pool_preserves_full_timeout_diagnostics(tmp_path, monkeypatch):
    """When every fit times out, the failure record keeps the full timeout diagnostics."""
    import time

    from .test_named_inputs import bg_run_args, bg_stub_fits
    adapter = bg_stub_fits(monkeypatch)
    monkeypatch.setattr(adapter, "count_fit", lambda *args, **kwargs: time.sleep(60))
    out = tmp_path / "all_timeout"
    with pytest.raises(ValueError, match="no models remain after fit_timeout"):
        adapter.run(bg_run_args(out, cell="000", rounds=0, fit_timeout_seconds=.05))
    failure = json.loads((out / "failure.json").read_text())
    assert len(failure["candidate_failures"]) == len(adapter.TASKS["death"].initial_structures())
    assert all(row["status"] == "fit_timeout" and row["timeout_seconds"] == .05
               and row["stage"] == "sampling" for row in failure["candidate_failures"])
