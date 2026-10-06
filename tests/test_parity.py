"""Parity against reference scores on real activations, with the published probes.

The fixture is the Hugging Face dataset ``FIXTURE_REPO`` at ``FIXTURE_REVISION``. It holds, per model, a few
rows of activations and token masks and the reference scores for those rows, but no probes. Each probe is
loaded with :func:`probe_inference.load_probe_from_hub` from the published weights at the package's pinned
``WEIGHTS_REVISION``, exactly as users load it; the fixture's ``manifest.json`` names its path there. Both
downloads go through the Hub cache (``HF_HOME``), so repeated runs and cached CI runs reuse them.

- ``nemotron-3-super-120b``: EFC and axial; one batch mixes 27- and 24-token read windows;
- ``qwen3.5-9b``: linear, MLP, EFC and axial.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from probe_inference import load_probe_from_hub
from probe_inference.load import WEIGHTS_REPO, WEIGHTS_REVISION

FIXTURE_REPO = "AlignmentResearch/probe-inference-parity"
FIXTURE_REVISION = "14382a5fe404dc1ce7a13edbacaecd3964024353"
SETS = {"nemotron-3-super-120b": ("efc", "axial"), "qwen3.5-9b": ("linear", "mlp", "efc", "axial")}
STACKED_TOLERANCE = {"efc": 5e-5, "axial": 1e-5}
PER_LAYER = ("linear", "mlp")
#: Every (set, architecture) the fixture has references for is exactly one of these cases.
STACKED_CASES = [(name, arch) for name, archs in SETS.items() for arch in archs if arch not in PER_LAYER]
PER_LAYER_CASES = [(name, arch) for name, archs in SETS.items() for arch in archs if arch in PER_LAYER]
#: Every probe published in the weights repository, as ``<model>/<arch>``.
PUBLISHED_MODELS = (
    "qwen3.5-2b",
    "qwen3.5-9b",
    "qwen3.6-27b",
    "qwen3.5-122b-a10b",
    "qwen3.5-397b-a17b",
    "nemotron-3-nano-30b-a3b",
    "nemotron-3-super-120b-a12b",
)


def download_fixture(repo: str = FIXTURE_REPO, revision: str = FIXTURE_REVISION) -> Path:
    """Download (or reuse from the Hub cache) the parity fixture and return its root."""
    from huggingface_hub import snapshot_download

    try:
        root = Path(snapshot_download(repo, repo_type="dataset", revision=revision))
    except Exception as error:
        pytest.fail(f"could not download the parity fixture {repo}@{revision}: {error!r}")
    if not (root / "manifest.json").exists():
        pytest.fail(f"{repo}@{revision} has no manifest.json; it is not a parity fixture")
    return root


@pytest.fixture(scope="session")
def fixture_root() -> Path:
    return download_fixture()


def _assert_fails(call, match: str) -> None:
    # pytest.raises(pytest.fail.Exception) lets pytest.skip's Skipped through, so check the type exactly.
    with pytest.raises(BaseException) as info:
        call()
    assert info.type is pytest.fail.Exception, f"expected pytest.fail, got {info.type.__name__}"
    info.match(match)


def test_download_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    calls = []

    def fake_snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    _assert_fails(lambda: download_fixture("owner/fixture", "abc123"), "no manifest.json")
    (tmp_path / "manifest.json").write_text("{}")
    assert download_fixture("owner/fixture", "abc123") == tmp_path
    assert calls[-1] == ("owner/fixture", {"repo_type": "dataset", "revision": "abc123"})

    def unreachable(repo_id, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", unreachable)
    _assert_fails(
        lambda: download_fixture("owner/fixture", "abc123"),
        "could not download the parity fixture owner/fixture@abc123",
    )


def _load_set(root: Path, name: str) -> tuple[dict[int, torch.Tensor], dict[str, torch.Tensor], dict]:
    base = root / name
    acts = torch.load(base / "acts.pt", map_location="cpu", weights_only=True)
    masks = torch.load(base / "masks.pt", map_location="cpu", weights_only=True)
    return acts, masks, json.loads((base / "reference.json").read_text())


def _published_probe(fixture_root: Path, name: str, arch: str):
    """The probe the fixture's references are for, loaded from the published weights at the pinned revision."""
    manifest = json.loads((fixture_root / "manifest.json").read_text())
    return load_probe_from_hub(manifest["sets"][name]["probes"][arch], revision=WEIGHTS_REVISION)


def test_manifest_and_references_cover_every_tested_probe(fixture_root: Path) -> None:
    """Every probe the fixture has reference scores for is one these tests load and check."""
    manifest = json.loads((fixture_root / "manifest.json").read_text())
    assert set(manifest["sets"]) == set(SETS)
    for name, archs in SETS.items():
        assert set(manifest["sets"][name]["probes"]) == set(archs)
        assert set(json.loads((fixture_root / name / "reference.json").read_text())) == set(archs)
    assert not list(fixture_root.rglob("probe_metadata.json")), "the fixture must not carry its own probes"
    assert sorted(STACKED_CASES + PER_LAYER_CASES) == sorted((n, a) for n, archs in SETS.items() for a in archs)
    assert set(STACKED_TOLERANCE) == {arch for _, arch in STACKED_CASES}


def test_every_published_probe_exists_at_the_pinned_revision() -> None:
    """All 28 published probes are present at ``WEIGHTS_REVISION``, not only the ones parity scores."""
    from huggingface_hub import HfApi

    files = set(HfApi().list_repo_files(WEIGHTS_REPO, revision=WEIGHTS_REVISION))
    expected = {f"{model}/{arch}" for model in PUBLISHED_MODELS for arch in ("linear", "mlp", "efc", "axial")}
    missing = sorted(probe for probe in expected if f"{probe}/probe_metadata.json" not in files)
    assert not missing, f"{WEIGHTS_REPO}@{WEIGHTS_REVISION} lacks {missing}"
    present = {path.rsplit("/", 1)[0] for path in files if path.endswith("/probe_metadata.json")}
    assert present == expected, f"unexpected probes at the pin: {sorted(present - expected)}"


def _read_mask(probe, masks: dict[str, torch.Tensor]) -> torch.Tensor:
    return probe.read_mask(masks["prompt_mask"], masks["completion_mask"], masks["followup_start_positions"])


def _max_diff(got: torch.Tensor, reference: list[float]) -> tuple[float, float]:
    ref = torch.tensor(reference, dtype=torch.float64)
    assert got.shape == ref.shape
    return float((got - ref).abs().max()), float(ref.abs().max())


@pytest.mark.parametrize(("name", "arch"), STACKED_CASES)
def test_stacked_probes_match_reference_scores(fixture_root: Path, name: str, arch: str) -> None:
    acts, masks, reference = _load_set(fixture_root, name)
    probe = _published_probe(fixture_root, name, arch)
    mask = _read_mask(probe, masks)
    if name == "nemotron-3-super-120b":
        assert len(set(mask.sum(dim=1).tolist())) > 1, "this set must hold ragged read windows"
    diff, scale = _max_diff(probe.score_batch(acts, mask), reference[arch]["score"])
    print(f"PARITY {name} {arch}: max |diff| = {diff:.3g} (max |score| {scale:.3g})")
    assert diff <= STACKED_TOLERANCE[arch]


@pytest.mark.parametrize(("name", "arch"), PER_LAYER_CASES)
def test_per_layer_probes_match_reference_scores(fixture_root: Path, name: str, arch: str) -> None:
    acts, masks, reference = _load_set(fixture_root, name)
    probe = _published_probe(fixture_root, name, arch)
    mask = _read_mask(probe, masks)
    per_layer = probe.layer_logits(acts, mask)
    assert {f"L{layer}" for layer in per_layer} == {key for key in reference[arch] if key.startswith("L")}
    for layer, logits in per_layer.items():
        diff, scale = _max_diff(logits, reference[arch][f"L{layer}"])
        print(f"PARITY {name} {arch} L{layer}: max |diff| = {diff:.3g} (max |logit| {scale:.3g})")
        assert diff <= 1e-5 * max(1.0, scale)
    combined = probe.score_batch(acts, mask)
    diff, _ = _max_diff(combined, reference[arch]["combined"])
    print(f"PARITY {name} {arch} layer mean: max |diff| = {diff:.3g}")
    assert diff <= 1e-6
    assert bool(((combined > 0.01) & (combined < 0.99)).all()), "per-layer scores should not be saturated"
