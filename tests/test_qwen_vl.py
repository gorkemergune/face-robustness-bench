"""Qwen2.5-VL backend with a mocked Ollama client (no server needed)."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest

from frbench import cli
from frbench.backends import ArcFaceRecognizer, QwenVLRecognizer
from frbench.backends.base import BACKEND_ERROR, INVALID_JSON, REFUSAL, UNREADABLE
from frbench.backends.qwen_vl import (
    PROMPT,
    Answer,
    HttpOllamaClient,
    OllamaError,
    encode_image,
    parse_answer,
)
from frbench.embedder import load_image_bgr
from frbench.evaluate import BenchConfig, run_all
from frbench.gallery import list_images
from frbench.report import comparison_table, summarize, write_reports

from conftest import NO_FACE, FakeEmbedder

REFUSAL_TEXT = "I'm sorry, but I can't help with identifying or comparing real people."


def decode(path: Path) -> tuple[int, bool, float]:
    """(identity, has_face, noise) from a conftest-coded PNG (lossless source)."""
    b, g, r = (int(v) for v in load_image_bgr(path)[0, 0])
    return r, b != NO_FACE, b / 100.0


class FakeOllama:
    """Answers from the encoded identity of the two images.

    ``encode_image`` is patched to pass file paths through (see the autouse
    fixture), so the fake can decode the lossless PNG codes directly.

    - probe without a face (NO_FACE code)  -> refusal text
    - probe noise >= 0.9 ("makeup")         -> correct but uncertain answer
    - files listed in ``invalid``           -> non-JSON chatter
    """

    def __init__(self, invalid: set[str] = frozenset()):
        self.invalid = set(invalid)
        self.calls: list[dict] = []
        self.checked: list[str] = []
        self.fail_next = 0

    def check_model(self, model):
        self.checked.append(model)

    def chat(self, model, prompt, images, options, format=None):
        self.calls.append({"model": model, "prompt": prompt, "n_images": len(images),
                           "options": options, "format": format})
        if self.fail_next:
            self.fail_next -= 1
            raise OllamaError("connection refused")
        ref, probe = (Path(i) for i in images)
        (ref_id, _, _), (probe_id, face, noise) = decode(ref), decode(probe)
        if not face:
            return REFUSAL_TEXT
        if probe.name in self.invalid:
            return "They look kind of similar, hard to say."
        same = ref_id == probe_id
        conf = 55 if noise >= 0.9 else 90
        return f'```json\n{{"same": {str(same).lower()}, "confidence": {conf}}}\n```'


@pytest.fixture(autouse=True)
def passthrough_images(request, monkeypatch):
    if "real_encode" not in request.keywords:
        def fake_encode(path, max_side):
            load_image_bgr(path)  # still fail on unreadable files, like the real encoder
            return path

        monkeypatch.setattr("frbench.backends.qwen_vl.encode_image", fake_encode)


@pytest.fixture
def fake_ollama():
    return FakeOllama()


def qwen(client, tmp_path=None, **kw) -> QwenVLRecognizer:
    cache = tmp_path / "cache" if tmp_path is not None else None
    return QwenVLRecognizer(client=client, cache_dir=cache, **kw)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text,same,conf",
    [
        ('{"same": true, "confidence": 87}', True, 87),
        ('```json\n{"same": false, "confidence": 12}\n```', False, 12),
        ('Sure! {"same": "yes", "confidence": "70%"} Hope this helps.', True, 70),
        ('{"same": "False", "confidence": 0}', False, 0),
        ('{"same": true, "confidence": 100.0, "reason": "same eyes"}', True, 100),
    ],
)
def test_parse_valid(text, same, conf):
    a = parse_answer(text)
    assert a.valid and a.same is same and a.confidence == conf


@pytest.mark.parametrize(
    "text,kind",
    [
        (REFUSAL_TEXT, REFUSAL),
        ("I cannot determine whether these are the same person.", REFUSAL),
        ("As an AI, I do not identify people.", REFUSAL),
        ("They look similar.", INVALID_JSON),
        ("", INVALID_JSON),
        ('{"same": true}', INVALID_JSON),
        ('{"same": true, "confidence": 140}', INVALID_JSON),
        ('{"same": "maybe", "confidence": 50}', INVALID_JSON),
        ('{"same": true, "confidence": true}', INVALID_JSON),
        ('[true, 80]', INVALID_JSON),
        ('{"same": tru', INVALID_JSON),
    ],
)
def test_parse_failures(text, kind):
    a = parse_answer(text)
    assert not a.valid and a.error == kind


def test_p_same_mapping():
    assert Answer(True, 80).p_same == pytest.approx(0.8)
    assert Answer(False, 80).p_same == pytest.approx(0.2)  # certain it's different
    assert Answer(False, 0).p_same == pytest.approx(0.0)  # "0% same" (qwen2.5vl style)
    assert Answer(False, 50).p_same == pytest.approx(0.5)
    assert all(Answer(False, c).p_same <= 0.5 for c in range(0, 101, 5))


# --------------------------------------------------------------------------- #
# Aggregation and references
# --------------------------------------------------------------------------- #


def test_aggregate_majority_and_mean():
    r = qwen(FakeOllama())
    out = r.aggregate(Path("p.png"), [Answer(True, 90), Answer(True, 70), Answer(False, 60)])
    assert out.valid
    assert out.score == pytest.approx((0.9 + 0.7 + 0.4) / 3)  # False/60 -> 0.4
    assert out.extra["vote_same"] is True and out.extra["n_same"] == 2
    assert out.extra["mean_confidence"] == pytest.approx(220 / 3)


def test_aggregate_tie_broken_by_score():
    r = qwen(FakeOllama())
    out = r.aggregate(Path("p.png"), [Answer(True, 90), Answer(False, 60), Answer(None, None, REFUSAL)])
    assert out.valid and out.extra["n_valid"] == 2
    assert out.extra["vote_same"] is True  # mean p_same 0.65 >= 0.5


def test_aggregate_too_few_valid_is_failure_row():
    r = qwen(FakeOllama())
    answers = [Answer(True, 90), Answer(None, None, REFUSAL), Answer(None, None, INVALID_JSON)]
    out = r.aggregate(Path("p.png"), answers)
    assert not out.valid and out.score is None
    assert out.error == REFUSAL  # tie -> refusal has priority
    assert "invalid_json=1" in out.message and "refusal=1" in out.message


def test_aggregate_majority_failure_kind():
    r = qwen(FakeOllama())
    answers = [Answer(None, None, INVALID_JSON)] * 2 + [Answer(None, None, REFUSAL)]
    assert r.aggregate(Path("p.png"), answers).error == INVALID_JSON


def test_references_deterministic_distinct_and_varied():
    r = qwen(FakeOllama())
    gallery = [Path(f"g{i}.jpg") for i in range(12)]
    a = r.references(gallery, "probe-sha-1")
    assert a == r.references(gallery, "probe-sha-1")
    assert len(a) == 3 and len(set(a)) == 3
    picks = {tuple(r.references(gallery, f"sha-{i}")) for i in range(20)}
    assert len(picks) > 1  # different probes see different references
    assert r.references(gallery[:2], "x") == gallery[:2]


# --------------------------------------------------------------------------- #
# Querying: prompt, determinism options, cache
# --------------------------------------------------------------------------- #


def test_query_uses_prompt_two_images_temperature_zero_seed(dataset, fake_ollama):
    r = qwen(fake_ollama)
    gallery = list_images(dataset / "gallery" / "alice")
    t = r.enroll("alice", gallery)
    out = r.score(t, [dataset / "probes" / "alice" / "filter" / "f_0.png"])[0]
    assert out.valid and out.extra["vote_same"] is True
    assert len(fake_ollama.calls) == 3
    call = fake_ollama.calls[0]
    assert call["prompt"] == PROMPT and call["n_images"] == 2
    assert call["options"] == {"temperature": 0, "seed": 42}
    assert call["format"] is None
    assert fake_ollama.checked == ["qwen2.5vl:7b"]


def test_structured_mode_sends_schema(dataset, fake_ollama):
    r = qwen(fake_ollama, structured=True)
    t = r.enroll("bob", list_images(dataset / "gallery" / "bob"))
    r.score(t, [dataset / "probes" / "bob" / "filter" / "f_0.png"])
    assert fake_ollama.calls[0]["format"]["required"] == ["same", "confidence"]


def test_responses_cached_on_disk_by_hash_pair(dataset, fake_ollama, tmp_path):
    gallery = list_images(dataset / "gallery" / "alice")
    probes = list_images(dataset / "probes" / "alice" / "filter")
    first = qwen(fake_ollama, tmp_path)
    s1 = first.score(first.enroll("alice", gallery), probes)
    n_calls = len(fake_ollama.calls)
    files = list((tmp_path / "cache" / "vlm").rglob("*.json"))
    assert n_calls == 3 * len(probes) and len(files) == n_calls
    record = json.loads(files[0].read_text())
    assert {"reference_sha256", "probe_sha256", "response", "model"} <= record.keys()

    second = qwen(fake_ollama, tmp_path)
    s2 = second.score(second.enroll("alice", gallery), probes)
    assert len(fake_ollama.calls) == n_calls  # all from cache
    assert second.stats["cache_hits"] == 3 * len(probes)
    assert [o.score for o in s1] == [o.score for o in s2]


def test_cache_invalidated_by_model_change(dataset, fake_ollama, tmp_path):
    gallery = list_images(dataset / "gallery" / "alice")
    probe = [dataset / "probes" / "alice" / "filter" / "f_0.png"]
    a = qwen(fake_ollama, tmp_path)
    a.score(a.enroll("alice", gallery), probe)
    b = qwen(fake_ollama, tmp_path, model="qwen2.5vl:3b")
    b.score(b.enroll("alice", gallery), probe)
    assert len(fake_ollama.calls) == 6


def test_transport_error_is_failure_and_not_cached(dataset, fake_ollama, tmp_path):
    gallery = list_images(dataset / "gallery" / "alice")
    probe = [dataset / "probes" / "alice" / "filter" / "f_0.png"]
    fake_ollama.fail_next = 3
    r = qwen(fake_ollama, tmp_path)
    out = r.score(r.enroll("alice", gallery), probe)[0]
    assert not out.valid and out.error == BACKEND_ERROR
    retry = qwen(fake_ollama, tmp_path).score(r.enroll("alice", gallery), probe)[0]
    assert retry.valid


def test_refusal_and_invalid_are_failure_rows(dataset, tmp_path):
    client = FakeOllama(invalid={"f_1.png"})
    r = qwen(client)
    t = r.enroll("alice", list_images(dataset / "gallery" / "alice"))
    effect = r.score(t, [dataset / "probes" / "alice" / "effect" / "e_0.png"])[0]  # no-face code
    assert not effect.valid and effect.error == REFUSAL
    invalid = r.score(t, [dataset / "probes" / "alice" / "filter" / "f_1.png"])[0]
    assert not invalid.valid and invalid.error == INVALID_JSON
    assert r.stats[REFUSAL] == 3 and r.stats[INVALID_JSON] == 3


def test_unreadable_probe_does_not_crash(dataset, fake_ollama):
    bad = dataset / "probes" / "alice" / "filter" / "broken.jpg"
    bad.write_bytes(b"not an image")
    r = qwen(fake_ollama)
    out = r.score(r.enroll("alice", list_images(dataset / "gallery" / "alice")), [bad])[0]
    assert not out.valid and out.error == UNREADABLE
    assert fake_ollama.calls == []


# --------------------------------------------------------------------------- #
# Full benchmark with both backends
# --------------------------------------------------------------------------- #


@pytest.fixture
def both(dataset, tmp_path, fake_ollama):
    cfg = BenchConfig(dataset, tmp_path / "results", use_cache=False)
    return run_all(cfg, [ArcFaceRecognizer(FakeEmbedder()), qwen(fake_ollama, max_impostors_per_person=20)])


def test_both_backends_same_split_and_probes(both):
    arc, vl = both
    assert vl.backend == "qwen_vl" and arc.backend == "arcface"
    key = ["person", "role", "category", "file"]
    clean = lambda r: r.rows[r.rows.role.isin(["clean", "probe"])][key].reset_index(drop=True)  # noqa: E731
    assert clean(arc).equals(clean(vl))


def test_qwen_calibrated_with_capped_impostors(both):
    vl = both[1]
    imp = vl.rows[vl.rows.role == "impostor"]
    assert imp.groupby("person").size().max() <= 20
    assert vl.threshold.n_impostor == len(imp[imp.valid])
    assert vl.threshold.genuine_mean > vl.threshold.impostor_mean
    assert vl.rows[vl.rows.role == "clean"].recognized.all()
    assert not imp.recognized.any()


def test_qwen_failures_counted_separately(both):
    vl = both[1]
    s = summarize(vl.rows, vl.categories).set_index("category")
    n = len(vl.persons)
    assert s.loc["effect", "refusal"] == 2 * n
    assert s.loc["effect", "detection_fails"] == 0
    assert s.loc["effect", "failures"] == 2 * n
    assert s.loc["effect", "probes"] == 4 * n  # failures not dropped
    assert s.loc["makeup", "vote_rate"] == 100.0
    assert s.loc["makeup", "mean_score"] < s.loc["filter", "mean_score"]
    arc = summarize(both[0].rows, both[0].categories).set_index("category")
    assert arc.loc["effect", "no_face"] == 2 * n and arc.loc["effect", "refusal"] == 0


def test_report_side_by_side(both, tmp_path):
    table = comparison_table(both)
    header = table.splitlines()[0]
    for col in ("ArcFace rec %", "Qwen-VL rec %", "ArcFace face score", "Qwen-VL failures"):
        assert col in header
    effect = next(line for line in table.splitlines() if line.startswith("| effect"))
    assert "no_face" in effect and "refusal" in effect

    paths = write_reports(both, tmp_path / "RESULTS.md")
    md = paths["results_md"].read_text()
    assert "## Comparison" in md and "## ArcFace" in md and "## Qwen-VL" in md
    assert "Vote same %" in md
    payload = json.loads(paths["summary_json"].read_text())
    assert set(payload["backends"]) == {"arcface", "qwen_vl"}
    assert payload["backends"]["qwen_vl"]["stats"]["refusal"] > 0


def test_preflight_fails_fast(dataset, tmp_path):
    class Down(FakeOllama):
        def check_model(self, model):
            raise OllamaError("Ollama is not running")

    emb = FakeEmbedder()
    with pytest.raises(OllamaError):
        run_all(BenchConfig(dataset, tmp_path / "r", use_cache=False),
                [ArcFaceRecognizer(emb), qwen(Down())])
    assert emb.calls == 0  # nothing expensive ran before the check


def test_cli_runs_both_backends(dataset, tmp_path, fake_ollama, capsys):
    out = tmp_path / "results"
    code = cli.main(
        ["run", "--data", str(dataset), "--out", str(out), "--results-md", "",
         "--backends", "arcface,qwen", "--qwen-max-impostors", "15"],
        factories={
            "arcface": lambda args, cache: ArcFaceRecognizer(FakeEmbedder(), cache),
            "qwen_vl": lambda args, cache: QwenVLRecognizer(
                fake_ollama, cache_dir=cache,
                max_impostors_per_person=args.qwen_max_impostors or None),
        },
    )
    assert code == 0
    text = capsys.readouterr().out
    assert "Comparison" in text and "Qwen-VL rec %" in text
    raw = (out / "raw" / "alice.csv").read_text()
    assert "arcface" in raw and "qwen_vl" in raw
    assert list((out / "cache" / "vlm").rglob("*.json"))


def test_cli_reports_ollama_down(dataset, tmp_path, capsys):
    class Down(FakeOllama):
        def check_model(self, model):
            raise OllamaError("Ollama request /api/tags failed: connection refused")

    code = cli.main(
        ["run", "--data", str(dataset), "--out", str(tmp_path / "r"), "--results-md", "",
         "--backends", "qwen"],
        factories={"qwen_vl": lambda args, cache: QwenVLRecognizer(Down())},
    )
    assert code == 2
    assert "ollama serve" in capsys.readouterr().err


def test_cli_rejects_unknown_backend():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run", "--backends", "gpt"])


# --------------------------------------------------------------------------- #
# Real HTTP client against a patched urlopen
# --------------------------------------------------------------------------- #


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_http_client_payload(monkeypatch):
    sent = {}

    def fake_urlopen(req, timeout):
        sent["url"], sent["body"] = req.full_url, json.loads(req.data)
        return FakeResponse(json.dumps({"message": {"content": '{"same": true, "confidence": 90}'}}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    text = HttpOllamaClient("http://ollama:11434/").chat(
        "qwen2.5vl:7b", PROMPT, ["AAA", "BBB"], {"temperature": 0, "seed": 42}, None
    )
    assert text == '{"same": true, "confidence": 90}'
    assert sent["url"] == "http://ollama:11434/api/chat"
    body = sent["body"]
    assert body["stream"] is False and body["options"] == {"temperature": 0, "seed": 42}
    assert body["messages"][0]["images"] == ["AAA", "BBB"]
    assert "format" not in body


def test_http_client_missing_model(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout: FakeResponse(json.dumps({"models": [{"name": "llava:latest"}]}).encode()),
    )
    with pytest.raises(OllamaError, match="ollama pull qwen2.5vl:7b"):
        HttpOllamaClient().check_model("qwen2.5vl:7b")


def test_http_client_server_down(monkeypatch):
    import urllib.error

    def refuse(req, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(OllamaError, match="connection refused"):
        HttpOllamaClient(retries=1).check_model("qwen2.5vl:7b")


@pytest.mark.real_encode
def test_encode_image_applies_exif_and_downscales(tmp_path):
    from PIL import Image

    img = Image.new("RGB", (2000, 1000), (0, 255, 0))
    exif = Image.Exif()
    exif[0x0112] = 6
    path = tmp_path / "big.jpg"
    img.save(path, exif=exif)
    decoded = Image.open(io.BytesIO(base64.b64decode(encode_image(str(path), 768))))
    assert decoded.size == (384, 768) and decoded.format == "JPEG"
    assert np.asarray(decoded)[0, 0, 1] > 200
