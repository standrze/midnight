#!/usr/bin/env python3
"""Exercise pinned patch replay, idempotence and conflict atomicity on temporary copies."""

from pathlib import Path
import re
import runpy
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
CHECKOUT = ROOT / ".build/checkouts/mlx-swift-lm"
PIN = "14414441fa44f45eee35a61e9fa0bab577cf9734"
GEMMA = runpy.run_path(str(ROOT / "Scripts/gemma4-window-cache-patches.py"))
MTP = runpy.run_path(str(ROOT / "Scripts/mtp-adaptive-draft-patches.py"))
MASK = "mlx-swift-lm-gemma-assistant-unmasked.patch"


def command(*arguments, cwd=None, check=True):
    return subprocess.run(arguments, cwd=cwd, text=True, capture_output=True, check=check)


def snapshot(fixture, paths):
    return {path: (fixture / path).read_bytes() if (fixture / path).exists() else None for path in paths}


def check_stack(patches, conflict_path, old, new):
    overlays = [ROOT / "Patches" / name for name in patches]
    paths = sorted(set().union(*(GEMMA["patch_paths"](patch) for patch in overlays)))
    with tempfile.TemporaryDirectory(prefix="midnight-mtp-replay-") as temporary:
        fixture = Path(temporary)
        for path in paths:
            result = command("git", "-C", str(CHECKOUT), "show", f"{PIN}:{path}", check=False)
            if result.returncode == 0:
                destination = fixture / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(result.stdout)
        command("git", "init", "-q", cwd=fixture)
        command("git", "add", ".", cwd=fixture)
        command("git", "-c", "user.name=Patch Test", "-c", "user.email=patch-test@example.invalid",
                "-c", "commit.gpgsign=false", "commit", "-qm", "Pinned source fixture", cwd=fixture)
        revision = command("git", "rev-parse", "HEAD", cwd=fixture).stdout.strip()
        before = snapshot(fixture, paths)
        try:
            GEMMA["prepare"](ROOT, fixture, patches=patches)
            raise AssertionError("Production helper accepted an incorrect dependency revision")
        except ValueError as error:
            assert "unexpected mlx-swift-lm revision" in str(error)
        assert before == snapshot(fixture, paths)
        assert GEMMA["prepare"](ROOT, fixture, patches=patches, expected_revision=revision)
        applied = snapshot(fixture, paths)
        assert not GEMMA["prepare"](ROOT, fixture, patches=patches, expected_revision=revision)
        assert applied == snapshot(fixture, paths)
        command("git", "diff", "--check", cwd=fixture)

        source = fixture / conflict_path
        assert old in source.read_text()
        source.write_text(source.read_text().replace(old, new, 1))
        conflicting = snapshot(fixture, paths)
        try:
            GEMMA["prepare"](ROOT, fixture, patches=patches, expected_revision=revision)
            raise AssertionError("Conflicting partial overlay was silently accepted")
        except ValueError:
            pass
        assert conflicting == snapshot(fixture, paths), "Conflict handling mutated the fixture"
        print(f"Passed replay, idempotence, revision and conflict atomicity: {patches[-1]}")


def main():
    actual = command("git", "-C", str(CHECKOUT), "rev-parse", "HEAD").stdout.strip()
    assert actual == PIN
    for name in ("mlx-swift-lm-mtp-adaptive-drafts.patch", "mlx-swift-lm-mtp-stateless-adaptation.patch"):
        adaptive = (ROOT / "Patches" / name).read_text()
        added = "\n".join(line[1:] for line in adaptive.splitlines() if line.startswith("+") and not line.startswith("+++"))
        assert not re.search(r"\b(eval|asyncEval|synchronize)\s*\(", added), "Adaptive path added an evaluation boundary"
    check_stack(
        MTP["PATCHES"], "Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift",
        "private var lastRoundUsedNativeHybridRewind: Bool?",
        "private var lastRoundUsedNativeHybridRewind: Bool? = true",
    )
    gemma_patches = tuple(name for name in GEMMA["PATCHES"] if name != MASK) + (MASK,)
    check_stack(
        gemma_patches, "Libraries/MLXLLM/Models/Gemma4TextAssistant.swift",
        'ProcessInfo.processInfo.environment["MIDNIGHT_GEMMA_ASSISTANT_UNMASKED"] == "1"',
        'ProcessInfo.processInfo.environment["MIDNIGHT_GEMMA_ASSISTANT_UNMASKED"] == "2"',
    )
    with tempfile.TemporaryDirectory(prefix="midnight-mtp-test-sources-") as temporary:
        fixture = Path(temporary)
        patch = str(ROOT / "Patches/midnight-mtp-runtime-tests.patch")
        command("git", "apply", "--check", "--whitespace=error-all", patch, cwd=fixture)
        command("git", "apply", patch, cwd=fixture)
        assert len(list(fixture.rglob("*.swift"))) == 3
        command("git", "apply", "--reverse", "--check", patch, cwd=fixture)
        command("git", "apply", "--reverse", patch, cwd=fixture)
        assert not list(fixture.rglob("*.swift"))
        print("Passed staged project test source apply/reverse checks")


if __name__ == "__main__":
    main()
