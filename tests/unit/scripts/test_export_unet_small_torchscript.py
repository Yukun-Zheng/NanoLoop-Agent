from __future__ import annotations

import ast
from pathlib import Path


def _script_source() -> tuple[str, ast.Module]:
    project_root = Path(__file__).parents[3]
    script_path = project_root / "scripts" / "models" / "export_unet_small_torchscript.py"
    source = script_path.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=str(script_path))


def _class_source(source: str, tree: ast.Module, name: str) -> str:
    definition = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name
    )
    segment = ast.get_source_segment(source, definition)
    assert segment is not None
    return segment


def test_small_unet_export_script_has_external_artifact_and_torchscript_guards() -> None:
    source, _tree = _script_source()

    assert 'parser.add_argument("--checkpoint", required=True, type=Path' in source
    assert 'parser.add_argument("--output", required=True, type=Path' in source
    assert 'model.load_state_dict(_load_state_dict(checkpoint), strict=True)' in source
    assert "model.eval()" in source
    assert "torch.jit.script(model)" in source
    assert 'torch.jit.load(str(temporary_output), map_location="cpu")' in source
    assert "MAX_ABS_ERROR = 1e-6" in source
    assert "_publish_verified_export(temporary_output, output)" in source
    assert "temporary_output.unlink(missing_ok=True)" in source
    assert 'torch.linspace(' in source
    assert "reshape(1, 1, profile.patch_size, profile.patch_size)" in source
    assert "output path must be outside the NanoLoop-Agent repository" in source
    assert 'C:\\' not in source
    assert 'D:\\' not in source
    assert '"checkpoint_sha256": checkpoint_sha256' in source
    assert '"export_sha256": _sha256(temporary_output)' in source
    assert (
        'parser.add_argument(\n        "--expected-checkpoint-sha256",\n        required=True'
        in source
    )
    assert "reloaded_logits_repeatability" in source
    assert "reloaded_probability_repeatability" in source
    assert "is not exactly deterministic" in source


def test_small_profile_preserves_batchnorm_and_128_key_contract() -> None:
    source, tree = _script_source()
    double_conv = _class_source(source, tree, "DoubleConv")
    up = _class_source(source, tree, "Up")

    assert 'SMALL_BATCHNORM = "small_batchnorm"' in source
    assert "architecture_profile: str = SMALL_BATCHNORM" in source
    assert "default=SMALL_BATCHNORM" in source
    assert "expected_key_count=128" in source
    assert double_conv.count("nn.BatchNorm2d(out_ch)") == 2
    assert "align_corners=True" in up


def test_large_profile_matches_confirmed_groupnorm_architecture() -> None:
    source, tree = _script_source()
    double_conv = _class_source(source, tree, "LargeDoubleConv")
    up = _class_source(source, tree, "LargeUp")
    unet = _class_source(source, tree, "LargeOptimizedUNet")

    assert 'LARGE_GROUPNORM_OPTIMIZED = "large_groupnorm_optimized"' in source
    assert "expected_key_count=56" in source
    assert "patch_size=512" in source
    assert "large training cropped 130 bottom pixels" in source
    assert "crops 180 bottom pixels" in source
    assert 'report["known_risks"] = list(profile.known_risks)' in source
    assert double_conv.count("bias=False") == 2
    assert double_conv.count("nn.GroupNorm(8, out_ch)") == 2
    assert "nn.MaxPool2d(2)" in _class_source(source, tree, "LargeDown")
    assert 'mode="bilinear", align_corners=False' in up
    assert "F.pad(" in up
    assert "torch.cat([x2, x1], dim=1)" in up
    assert "self.inc = LargeDoubleConv(1, 32)" in unet
    assert "LargeDown(32, 64), LargeDown(64, 128)" in unet
    assert "LargeDown(128, 256), LargeDown(256, 256)" in unet
    assert "LargeUp(512, 128), LargeUp(256, 64)" in unet
    assert "LargeUp(128, 32), LargeUp(64, 32)" in unet
    assert "self.outc = nn.Conv2d(32, 1, 1)" in unet
    assert '"--expected-checkpoint-sha256"' in source
    assert "checkpoint SHA-256 does not match the explicit expected digest" in source


def test_profile_selection_rejects_unknown_names_and_drives_example_shape() -> None:
    source, tree = _script_source()
    profile = _class_source(source, tree, "ArchitectureProfile")
    selector = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_architecture_profile"
    )
    selector_source = ast.get_source_segment(source, selector)

    assert selector_source is not None
    assert 'raise ValueError(f"unknown architecture profile: {name}")' in selector_source
    assert "choices=sorted(ARCHITECTURE_PROFILES)" in source
    assert "steps=profile.patch_size * profile.patch_size" in source
    assert "reshape(1, 1, profile.patch_size, profile.patch_size)" in source
    assert "return UNet()" in profile
    assert "return LargeOptimizedUNet()" in profile
