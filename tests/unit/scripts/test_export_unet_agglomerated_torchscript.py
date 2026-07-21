from __future__ import annotations

import ast
from pathlib import Path


def _script_source() -> tuple[str, ast.Module]:
    project_root = Path(__file__).parents[3]
    path = project_root / "scripts" / "models" / "export_unet_agglomerated_torchscript.py"
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=str(path))


def _class_source(source: str, tree: ast.Module, name: str) -> str:
    definition = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name
    )
    segment = ast.get_source_segment(source, definition)
    assert segment is not None
    return segment


def test_exporter_freezes_checkpoint_identity_and_external_output_guards() -> None:
    source, _tree = _script_source()

    assert "e2be19c6fe1e843856fb339d13de8baed8d748f88558ba7bd3eaaa20b90ede21" in source
    assert "EXPECTED_KEY_COUNT = 63" in source
    assert 'parser.add_argument("--checkpoint", required=True, type=Path)' in source
    assert 'parser.add_argument("--output", required=True, type=Path)' in source
    assert "output path must be outside the NanoLoop-Agent repository" in source
    assert "refusing to overwrite existing output" in source
    assert "checkpoint SHA-256 differs" in source
    assert "torchscript_sha256" in source
    assert "C:\\" not in source
    assert "D:\\" not in source
    assert "/share/" not in source


def test_exporter_matches_confirmed_agglomerated_architecture() -> None:
    source, tree = _script_source()
    residual = _class_source(source, tree, "ResidualBlock")
    down = _class_source(source, tree, "DownBlock")
    up = _class_source(source, tree, "UpBlock")
    model = _class_source(source, tree, "AgglomeratedUNet")

    assert residual.count("bias=False") == 3
    assert residual.count("nn.GroupNorm(group_count(out_channels), out_channels)") == 2
    assert "self.activation = nn.SiLU(inplace=True)" in residual
    assert "stride=2" in down and "bias=False" in down
    assert 'mode="bilinear", align_corners=False' in up
    assert "self.encoder1 = ResidualBlock(1, 32)" in model
    assert "self.encoder2 = DownBlock(32, 64)" in model
    assert "self.encoder3 = DownBlock(64, 128)" in model
    assert "self.encoder4 = DownBlock(128, 256)" in model
    assert "ResidualBlock(256, 256, dilation=2)" in model
    assert "ResidualBlock(256, 256, dilation=4)" in model
    assert "self.decoder3 = UpBlock(256, 128, 128)" in model
    assert "self.decoder2 = UpBlock(128, 64, 64)" in model
    assert "self.decoder1 = UpBlock(64, 32, 32)" in model
    assert "self.output = nn.Conv2d(32, 1, 1)" in model
    assert "63-key checkpoint include output.bias" in model


def test_exporter_scripts_reloads_and_compares_384_outputs() -> None:
    source, _tree = _script_source()

    assert "model.load_state_dict(checkpoint_state, strict=True)" in source
    assert "model.eval()" in source
    assert "torch.jit.script(model)" in source
    assert 'torch.jit.load(str(temporary_output), map_location="cpu")' in source
    assert "MAX_ABS_ERROR = 1e-6" in source
    assert "_publish_verified_export(temporary_output, output)" in source
    assert "temporary_output.unlink(missing_ok=True)" in source
    assert "steps=PATCH_SIZE * PATCH_SIZE" in source
    assert "reshape(1, 1, PATCH_SIZE, PATCH_SIZE)" in source
    assert "eager_logits" in source and "scripted_logits" in source
    assert "eager_probability" in source and "scripted_probability" in source
    assert '"max_abs_error"' in source
