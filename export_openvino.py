#!/usr/bin/env python3
"""Bake YOLOE prompts into a model and export it to OpenVINO.

The exported model has a fixed vocabulary.  To change the prompts, run this
script again and write to a different output directory.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# 也可以直接修改这里，然后运行: python3 export_openvino.py
# 文字提示写在 PROMPTS 中，每一项对应一个类别，例如 ["person", "red cup"]。
# 参考图写在 REFERENCE_IMAGE 中；若同时设置 REFERENCE_IMAGE，将使用参考图模式。
# 参考框格式为 (x1, y1, x2, y2)，坐标是参考图上的像素坐标；None 表示整张图。
# 命令行示例:
#   python3 export_openvino.py --prompt person car "red cup"
#   python3 export_openvino.py --reference-image assets/cup.jpg --reference-box 20,30,220,260
#   python3 export_openvino.py --prompt person car --precision fp16
# ---------------------------------------------------------------------------
MODEL = Path(__file__).with_name("yoloe-26l-seg.pt")
PROMPTS = ["person", "car"]
REFERENCE_IMAGE: str | None = None
REFERENCE_BOX: tuple[float, float, float, float] | None = None


def parse_box(value: str) -> tuple[float, float, float, float]:
    """Parse x1,y1,x2,y2 from a CLI argument."""
    try:
        parts = tuple(float(x.strip()) for x in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("框必须是 x1,y1,x2,y2，例如 10,20,200,240") from exc
    if len(parts) != 4 or parts[2] <= parts[0] or parts[3] <= parts[1]:
        raise argparse.ArgumentTypeError("框必须满足 x2>x1 且 y2>y1")
    return parts  # type: ignore[return-value]


def parse_prompts(values: list[str]) -> list[str]:
    """Accept both repeated arguments and comma-separated prompts."""
    prompts = [item.strip() for value in values for item in value.split(",") if item.strip()]
    if not prompts:
        raise ValueError("至少需要一个非空文字提示")
    return prompts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将 YOLOE 文字提示或参考图提示烘焙后导出 OpenVINO")
    parser.add_argument("--model", type=Path, default=MODEL, help="YOLOE .pt 权重路径")
    parser.add_argument("--prompt", nargs="+", help="文字提示，可重复或用逗号分隔")
    parser.add_argument("--reference-image", type=Path, help="参考图路径；与 --prompt 二选一")
    parser.add_argument("--reference-box", type=parse_box, help="参考框 x1,y1,x2,y2；省略则使用整图")
    parser.add_argument("--output", type=Path, help="导出目录，默认是 <model>_openvino_model")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有的自定义导出目录")
    parser.add_argument("--imgsz", type=int, default=640, help="导出输入尺寸，默认 640")
    parser.add_argument("--device", default="cpu", help="PyTorch 导出设备，例如 cpu 或 0")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32", help="OpenVINO 权重精度")
    return parser


def bake_visual_prompt(model, image_path: Path, box: tuple[float, float, float, float] | None, imgsz: int) -> None:
    """Extract a visual prompt embedding and install it on the YOLOE model."""
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    with Image.open(image_path) as image:
        width, height = image.size
    reference_box = box or (0.0, 0.0, float(width), float(height))
    visual_prompts = {
        "bboxes": np.asarray([reference_box], dtype=np.float32),
        "cls": np.asarray([0], dtype=np.int64),
    }
    # Calling predict with refer_image makes YOLOE compute the visual prompt
    # embedding and set it on model.model before the OpenVINO export.
    model.predict(
        source=str(image_path),
        visual_prompts=visual_prompts,
        refer_image=str(image_path),
        predictor=YOLOEVPSegPredictor,
        imgsz=imgsz,
        save=False,
        verbose=False,
    )
    # YOLOE assigns the temporary name ``object0`` while extracting a visual
    # prompt. Keep the embedding but expose a stable, readable output label.
    model.set_classes(["reference_object"], model.model.pe)


def main() -> None:
    args = build_parser().parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"找不到模型文件: {args.model}")
    if args.imgsz <= 0:
        raise ValueError("--imgsz 必须是正整数")
    if args.prompt and args.reference_image:
        raise ValueError("--prompt 和 --reference-image 只能选择一个")
    if args.reference_box and not args.reference_image:
        raise ValueError("--reference-box 只能与 --reference-image 一起使用")

    prompts = parse_prompts(args.prompt) if args.prompt else PROMPTS
    reference_image = args.reference_image or (Path(REFERENCE_IMAGE) if REFERENCE_IMAGE else None)
    reference_box = args.reference_box or REFERENCE_BOX
    if reference_image is not None:
        if args.prompt:
            raise ValueError("参考图模式不能同时指定文字提示")
        if not reference_image.is_file():
            raise FileNotFoundError(f"找不到参考图: {reference_image}")
    elif not prompts:
        raise ValueError("请设置 PROMPTS，或提供 --reference-image")

    from ultralytics import YOLOE

    print(f"加载模型: {args.model}")
    model = YOLOE(str(args.model), verbose=False)
    if reference_image is not None:
        print(f"烘焙参考图: {reference_image} (box={reference_box or 'full-image'})")
        bake_visual_prompt(model, reference_image, reference_box, args.imgsz)
        baked_names = ["reference_object"]
    else:
        print(f"烘焙文字提示: {prompts}")
        model.set_classes(prompts)
        baked_names = prompts

    output = model.export(
        format="openvino",
        imgsz=args.imgsz,
        device=args.device,
        quantize=16 if args.precision == "fp16" else None,
        dynamic=False,
        nms=False,
    )
    export_dir = Path(str(output))
    if export_dir.is_file():
        export_dir = export_dir.parent
    if args.output:
        destination = args.output.resolve()
        if not destination.name.endswith("_openvino_model"):
            destination = destination.with_name(destination.name + "_openvino_model")
        source = export_dir.resolve()
        if source != destination:
            if destination.exists():
                if not args.overwrite:
                    raise FileExistsError(f"导出目录已存在: {destination}；需要覆盖时加 --overwrite")
                shutil.rmtree(destination)
            shutil.move(str(source), str(destination))
            export_dir = destination

    # Save the names and embeddings beside the IR so the baked configuration
    # is inspectable and can be loaded with YOLOE.load_prompt_embeddings().
    embedding_file = export_dir / "prompt_embeddings.npz"
    model.save_prompt_embeddings(embedding_file)
    (export_dir / "prompts.txt").write_text("\n".join(baked_names) + "\n", encoding="utf-8")
    xml_files = sorted(export_dir.glob("*.xml"))
    if len(xml_files) != 1:
        raise RuntimeError(f"导出目录中应有且只有一个 XML 文件，实际找到: {xml_files}")
    print(f"OpenVINO 模型目录: {export_dir}")
    print(f"OpenVINO XML: {xml_files[0]}")
    print(f"提示 embedding: {embedding_file}")


if __name__ == "__main__":
    main()
