#!/usr/bin/env python3
"""Run inference with a prompt-baked YOLOE OpenVINO model."""

from __future__ import annotations

import argparse
from pathlib import Path


# 命令行示例:
#   python3 infer_openvino.py --model yoloe-26l-seg_openvino_model --source assets/test.jpg
# 这里的 source 是待检测图片/目录；提示词或参考图已经在 export_openvino.py 中烘焙。


def main() -> None:
    parser = argparse.ArgumentParser(description="使用已烘焙提示的 OpenVINO YOLOE 模型推理")
    parser.add_argument("--model", type=Path, required=True, help="model.xml 或 OpenVINO 模型目录")
    parser.add_argument("--source", required=True, help="输入图片、目录、视频或摄像头编号")
    parser.add_argument("--output", type=Path, default=Path("runs/openvino"), help="结果保存目录")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    parser.add_argument("--device", default="CPU", help="OpenVINO 设备，例如 CPU、GPU")
    args = parser.parse_args()

    model_path = args.model
    if model_path.is_file() and model_path.suffix.lower() == ".xml":
        model_path = model_path.parent
    if model_path.is_dir():
        xml_files = sorted(model_path.glob("*.xml"))
        if len(xml_files) != 1:
            raise FileNotFoundError(f"模型目录中应有且只有一个 .xml 文件: {model_path}")
    else:
        raise FileNotFoundError(f"找不到 OpenVINO 模型目录: {model_path}")
    if not model_path.name.endswith("_openvino_model"):
        raise ValueError(
            f"OpenVINO 模型目录名必须以 '_openvino_model' 结尾（当前为 {model_path.name}）；"
            "请使用 export_openvino.py 重新导出"
        )
    if not 0 <= args.conf <= 1:
        raise ValueError("--conf 必须位于 0 到 1 之间")

    from ultralytics import YOLO

    model = YOLO(str(model_path), task="segment")
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=True,
        project=str(args.output.parent),
        name=args.output.name,
        exist_ok=True,
        verbose=True,
    )
    print(f"处理完成，共 {len(results)} 张/帧；结果目录: {args.output}")


if __name__ == "__main__":
    main()
