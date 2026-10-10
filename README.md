# YOLOE 26L Seg -> OpenVINO

这个目录把已经下载的 `yoloe-26l-seg.pt` 做成可直接运行的 OpenVINO 实例分割模型。YOLOE 的文字提示或参考图只在导出阶段使用，导出的 OpenVINO IR 已经固定了类别 embedding。

## 安装

```bash
python3 -m pip install -r requirements.txt
```

首次使用文字提示时，MobileCLIP2 权重（约 242 MB）可能需要联网下载；模型文件应放在本目录，或通过 `--model` 指定其他路径。

## 文字提示烘焙

提示词可以写在 `export_openvino.py` 顶部的 `PROMPTS`，也可以在命令行传入。每个提示对应一个输出类别：

```bash
python3 export_openvino.py --prompt person car "red cup"
```

默认输出目录是 `yoloe-26l-seg_openvino_model/`。也可以显式指定：

```bash
python3 export_openvino.py --prompt person car --output artifacts/person_car
```

自定义目录会自动补上 `_openvino_model` 后缀，例如上面的实际目录是 `artifacts/person_car_openvino_model/`。
目录已存在时脚本会停止；确认要替换时加 `--overwrite`。

## 参考图烘焙

参考图路径写在 `export_openvino.py` 的 `REFERENCE_IMAGE`，或使用命令行。参考框使用原图像素坐标 `x1,y1,x2,y2`；省略参考框时会把整张图作为参考对象：

```bash
python3 export_openvino.py \
  --reference-image assets/cup.jpg \
  --reference-box 20,30,220,260 \
  --output artifacts/cup
```

参考图模式导出一个名为 `reference_object` 的类别。参考图和待检测图可以不同，待检测图在下一步的 `--source` 中指定。

## OpenVINO 推理

```bash
python3 infer_openvino.py \
  --model yoloe-26l-seg_openvino_model \
  --source assets/blue_block_0.jpg \
  --output runs/bb_output_0
```

推理结果会保存为带框和分割掩码的图片。可用 `--device GPU` 切换 OpenVINO 设备，用 `--conf 0.4` 调整阈值。

## 生成文件

导出目录包含一个 `.xml`、对应的 `.bin`、`metadata.yaml`，以及便于追溯的 `prompts.txt` 和 `prompt_embeddings.npz`。XML 文件名由 Ultralytics 根据权重名生成；推理脚本传入目录即可自动找到它。更换提示词或参考图时重新执行导出，并使用新的输出目录。
