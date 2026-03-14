from __future__ import annotations

import gradio as gr
import cv2

from app.core.config import settings
from app.services.model import YoloService
from app.services.video import process_video

yolo = YoloService(
    weights_path=settings.model_path,
    conf_thres=settings.conf_thres,
    iou_thres=settings.iou_thres,
    device=0,  # или None/"cpu"
)

CSS = """
/* фиксируем высоту видео, чтобы не раздувалось от разрешения */
#out_video video { max-height: 420px !important; height: 420px !important; width: 100% !important; object-fit: contain; }
#in_file { max-width: 100%; }
"""

def ui_detect_image(image):
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    dets = yolo.predict(bgr)

    out = bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            out,
            f"{d.class_name} {d.conf:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return rgb, [d.__dict__ for d in dets]


def ui_detect_video(video_file):
    if video_file is None:
        raise gr.Error("Видео не выбрано.")

    # gr.File обычно отдаёт объект с атрибутом .name (путь)
    video_path = getattr(video_file, "name", None) or str(video_file)

    artifacts = settings.artifacts_path()
    result = process_video(
        video_path=video_path,
        yolo=yolo,
        artifacts_dir=artifacts,
        sample_fps=settings.sample_fps,
        fire_consecutive=settings.fire_consecutive,
    )
    return result["video_out"], result["events"], {
        "max_conf": result["max_conf"],
        "frames_processed": result["frames_processed"],
        "processed_fps": result["processed_fps"],
    }


with gr.Blocks(title="Fire Early Detection", css=CSS) as demo:
    gr.Markdown("# Раннее обнаружение огня (YOLO)")

    with gr.Tab("Картинка"):
        inp_img = gr.Image(type="numpy", label="Загрузить изображение")
        out_img = gr.Image(type="numpy", label="Результат")
        out_json = gr.JSON(label="Detections")
        btn = gr.Button("Запустить детект")
        btn.click(ui_detect_image, inputs=[inp_img], outputs=[out_img, out_json])

    with gr.Tab("Видео"):
        # Важно: File вместо Video, чтобы Gradio не пытался проигрывать исходник
        inp_vid = gr.File(label="Загрузить видео (mp4/avi/...)", file_types=["video"], elem_id="in_file")
        out_vid = gr.Video(label="Размеченное видео", elem_id="out_video")
        out_events = gr.JSON(label="События (ALARM)")
        out_summary = gr.JSON(label="Сводка")
        btn2 = gr.Button("Обработать видео")
        btn2.click(ui_detect_video, inputs=[inp_vid], outputs=[out_vid, out_events, out_summary])

demo.launch(server_name="127.0.0.1", server_port=7860)