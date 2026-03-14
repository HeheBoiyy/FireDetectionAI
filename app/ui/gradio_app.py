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
:root{
  --bg:#0b1220;
  --panel:#0f1a2e;
  --card:#0e1930;
  --text:#e8eefc;
  --muted:rgba(232,238,252,.75);
  --accent:#ffcc66;
  --accent2:#7dd3fc;
  --ok:#34d399;
  --bad:#fb7185;
  --border:rgba(255,255,255,.10);
  --shadow: 0 10px 30px rgba(0,0,0,.35);
}

.gradio-container{
  background: radial-gradient(1200px 600px at 20% 0%, rgba(125,211,252,.18), transparent 60%),
              radial-gradient(900px 500px at 90% 20%, rgba(255,204,102,.14), transparent 55%),
              linear-gradient(180deg, var(--bg), #070b14);
  color: var(--text);
}

/* фиксируем высоту видео */
#out_video video { max-height: 420px !important; height: 420px !important; width: 100% !important; object-fit: contain; }
#in_file { max-width: 100%; }

.hero{
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 22px 22px;
  background: linear-gradient(180deg, rgba(15,26,46,.92), rgba(14,25,48,.86));
  box-shadow: var(--shadow);
}

.hero h1{
  font-size: 28px;
  margin: 0 0 8px 0;
  letter-spacing: .2px;
}

.hero p{
  margin: 8px 0 0 0;
  color: var(--muted);
  line-height: 1.55;
  font-size: 14.5px;
}

.badges{
  display:flex;
  gap:10px;
  flex-wrap:wrap;
  margin-top:14px;
}
.badge{
  display:inline-flex;
  align-items:center;
  gap:8px;
  padding:8px 10px;
  border-radius: 999px;
  background: rgba(255,255,255,.06);
  border: 1px solid var(--border);
  font-size: 12.5px;
  color: var(--text);
}

.cards{
  display:grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-top: 14px;
}
@media (max-width: 900px){
  .cards{ grid-template-columns: 1fr; }
}
.card{
  padding: 14px 14px;
  border-radius: 16px;
  background: rgba(255,255,255,.05);
  border: 1px solid var(--border);
}
.card h3{
  margin: 0 0 8px 0;
  font-size: 14px;
}
.card p{
  margin: 0;
  color: var(--muted);
  line-height: 1.45;
  font-size: 13.5px;
}

.ctaRow{
  display:flex;
  gap: 10px;
  align-items:center;
  margin-top: 16px;
  flex-wrap:wrap;
}
#btn_try button{
  background: linear-gradient(90deg, var(--accent), #ffd58a);
  color: #1b1b1b;
  font-weight: 700;
  border: 0;
}
.smallnote{
  color: var(--muted);
  font-size: 12.5px;
}

.resultbox{
  border: 1px dashed rgba(255,255,255,.18);
  background: rgba(255,255,255,.035);
  border-radius: 14px;
  padding: 12px 12px;
}
"""

def _summary_text_from_dets(dets) -> str:
    if not dets:
        return "Огонь не обнаружен. Попробуйте другое изображение/видео или лучшее освещение."
    # если классов несколько — делаем простую фразу
    max_conf = max(getattr(d, "conf", 0.0) for d in dets)
    return f"Обнаружены признаки огня. Уверенность: {max_conf:.2f}. (Результат демонстрационный)"

def ui_detect_image(image):
    if image is None:
        raise gr.Error("Изображение не выбрано.")

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
    summary = _summary_text_from_dets(dets)
    return rgb, summary, [d.__dict__ for d in dets]

def ui_detect_video(video_file):
    if video_file is None:
        raise gr.Error("Видео не выбрано.")

    video_path = getattr(video_file, "name", None) or str(video_file)

    artifacts = settings.artifacts_path()
    result = process_video(
        video_path=video_path,
        yolo=yolo,
        artifacts_dir=artifacts,
        sample_fps=settings.sample_fps,
        fire_consecutive=settings.fire_consecutive,
    )

    # human-friendly summary
    max_conf = float(result.get("max_conf", 0.0) or 0.0)
    summary = "Признаков огня не обнаружено." if max_conf <= 0 else f"На видео обнаружены признаки огня. Макс. уверенность: {max_conf:.2f}."
    return result["video_out"], summary, result["events"], {
        "max_conf": result["max_conf"],
        "frames_processed": result["frames_processed"],
        "processed_fps": result["processed_fps"],
    }

with gr.Blocks(title="Fire Early Detection", css=CSS) as demo:
    with gr.Tab("Главная"):
        gr.Markdown(
            """
<div class="hero">
  <h1>Раннее обнаружение огня на изображениях и видео</h1>
  <p>
    Это демонстрационный прототип на базе нейросети, который помогает автоматически находить признаки огня.
    Идея — сократить время реакции: оператору не нужно просматривать всё вручную, система выделяет подозрительные кадры.
  </p>

  <div class="badges">
    <div class="badge">Фото и видео</div>
    <div class="badge">Быстрый результат</div>
    <div class="badge">Понятный вывод</div>
  </div>

  <div class="cards">
    <div class="card">
      <h3>Зачем это нужно</h3>
      <p>Чтобы быстрее замечать опасные ситуации и снижать нагрузку на человека при мониторинге.</p>
    </div>
    <div class="card">
      <h3>Как это работает</h3>
      <p>Загрузите файл → модель анализирует → система показывает размеченный результат и краткое пояснение.</p>
    </div>
    <div class="card">
      <h3>Важно</h3>
      <p>Это прототип. Точность зависит от качества видео, освещения и условий съёмки.</p>
    </div>
  </div>

  <div class="ctaRow">
    <div id="btn_try"></div>
    <div class="smallnote">Нажмите “Попробовать”, чтобы открыть демо ниже.</div>
  </div>
</div>
"""
        )

        btn_try = gr.Button("Попробовать демо", elem_id="btn_try")
        btn_hide = gr.Button("Скрыть демо", visible=False)

        with gr.Column(visible=False) as demo_panel:
            gr.Markdown("## Демо детекции")
            gr.Markdown("<div class='smallnote'>Выберите фото или видео, нажмите кнопку и получите результат.</div>")

            with gr.Tabs():
                with gr.TabItem("Фото"):
                    inp_img = gr.Image(type="numpy", label="Загрузить изображение")
                    btn = gr.Button("Запустить")
                    out_img = gr.Image(type="numpy", label="Результат", show_label=True)
                    out_text = gr.Markdown(elem_classes=["resultbox"])
                    with gr.Accordion("Технические детали (для разработчика)", open=False):
                        out_json = gr.JSON(label="Detections (raw)")
                    btn.click(ui_detect_image, inputs=[inp_img], outputs=[out_img, out_text, out_json])

                with gr.TabItem("Видео"):
                    inp_vid = gr.File(label="Загрузить видео (mp4/avi/...)", file_types=["video"], elem_id="in_file")
                    btn2 = gr.Button("Обработать")
                    out_vid = gr.Video(label="Размеченное видео", elem_id="out_video")
                    out_text_v = gr.Markdown(elem_classes=["resultbox"])
                    with gr.Accordion("Технические детали (для разработчика)", open=False):
                        out_events = gr.JSON(label="События (ALARM)")
                        out_summary = gr.JSON(label="Сводка (raw)")
                    btn2.click(ui_detect_video, inputs=[inp_vid], outputs=[out_vid, out_text_v, out_events, out_summary])

        # Показать/скрыть демо без переключения вкладок
        btn_try.click(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)),
            inputs=None,
            outputs=[demo_panel, btn_try, btn_hide],
        )
        btn_hide.click(
            fn=lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)),
            inputs=None,
            outputs=[demo_panel, btn_try, btn_hide],
        )

demo.launch(server_name="127.0.0.1", server_port=7860)