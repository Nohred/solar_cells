import streamlit as st

from audit import record_event
from metrics import (
	IMAGES_PROCESSED,
	INFERENCE_LATENCY,
	INFERENCE_REQUESTS,
	set_active_model,
)
from triton_client import TritonClient, TritonClientError


MAX_IMAGES = 5


def main() -> None:
	st.set_page_config(page_title="Solar Cells Inspector", page_icon="☀️", layout="wide")
	st.title("Solar Cells Inspector")
	st.caption("Clasificación de defectos mediante el modelo activo en Triton")

	uploads = st.file_uploader(
		"Sube hasta 5 imágenes",
		type=["jpg", "jpeg", "png", "bmp"],
		accept_multiple_files=True,
	)
	if len(uploads) > MAX_IMAGES:
		st.error("El lote no puede superar 5 imágenes.")
		return
	if not uploads:
		st.info("Selecciona una o más imágenes para comenzar.")
		return

	if st.button("Analizar lote", type="primary"):
		client = TritonClient()
		image_bytes = [upload.getvalue() for upload in uploads]
		try:
			with st.spinner("Consultando Triton..."):
				model_name, predictions, latency = client.predict(image_bytes)
			set_active_model(model_name)
			INFERENCE_REQUESTS.labels("success").inc()
			IMAGES_PROCESSED.inc(len(uploads))
			INFERENCE_LATENCY.observe(latency)
			record_event(
				"inference",
				batch_size=len(uploads),
				model=model_name,
				latency_seconds=round(latency, 6),
				status="success",
			)
			st.success(f"Lote procesado en {latency:.3f} s")
			columns = st.columns(min(len(uploads), 3))
			for index, (upload, prediction) in enumerate(zip(uploads, predictions)):
				with columns[index % len(columns)]:
					st.image(upload, caption=upload.name, use_column_width=True)
					st.metric(prediction.class_name, f"{prediction.confidence:.1%}")
		except (TritonClientError, ValueError, OSError) as error:
			INFERENCE_REQUESTS.labels("error").inc()
			record_event("inference", batch_size=len(uploads), status="error", error=str(error))
			st.error(str(error))


if __name__ == "__main__":
	main()
