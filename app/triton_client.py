from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EmotionPrediction:
    label: str
    score: float
    scores: dict[str, float]


def _load_http_client_module():
    try:
        import tritonclient.http as httpclient
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'tritonclient'. Install app dependencies with "
            "`python -m pip install -r app\\requirements.txt`."
        ) from exc

    return httpclient


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def infer_logits(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    *,
    triton_url: str = "localhost:8000",
    model_name: str = "emotion_phobert",
    timeout_s: float = 30.0,
) -> np.ndarray:
    httpclient = _load_http_client_module()
    client = httpclient.InferenceServerClient(url=triton_url)

    triton_inputs = [
        httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
        httpclient.InferInput("attention_mask", attention_mask.shape, "INT64"),
    ]
    triton_inputs[0].set_data_from_numpy(input_ids.astype(np.int64, copy=False))
    triton_inputs[1].set_data_from_numpy(attention_mask.astype(np.int64, copy=False))

    outputs = [httpclient.InferRequestedOutput("logits")]
    result = client.infer(
        model_name=model_name,
        inputs=triton_inputs,
        outputs=outputs,
        timeout=int(timeout_s * 1_000_000),
    )

    logits = result.as_numpy("logits")
    if logits is None:
        raise RuntimeError("Triton response does not contain output 'logits'.")

    return logits


def predict_from_logits(logits: np.ndarray, labels: dict[int, str]) -> EmotionPrediction:
    probabilities = softmax(logits)
    predicted_id = int(np.argmax(probabilities[0]))

    scores = {
        labels[label_id]: float(probabilities[0, label_id])
        for label_id in sorted(labels)
    }

    return EmotionPrediction(
        label=labels[predicted_id],
        score=float(probabilities[0, predicted_id]),
        scores=scores,
    )


def check_model_ready(
    *,
    triton_url: str = "localhost:8000",
    model_name: str = "emotion_phobert",
) -> bool:
    httpclient = _load_http_client_module()
    client = httpclient.InferenceServerClient(url=triton_url)
    return client.is_server_ready() and client.is_model_ready(model_name)
