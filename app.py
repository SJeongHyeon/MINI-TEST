# MNIST 손글씨 분류기 — 학생용 핵심 스캐폴딩
# 실행: python3.11 -m streamlit run app.py
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "mnist_cnn.pt"
MY_NAME = "서정현"


class ScaffoldIncomplete(RuntimeError):
    """학생이 아직 완성하지 않은 핵심 구간을 화면에 친절하게 알립니다."""


class CNN(nn.Module):
    def __init__(self, conv1=32, conv2=64, hidden=128, dropout=0.3):
        super().__init__()

        # TODO 1-A — 노트북 [Step 3]의 CNN과 같은 feature extractor를 작성하세요.
        # 확인할 shape 흐름:
        # (N, 1, 28, 28) → 첫 합성곱/풀링 → (N, conv1, 14, 14)
        #                    → 둘째 합성곱/풀링 → (N, conv2, 7, 7)
        # 질문: ReLU는 어느 연산 뒤에 있어야 하고, MaxPool2d는 공간 크기를 어떻게 바꾸나요?
        self.features = nn.Sequential(
            nn.Conv2d(1, conv1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(conv1, conv2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # TODO 1-B — 7×7 feature map을 숫자 10개의 logit으로 바꾸는 분류부를 작성하세요.
        # 체크포인트의 conv2·hidden 값이 달라도 동작해야 하므로 숫자를 고정하지 마세요.
        # 마지막 층에 Softmax를 작성하지 않습니다. 학습한 모델은 logit을 출력합니다.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv2 * 7 * 7, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 10),
        )


    def forward(self, x):
        
        # 먼저 중간 shape을 print해 보면 두 블록이 왜 나뉘는지 이해하기 쉽습니다.
        return self.classifier(self.features(x))
     


@st.cache_resource
def load_model():
    """체크포인트로 학습 때와 같은 CNN을 복원합니다."""
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)

   
    # ① checkpoint["model_config"]로 CNN 구조를 다시 만들기
    # ② checkpoint["state_dict"]의 학습된 가중치 복원하기
    # ③ 추론 모드로 전환하기
    # 반환 계약: (model, checkpoint)
    model = CNN(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def make_mnist_canvas(image: Image.Image) -> np.ndarray:
    """화면용 보조 코드: 다양한 사진을 MNIST와 비슷한 28×28 캔버스로 정렬합니다."""
    rgba = image.convert("RGBA")
    white = Image.new("RGBA", rgba.size, "white")
    gray = ImageOps.grayscale(Image.alpha_composite(white, rgba).convert("RGB"))
    arr = np.array(gray, dtype=np.uint8)

    if float(arr.mean()) > 127:
        arr = 255 - arr
    arr[arr < 30] = 0

    mask = arr > 0
    if mask.any():
        ys, xs = np.where(mask)
        digit = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    else:
        digit = arr

    height, width = digit.shape
    side = max(height, width, 1)
    square = np.zeros((side, side), dtype=np.uint8)
    top, left = (side - height) // 2, (side - width) // 2
    square[top:top + height, left:left + width] = digit

    resized = Image.fromarray(square).resize((20, 20), Image.Resampling.LANCZOS)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    canvas[4:24, 4:24] = np.asarray(resized, dtype=np.uint8)
    return canvas


def preprocess_image(image: Image.Image) -> tuple[torch.Tensor, np.ndarray]:
    """업로드 이미지 → 모델 입력 tensor와 화면용 28×28 배열."""
    canvas = make_mnist_canvas(image)

   
    # 최종 확인: shape (1, 1, 28, 28), dtype float32, 값 범위 0~1
    # 첫 번째 1은 배치, 두 번째 1은 흑백 채널입니다.
    normalized = canvas.astype("float32") / 255.0
    x = torch.from_numpy(normalized).reshape(1, 1, 28, 28)

    assert x.shape == (1, 1, 28, 28)
    assert x.dtype == torch.float32
    assert 0.0 <= x.min().item() <= x.max().item() <= 1.0

    return x, canvas


def predict_probabilities(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    """모델 logit을 숫자 0~9의 확률 배열로 변환합니다."""
    
    # 반환 계약: shape (10,)의 NumPy 배열
    # 질문: argmax만 사용하면 무엇을 잃고, softmax를 두 번 적용하면 왜 잘못될까요?
    with torch.inference_mode():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)

    return probabilities.squeeze(0).cpu().numpy()
   


def apply_page_style() -> None:
    """Part I showcase의 카드형 레이아웃을 이 과제의 실험 노트 콘셉트로 축소했습니다."""
    st.markdown(
        """
        <style>
        :root { --ink:#17324d; --paper:#f7f1e5; --coral:#e76f51; --mint:#2a9d8f; }
        .stApp { background: linear-gradient(135deg, #fbf8f1 0%, #eef6f3 100%); color:var(--ink); }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] { background:#fffaf0; border-right:1px solid #decfb8; }
        .mp-hero { padding:1.6rem 1.8rem; border:1px solid #d8c8ad; border-radius:18px;
          background:rgba(255,255,255,.82); box-shadow:0 12px 30px rgba(23,50,77,.08); margin-bottom:1.2rem; }
        .mp-kicker { color:var(--coral); font-weight:800; letter-spacing:.08em; font-size:.78rem; }
        .mp-title { color:var(--ink); font-size:clamp(1.8rem,4vw,3rem); line-height:1.08; margin:.35rem 0; }
        .mp-sub { color:#506579; margin:0; max-width:760px; }
        .mp-step { border-left:4px solid var(--mint); padding:.35rem .8rem; color:#40566b; }
        [data-testid="stVerticalBlockBorderWrapper"] { border-radius:16px; background:rgba(255,255,255,.68); }
        .stButton>button { border-radius:12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <section class="mp-hero">
          <div class="mp-kicker">MODEL LAB · CLASSIFICATION</div>
          <h1 class="mp-title">손글씨 한 장이<br>열 개의 확률이 되기까지</h1>
          <p class="mp-sub">직접 학습한 CNN의 체크포인트를 복원하고, 사진을 28×28 tensor로 바꿔 예측합니다.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="MNIST 분류 모델 랩", page_icon="✍️", layout="wide")
    apply_page_style()
    render_header()

    with st.expander("🧭 이 앱에서 내가 직접 완성할 핵심 4단계", expanded=False):
        st.markdown(
            """
            1. `CNN` — 학습 때와 같은 모델 구조 재구성
            2. `load_model()` — 구조와 `state_dict` 복원, 추론 모드 전환
            3. `preprocess_image()` — 사진을 `(1,1,28,28)` 입력으로 변환
            4. `predict_probabilities()` — logit을 10개 확률로 변환

            화면 구성은 제공됩니다. 네 함수가 연결되어야 실제 예측이 시작됩니다.
            """
        )

    if not MODEL_PATH.exists():
        st.error("`mnist_cnn.pt`가 없습니다. 과제 노트북 [Step 7]을 실행해 이 폴더에 생성하세요.")
        st.stop()

    try:
        model, checkpoint = load_model()
    except ScaffoldIncomplete as exc:
        st.warning(str(exc))
        st.info("`app.py`의 TODO 1→2 순서로 완성한 뒤 파일을 저장하면 화면이 자동으로 다시 실행됩니다.")
        st.stop()
    except (KeyError, RuntimeError, TypeError) as exc:
        st.error("학습 때의 모델 구조와 앱의 구조가 일치하지 않습니다. TODO 1·2와 checkpoint key를 확인하세요.")
        st.code(str(exc))
        st.stop()

    metrics = checkpoint["metrics"]
    train_config = checkpoint["training_config"]
    model_config = checkpoint["model_config"]
    st.sidebar.header("MODEL PASSPORT")
    st.sidebar.metric("Validation 정확도", f"{metrics['val_acc']:.4f}")
    st.sidebar.metric("최종 Test 정확도", f"{metrics['test_acc']:.4f}")
    st.sidebar.metric("파라미터", f"{checkpoint['n_params']:,}")
    st.sidebar.caption(f"epochs {train_config['epochs']} · lr {train_config['lr']}")
    st.sidebar.caption(f"conv {model_config['conv1']}/{model_config['conv2']} · hidden {model_config['hidden']}")
    st.sidebar.caption(f"제작: {MY_NAME}")

    input_col, result_col = st.columns([1, 1], gap="large")
    image = x = preview = None
    image = None
    x = None
    preview = None

    with input_col:
        st.subheader("01 · 입력 이미지")

        method = st.radio(
            "입력 방식",
            ["직접 그리기", "이미지 업로드", "카메라 촬영"],
            horizontal=True,
        )

        # ─────────────────────────────
        # ① 직접 그리기
        # ─────────────────────────────
        if method == "직접 그리기":
            st.caption("검은 화면에 마우스로 흰색 숫자를 크게 그려주세요.")

            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 1)",
                stroke_width=18,
                stroke_color="#FFFFFF",
                background_color="#000000",
                update_streamlit=True,
                height=280,
                width=280,
                drawing_mode="freedraw",
                display_toolbar=True,
                key="mnist_draw_canvas",
            )

            if canvas_result.image_data is not None:
                canvas_array = np.asarray(
                    canvas_result.image_data
                ).astype(np.uint8)

                # 빈 캔버스가 아닐 때만 이미지로 사용합니다.
                rgb_pixels = canvas_array[:, :, :3]

                if rgb_pixels.max() > 20:
                    image = Image.fromarray(canvas_array)

        # ─────────────────────────────
        # ② 이미지 업로드
        # ─────────────────────────────
        elif method == "이미지 업로드":
            source = st.file_uploader(
                "PNG 또는 JPG",
                type=["png", "jpg", "jpeg"],
            )

            if source is not None:
                try:
                    image = Image.open(source)
                except Exception:
                    st.error(
                        "이미지를 열 수 없습니다. "
                        "정상적인 PNG/JPG 파일인지 확인하세요."
                    )
                    st.stop()

        # ─────────────────────────────
        # ③ 카메라 촬영
        # ─────────────────────────────
        else:
            source = st.camera_input(
                "종이의 숫자를 크게 촬영하세요"
            )

            if source is not None:
                try:
                    image = Image.open(source)
                except Exception:
                    st.error(
                        "카메라 이미지를 열 수 없습니다."
                    )
                    st.stop()

        # ─────────────────────────────
        # 선택된 이미지를 모델 입력으로 변환
        # ─────────────────────────────
        if image is None:
            st.info(
                "숫자를 그리거나 이미지를 입력하면 "
                "오른쪽에 예측 결과가 표시됩니다."
            )

        else:
            try:
                x, preview = preprocess_image(image)

            except ScaffoldIncomplete as exc:
                st.warning(str(exc))
                st.stop()

            except Exception as exc:
                st.error("이미지 전처리 중 오류가 발생했습니다.")
                st.code(str(exc))
                st.stop()

            # 업로드·카메라는 원본과 전처리 결과를 비교합니다.
            if method != "직접 그리기":
                before, after = st.columns(2)

                before.image(
                    image,
                    caption="원본",
                    width="stretch",
                )

                after.image(
                    preview,
                    caption="모델 입력 28×28",
                    clamp=True,
                    width="stretch",
                )

            # 직접 그린 경우에는 모델이 실제로 보는 모습만 표시합니다.
            else:
                st.image(
                    preview,
                    caption="CNN이 실제로 보는 28×28 이미지",
                    clamp=True,
                    width=180,
                )

    # ─────────────────────────────────
    # 오른쪽 예측 결과
    # ─────────────────────────────────
    with result_col:
        st.subheader("02 · 모델의 판단")

        if x is None:
            with st.container(border=True):
                st.markdown("### 결과 대기 중")
                st.caption(
                    "왼쪽에서 숫자를 그리거나 이미지를 입력하면 "
                    "예측 결과가 표시됩니다."
                )

        else:
            try:
                probabilities = predict_probabilities(
                    model,
                    x,
                )

            except ScaffoldIncomplete as exc:
                st.warning(str(exc))
                st.stop()

            except Exception as exc:
                st.error("모델 예측 중 오류가 발생했습니다.")
                st.code(str(exc))
                st.stop()

            prediction = int(probabilities.argmax())
            confidence = float(probabilities[prediction])

            with st.container(border=True):
                st.metric(
                    "예측 숫자",
                    prediction,
                )

                st.metric(
                    "신뢰도",
                    f"{confidence * 100:.2f}%",
                )

                # 숫자 0~9의 전체 확률
                chart_data = {
                    str(number): float(probability)
                    for number, probability
                    in enumerate(probabilities)
                }

                st.bar_chart(chart_data)

                top3 = probabilities.argsort()[-3:][::-1]

                st.caption(
                    "Top 3 · "
                    + " · ".join(
                        f"{int(number)}: "
                        f"{probabilities[number]:.1%}"
                        for number in top3
                    )
                )

            st.caption(
                "직접 그린 숫자는 MNIST 학습 이미지와 "
                "굵기·위치·모양이 달라 예측이 틀릴 수 있습니다."
            )


if __name__ == "__main__":
    main()
