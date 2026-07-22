from pathlib import Path

import streamlit as st

from profile import PROFILE
from theme import apply_theme, hero


PROJECT_ROOT = Path(__file__).resolve().parent


def home_page() -> None:
    """두 모델을 소개하고 각 예측 페이지로 연결하는 첫 화면입니다."""
    apply_theme()
    hero(PROFILE["title"], PROFILE["subtitle"], PROFILE["name"])

    classification_model = PROJECT_ROOT / "mnist_cnn.pt"
    regression_model = PROJECT_ROOT / "bike_reg.pt"
    ready_count = sum(
        path.exists() for path in [classification_model, regression_model]
    )

    st.markdown("### 프로젝트 현황")
    m1, m2, m3 = st.columns(3)
    m1.metric("완성한 모델", f"{ready_count}/2")
    m2.metric("문제 유형", "분류 + 회귀")
    m3.metric("최종 데모", "1 showcase")

    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown("### ✍️ 이미지 분류")
            st.caption("MNIST · CNN · Accuracy")
            st.write("사진이나 직접 그린 숫자를 28×28 tensor로 바꿔 0~9의 확률을 예측합니다.")
            st.success(
                "체크포인트 준비 완료"
                if classification_model.exists()
                else "mnist_cnn.pt가 필요합니다"
            )
            st.page_link(
                "1_분류_MNIST.py",
                label="분류 모델 열기 →",
                use_container_width=True,
            )

    with right:
        with st.container(border=True):
            st.markdown("### 🚲 수요 회귀")
            st.caption("Seoul Bike · MLP · MAE")
            st.write("시간과 날씨를 train 통계로 표준화해 미래 시간당 대여량을 예측합니다.")
            st.success(
                "체크포인트 준비 완료"
                if regression_model.exists()
                else "bike_reg.pt가 필요합니다"
            )
            st.page_link(
                "2_회귀_자전거.py",
                label="회귀 모델 열기 →",
                use_container_width=True,
            )

    st.markdown("### 내가 설명할 수 있어야 하는 것")
    q1, q2, q3 = st.columns(3)
    with q1:
        with st.container(border=True):
            st.markdown("**01 · 분류의 한계**")
            st.write(PROFILE["classification_insight"])
    with q2:
        with st.container(border=True):
            st.markdown("**02 · 회귀의 오차**")
            st.write(PROFILE["regression_insight"])
    with q3:
        with st.container(border=True):
            st.markdown("**03 · 다음 실험**")
            st.write(PROFILE["next_step"])


st.set_page_config(
    page_title="나의 딥러닝 모델 랩",
    page_icon="🧪",
    layout="wide",
)

navigation = st.navigation(
    [
        st.Page(home_page, title="Home", icon="🏠", default=True),
        st.Page("1_분류_MNIST.py", title="분류 MNIST", icon="✍️"),
        st.Page("2_회귀_자전거.py", title="회귀 자전거", icon="🚲"),
    ]
)
navigation.run()
