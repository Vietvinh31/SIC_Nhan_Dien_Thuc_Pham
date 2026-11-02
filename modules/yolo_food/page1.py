import json
from pathlib import Path

import streamlit as st
from PIL import Image
from model_loader import load_model


BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "food_items.json", "r", encoding="utf-8") as f:
    FOOD_ITEMS = json.load(f)


def text_detection(model, file):
    uploaded_image = Image.open(file).convert("RGB")

    results = model.predict(
        source=uploaded_image,
        conf=0.25,
        imgsz=640,
        device="cpu",
        verbose=False
    )

    result = results[0]
    detections = []

    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        class_name = result.names[class_id]

        detections.append({
            "name": class_name,
            "key": class_name.lower().strip(),
            "confidence": confidence
        })

    unique_foods = {}

    for detection in detections:
        key = detection["key"]

        if (
            key not in unique_foods
            or detection["confidence"] > unique_foods[key]["confidence"]
        ):
            unique_foods[key] = detection

    # YOLO trả về ảnh BGR, chuyển sang RGB
    plotted_image = Image.fromarray(
        result.plot()[..., ::-1]
    )

    return {
        "unique_foods": list(unique_foods.values()),
        "res_plotted": plotted_image
    }


def app():
    st.title("Food Recognition & Macro Calculator")

    model = load_model()

    file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if file is not None:
        st.image(file, caption="Original Image", width=600)

        if st.button("Detect Food", key="detect_food"):
            with st.spinner("AI is analyzing the image..."):
                result = text_detection(model, file)

            st.session_state["page1_unique_foods"] = result["unique_foods"]
            st.session_state["page1_res_plotted"] = result["res_plotted"]
            st.session_state["page1_processed"] = True

    if not st.session_state.get("page1_processed", False):
        return

    foods = st.session_state.get("page1_unique_foods", [])

    if not foods:
        st.warning("No food detected. Try another image.")
        return

    st.subheader("Detection Results")

    st.image(
        st.session_state["page1_res_plotted"],
        width=600
    )

    st.success(f"Detected {len(foods)} food categories")

    for food in foods:
        st.write(
            f"**{food['name']}** — "
            f"Confidence: {food['confidence']:.2%}"
        )

    st.subheader("Nutrition Calculator")

    total_protein = 0.0
    total_carbs = 0.0
    total_fat = 0.0
    total_calories = 0.0

    available_foods = []

    for food in foods:
        key = food["key"]

        if key not in FOOD_ITEMS:
            st.info(
                f"Nutrition data for '{food['name']}' "
                "is not available yet."
            )
            continue

        available_foods.append(food)

        st.number_input(
            f"Weight of {food['name']} (g)",
            min_value=0,
            value=100,
            step=10,
            key=f"weight_{key}"
        )

    if st.button("Calculate Macros", key="calculate_food"):
        if not available_foods:
            st.warning("No nutrition records available.")
            return

        for food in available_foods:
            key = food["key"]
            weight = st.session_state[f"weight_{key}"]
            data = FOOD_ITEMS[key]

            total_protein += data["protein"] * weight / 100
            total_carbs += data["carbs"] * weight / 100
            total_fat += data["fat"] * weight / 100
            total_calories += data["calories"] * weight / 100

        st.subheader("Macro Breakdown")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Calories", f"{total_calories:.0f} kcal")
        c2.metric("Protein", f"{total_protein:.2f} g")
        c3.metric("Carbs", f"{total_carbs:.2f} g")
        c4.metric("Fat", f"{total_fat:.2f} g")