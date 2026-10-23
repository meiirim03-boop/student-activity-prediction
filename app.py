from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Оқу белсенділігін болжау", layout="centered")
st.title("📊 Келесі аптадағы оқу белсенділігін болжау")
st.write("200 білім алушының деректері негізінде оқытылған модель")

@st.cache_resource
def load_model():
    path = Path(__file__).parent / "research_model.joblib"
    return joblib.load(path)

bundle = load_model()
model = bundle["model"]
features = bundle["features"]

with st.form("prediction_form"):
    active_w1 = st.slider("1-аптадағы белсенді күндер саны", 0, 7, 3)
    active_w2 = st.slider("2-аптадағы белсенді күндер саны", 0, 7, 4)
    tasks_comp = st.number_input("Орындалған жұмыстар", min_value=0, max_value=20, value=5)
    resources = st.number_input("Қаралған ресурстар", min_value=0, max_value=100, value=15)
    assigned = st.number_input("Берілген тапсырмалар", min_value=1, max_value=20, value=8)
    
    clicked = st.form_submit_button("Болжам жасау")

if clicked:
    if tasks_comp > assigned:
        st.error("Орындалған жұмыстар берілген тапсырмалардан аспауы тиіс.")
    else:
        completion_pct = round((tasks_comp / assigned) * 100, 1)
        row = pd.DataFrame([{
            "active_days_w1": active_w1,
            "active_days_w2": active_w2,
            "tasks_completed": tasks_comp,
            "resources_viewed": resources,
            "task_completion_pct": completion_pct
        }])
        
        raw_pred = model.predict(row[features])[0]
        clipped_pred = int(np.clip(np.round(raw_pred), 0, 7))
        st.metric("Келесі аптадағы болжамды белсенді күндер", f"{clipped_pred} / 7 күн")
