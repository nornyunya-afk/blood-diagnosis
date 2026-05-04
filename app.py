import numpy as np
import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import warnings
warnings.filterwarnings('ignore')

# Загрузка модели
try:
    model = load_model('blood_diagnosis_model.keras')
    scaler = joblib.load('blood_scaler.pkl')
    print("Модель и scaler загружены!")
except Exception as e:
    print(f"Ошибка загрузки: {e}")
    model = None
    scaler = None

class_names = {
    0: 'Норма (здоров)',
    1: 'Железодефицитная анемия (ЖДА)',
    2: 'B12-дефицитная анемия',
    3: 'Истинная полицитемия (эритремия)',
    4: 'Острая бактериальная инфекция / сепсис',
    5: 'Острая вирусная инфекция (ОРВИ/грипп)',
    6: 'Хронический миелолейкоз (ХМЛ)',
    7: 'Идиопатическая тромбоцитопеническая пурпура (ИТП)',
    8: 'Гельминтоз / паразитарная инвазия',
    9: 'Апластическая анемия (панцитопения)'
}

def predict(WBC, RBC, HGB, HCT, MCV, MCH, PLT, NEUT, LYMPH, EO, Age, Sex):
    if model is None:
        return {"Ошибка": "Модель не загружена"}, ""
    
    sex_num = 0 if Sex == "Мужской" else 1
    data = np.array([[WBC, RBC, HGB, HCT, MCV, MCH, PLT, NEUT, LYMPH, EO, Age, sex_num]])
    data_s = scaler.transform(data)
    pred = model.predict(data_s, verbose=0)[0]
    top3 = np.argsort(pred)[-3:][::-1]
    
    result = {}
    result["ДИАГНОЗ"] = f"{class_names[top3[0]]} ({pred[top3[0]]*100:.1f}%)"
    for i, idx in enumerate(top3[1:], 2):
        result[f"{i}. {class_names[idx]}"] = f"{pred[idx]*100:.1f}%"
    
    warns = []
    if WBC > 12: warns.append(f"Лейкоциты повышены ({WBC:.1f})")
    elif WBC < 3.5: warns.append(f"Лейкоциты понижены ({WBC:.1f})")
    if HGB < 100: warns.append(f"Гемоглобин низкий ({HGB:.0f})")
    if not warns: warns.append("Показатели в норме")
    
    return result, "\n".join(warns)

with gr.Blocks(title="Диагностика по ОАК") as app:
    gr.Markdown("# 🏥 ДИАГНОСТИКА ЗАБОЛЕВАНИЙ ПО АНАЛИЗУ КРОВИ")
    with gr.Row():
        with gr.Column():
            WBC = gr.Number(label="Лейкоциты (WBC)", value=6.5)
            RBC = gr.Number(label="Эритроциты (RBC)", value=4.8)
            HGB = gr.Number(label="Гемоглобин (HGB)", value=145.0)
            HCT = gr.Number(label="Гематокрит (HCT)", value=44.0)
            MCV = gr.Number(label="MCV", value=89.0)
            MCH = gr.Number(label="MCH", value=30.0)
            PLT = gr.Number(label="Тромбоциты (PLT)", value=260.0)
            NEUT = gr.Slider(label="Нейтрофилы %", value=58.0, minimum=0, maximum=100)
            LYMPH = gr.Slider(label="Лимфоциты %", value=32.0, minimum=0, maximum=100)
            EO = gr.Slider(label="Эозинофилы %", value=2.5, minimum=0, maximum=100)
            Age = gr.Number(label="Возраст", value=35)
            Sex = gr.Radio(label="Пол", choices=["Мужской", "Женский"], value="Мужской")
            btn = gr.Button("🔍 ДИАГНОСТИРОВАТЬ", variant="primary", size="lg")
        with gr.Column():
            diag = gr.Label(label="Результат", num_top_classes=3)
            warns_out = gr.Textbox(label="Анализ показателей", lines=5)
    
    btn.click(
        fn=predict,
        inputs=[WBC, RBC, HGB, HCT, MCV, MCH, PLT, NEUT, LYMPH, EO, Age, Sex],
        outputs=[diag, warns_out]
    )

app.launch()
