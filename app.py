import numpy as np
import gradio as gr
import joblib
import warnings
warnings.filterwarnings('ignore')

# Загружаем scaler (модель .keras не используем — она слишком тяжёлая для бесплатного хостинга)
try:
    scaler = joblib.load('blood_scaler.pkl')
    print("Scaler загружен")
except:
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
    # Правила из kod.py
    if HGB < 100 and MCV < 77:
        diag, conf = 1, 85
    elif HGB < 100 and MCV > 105:
        diag, conf = 2, 85
    elif HGB > 175 and WBC > 10:
        diag, conf = 3, 85
    elif WBC > 16 and NEUT > 80:
        diag, conf = 4, 90
    elif WBC < 4.5 and LYMPH > 45:
        diag, conf = 5, 90
    elif WBC > 55 and EO > 5:
        diag, conf = 6, 80
    elif PLT < 50:
        diag, conf = 7, 90
    elif EO > 14:
        diag, conf = 8, 85
    elif WBC < 3.5 and HGB < 100 and PLT < 100:
        diag, conf = 9, 85
    else:
        diag, conf = 0, 90
    
    result = {}
    result["ДИАГНОЗ"] = f"{class_names[diag]}"
    result["Уверенность"] = f"{conf:.1f}%"
    
    warns = []
    if WBC > 12: warns.append(f"Лейкоциты повышены ({WBC:.1f})")
    elif WBC < 3.5: warns.append(f"Лейкоциты понижены ({WBC:.1f})")
    if HGB < 100: warns.append(f"Гемоглобин низкий ({HGB:.0f})")
    elif HGB > 170: warns.append(f"Гемоглобин высокий ({HGB:.0f})")
    if MCV < 77: warns.append(f"MCV понижен ({MCV:.0f})")
    elif MCV > 105: warns.append(f"MCV повышен ({MCV:.0f})")
    if PLT < 50: warns.append(f"Тромбоциты критически низкие!")
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
    
    btn.click(fn=predict, inputs=[WBC, RBC, HGB, HCT, MCV, MCH, PLT, NEUT, LYMPH, EO, Age, Sex], outputs=[diag, warns_out])

app.launch(server_name="0.0.0.0", server_port=8080)
