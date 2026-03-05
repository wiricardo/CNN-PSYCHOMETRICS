import cv2
import torch
import timm
import os
import io
import numpy as np
from torchvision import transforms
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from datetime import datetime

# ============================================
# CONFIGURACIÓN
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELO_EMOCIONES = os.path.join(BASE_DIR, '../Modelo/mejor_modelo.pth')
DEVICE = torch.device('cpu')

CLASES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

CLASES_ES = {
    'angry':    'enojo',
    'disgust':  'asco',
    'fear':     'miedo',
    'happy':    'feliz',
    'neutral':  'neutral',
    'sad':      'tristeza',
    'surprise': 'sorpresa'
}

ITEMS = [
    {'id': 1,  'texto': 'Soy reservado/a',                    'dimension': 'E', 'direccion': '-'},
    {'id': 2,  'texto': 'En general confio en los demas',     'dimension': 'A', 'direccion': '+'},
    {'id': 3,  'texto': 'Tiendo a ser perezoso/a',            'dimension': 'C', 'direccion': '-'},
    {'id': 4,  'texto': 'Me relajo facilmente',               'dimension': 'N', 'direccion': '-'},
    {'id': 5,  'texto': 'Tengo poca imaginacion artistica',   'dimension': 'O', 'direccion': '-'},
    {'id': 6,  'texto': 'Soy sociable y extrovertido/a',      'dimension': 'E', 'direccion': '+'},
    {'id': 7,  'texto': 'Tiendo a criticar a los demas',      'dimension': 'A', 'direccion': '-'},
    {'id': 8,  'texto': 'Hago las cosas eficientemente',      'dimension': 'C', 'direccion': '+'},
    {'id': 9,  'texto': 'Me pongo nervioso/a facilmente',     'dimension': 'N', 'direccion': '+'},
    {'id': 10, 'texto': 'Tengo una imaginacion activa',       'dimension': 'O', 'direccion': '+'},
]

CONGRUENCIA = {
    1:  {'alta': ['neutral', 'sad'],          'baja': ['happy', 'surprise']},
    2:  {'alta': ['happy', 'neutral'],         'baja': ['angry', 'disgust', 'fear']},
    3:  {'alta': ['neutral', 'sad'],           'baja': ['happy']},
    4:  {'alta': ['happy', 'neutral'],         'baja': ['fear', 'angry', 'sad']},
    5:  {'alta': ['neutral'],                  'baja': ['surprise', 'happy']},
    6:  {'alta': ['happy', 'surprise'],        'baja': ['sad', 'neutral', 'fear']},
    7:  {'alta': ['angry', 'disgust'],         'baja': ['happy', 'neutral']},
    8:  {'alta': ['happy', 'neutral'],         'baja': ['sad', 'angry', 'fear']},
    9:  {'alta': ['fear', 'angry', 'sad'],     'baja': ['happy', 'neutral']},
    10: {'alta': ['surprise', 'happy'],        'baja': ['neutral', 'sad']},
}

NOMBRES_DIM = {
    'O': 'Apertura a la experiencia',
    'C': 'Responsabilidad',
    'E': 'Extraversion',
    'A': 'Amabilidad',
    'N': 'Neuroticismo'
}

# ============================================
# CARGAR MODELOS
# ============================================
def cargar_modelos():
    model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
    yolo = YOLO(model_path)

    efficientnet = timm.create_model('efficientnet_b0', pretrained=False, num_classes=7)
    efficientnet.load_state_dict(torch.load(MODELO_EMOCIONES, map_location=DEVICE))
    efficientnet = efficientnet.to(DEVICE)
    efficientnet.eval()

    return yolo, efficientnet

# ============================================
# PREPROCESAMIENTO
# ============================================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============================================
# DETECTAR EMOCION
# ============================================
def detectar_emocion(frame, yolo, efficientnet):
    results = yolo(frame, verbose=False)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cara = frame[y1:y2, x1:x2]
            if cara.size == 0:
                continue
            try:
                cara_tensor = transform(cara).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    output = efficientnet(cara_tensor)
                    probs = torch.softmax(output, dim=1)
                    conf = float(probs.max())
                    idx = output.argmax(1).item()
                    return CLASES[idx], conf
            except:
                pass
    return 'neutral', 0.0

# ============================================
# CALCULAR CONGRUENCIA
# ============================================
def calcular_congruencia(item_id, respuesta, emocion):
    if respuesta == 3:
        return 'neutral'
    regla = CONGRUENCIA[item_id]
    if respuesta >= 4:
        return 'congruente' if emocion in regla['alta'] else 'incongruente'
    else:
        return 'congruente' if emocion in regla['baja'] else 'incongruente'

# ============================================
# CALCULAR OCEAN
# ============================================
def calcular_ocean(respuestas):
    items_por_dimension = {
        'O': [5, 10], 'C': [3, 8], 'E': [1, 6],
        'A': [2, 7],  'N': [4, 9]
    }
    puntajes = {}
    for dim, ids in items_por_dimension.items():
        valores = []
        for item_id in ids:
            item = next(i for i in ITEMS if i['id'] == item_id)
            r = respuestas[item_id]
            if item['direccion'] == '-':
                r = 6 - r
            valores.append(r)
        puntajes[dim] = round(sum(valores) / 2, 2)
    return puntajes

# ============================================
# GENERAR PDF
# ============================================
def generar_pdf(nombre, respuestas, emociones, congruencias, puntajes_ocean):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    elementos = []

    titulo_style = ParagraphStyle('titulo', parent=styles['Title'],
                                   fontSize=18, textColor=colors.HexColor('#2E4057'))
    subtitulo_style = ParagraphStyle('subtitulo', parent=styles['Heading2'],
                                      fontSize=13, textColor=colors.HexColor('#2E4057'))

    elementos.append(Paragraph('Reporte Big Five - BFI-10', titulo_style))
    elementos.append(Paragraph(f'Participante: {nombre}', styles['Normal']))
    elementos.append(Paragraph(f'Fecha: {datetime.now().strftime("%d/%m/%Y %H:%M")}', styles['Normal']))
    elementos.append(Spacer(1, 0.5*cm))
    elementos.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#2E4057')))
    elementos.append(Spacer(1, 0.5*cm))

    elementos.append(Paragraph('Perfil de Personalidad OCEAN', subtitulo_style))
    elementos.append(Spacer(1, 0.3*cm))

    datos_ocean = [['Dimension', 'Puntaje', 'Nivel']]
    for dim, puntaje in puntajes_ocean.items():
        nivel = 'Alto' if puntaje >= 3.5 else 'Bajo'
        datos_ocean.append([NOMBRES_DIM[dim], str(puntaje), nivel])

    tabla_ocean = Table(datos_ocean, colWidths=[9*cm, 3*cm, 3*cm])
    tabla_ocean.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2E4057')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#F0F4F8'), colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CCCCCC')),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    elementos.append(tabla_ocean)
    elementos.append(Spacer(1, 0.7*cm))

    elementos.append(Paragraph('Tabla de Congruencia Respuesta - Emocion', subtitulo_style))
    elementos.append(Spacer(1, 0.3*cm))

    datos_tabla = [['#', 'Item', 'Respuesta', 'Emocion', 'Congruencia']]
    for item in ITEMS:
        iid = item['id']
        cong = congruencias[iid]
        simbolo = 'Congruente' if cong == 'congruente' else ('Neutral' if cong == 'neutral' else 'Incongruente')
        datos_tabla.append([
            str(iid),
            item['texto'],
            str(respuestas[iid]),
            CLASES_ES[emociones[iid]],
            simbolo
        ])

    tabla_cong = Table(datos_tabla, colWidths=[1*cm, 7.5*cm, 2.5*cm, 2.5*cm, 2.5*cm])
    tabla_cong.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2E4057')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('ALIGN', (1,1), (1,-1), 'LEFT'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#F0F4F8'), colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CCCCCC')),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    elementos.append(tabla_cong)

# Limitaciones
    elementos.append(Spacer(1, 0.7*cm))
    elementos.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#2E4057')))
    elementos.append(Spacer(1, 0.3*cm))
    elementos.append(Paragraph('Limitaciones del sistema', subtitulo_style))
    elementos.append(Spacer(1, 0.3*cm))

    limitaciones = [
        'Del modelo de reconocimiento facial: entrenado en FER2013, dataset recolectado '
        'en condiciones de laboratorio. Presenta sesgo hacia la clase neutral por '
        'desbalance del dataset original. La resolucion de 48x48 pixeles limita la '
        'deteccion de micro expresiones.',

        'Del diseno metodologico: la foto se captura al presionar Siguiente, por lo que '
        'puede no coincidir con la expresion espontanea. Las personas tienden a suprimir '
        'expresiones al saber que son observadas (efecto del observador). La tabla de '
        'congruencia fue definida teoricamente, no validada empiricamente.',

        'Del instrumento: el BFI-10 es una version reducida del BFI-44 con menor '
        'precision psicometrica. Los resultados son orientativos y no constituyen '
        'un diagnostico psicologico.',
    ]

    for lim in limitaciones:
        elementos.append(Paragraph(f'- {lim}', styles['Normal']))
        elementos.append(Spacer(1, 0.2*cm))

    # Referencias
    elementos.append(Spacer(1, 0.5*cm))
    elementos.append(Paragraph('Referencias', subtitulo_style))
    elementos.append(Spacer(1, 0.3*cm))

    referencias = [
        'Chen, X., Liang, C., Huang, D., Real, E., Wang, K., Liu, Y., & Le, Q. (2023). '
        'Symbolic discovery of optimization algorithms. Advances in Neural Information '
        'Processing Systems (NeurIPS), 36.',

        'Costa, P. T., & McCrae, R. R. (1992). Revised NEO Personality Inventory '
        '(NEO-PI-R) and NEO Five-Factor Inventory (NEO-FFI) professional manual. '
        'Psychological Assessment Resources.',

        'Ekman, P. (1992). An argument for basic emotions. Cognition & Emotion, '
        '6(3-4), 169-200.',

        'Goodfellow, I., Erhan, D., Carrier, P. L., Courville, A., Mirza, M., '
        'Hamou, B., & Bengio, Y. (2013). Challenges in representation learning: '
        'A report on three machine learning contests. Neural Networks, 64, 59-71.',

        'Rammstedt, B., & John, O. P. (2007). Measuring personality in one minute '
        'or less: A 10-item short version of the Big Five Inventory. Journal of '
        'Research in Personality, 41(1), 203-212.',

        'Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for '
        'convolutional neural networks. International Conference on Machine '
        'Learning (ICML), 6105-6114.',
    ]

    ref_style = ParagraphStyle('ref', parent=styles['Normal'],
                                fontSize=8, leftIndent=20, firstLineIndent=-20)

    for ref in referencias:
        elementos.append(Paragraph(ref, ref_style))
        elementos.append(Spacer(1, 0.2*cm))


    doc.build(elementos)
    buffer.seek(0)
    return buffer