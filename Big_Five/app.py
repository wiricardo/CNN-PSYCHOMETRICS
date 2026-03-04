import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from bigfive import (
    cargar_modelos, detectar_emocion, calcular_congruencia,
    calcular_ocean, generar_pdf, ITEMS, CLASES_ES, NOMBRES_DIM
)

st.markdown("""
<style>
[data-testid="stCameraInputButton"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)





st.set_page_config(page_title='BFI-10', layout='centered')

# Estado inicial
for key, val in [('pagina', 'inicio'), ('item_actual', 0),
                  ('respuestas', {}), ('emociones', {}), ('congruencias', {})]:
    if key not in st.session_state:
        st.session_state[key] = val

# ============================================
# PAGINA: INICIO
# ============================================
if st.session_state.pagina == 'inicio':
    st.title('Test de Personalidad Big Five')
    st.markdown('#### BFI-10 con analisis de expresiones faciales')
    st.markdown('''
    Este test evaluara tu personalidad en 5 dimensiones:

    - Apertura a la experiencia
    - Responsabilidad
    - Extraversion
    - Amabilidad
    - Neuroticismo

    Durante el test, la camara registrara tus expresiones faciales
    para analizar la congruencia con tus respuestas.
    ''')
    nombre = st.text_input('Ingresa tu nombre:', placeholder='Nombre completo')
    if st.button('Comenzar test', type='primary') and nombre:
        st.session_state.nombre = nombre
        st.session_state.pagina = 'test'
        st.rerun()

# ============================================
# PAGINA: TEST
# ============================================
elif st.session_state.pagina == 'test':
    yolo, efficientnet = st.cache_resource(cargar_modelos)()
    idx = st.session_state.item_actual
    item = ITEMS[idx]

    st.progress(idx / 10, text=f'Pregunta {idx+1} de 10')
    st.markdown('*Que tan de acuerdo estas con esta afirmacion?*')
    st.markdown(f'### {item["texto"]}')

    opciones = {
        1: '1 - Muy en desacuerdo',
        2: '2 - En desacuerdo',
        3: '3 - Neutral',
        4: '4 - De acuerdo',
        5: '5 - Muy de acuerdo'
    }

    respuesta = st.radio('', list(opciones.values()), index=2,
                         key=f'radio_{idx}', label_visibility='collapsed')
    respuesta_num = list(opciones.keys())[list(opciones.values()).index(respuesta)]


    foto = st.camera_input('', label_visibility='collapsed')

    if st.button('Siguiente', type='primary'):
        emocion = 'neutral'
        if foto is not None:
            bytes_data = foto.getvalue()
            nparr = np.frombuffer(bytes_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            emocion, _ = detectar_emocion(frame, yolo, efficientnet)

        congruencia = calcular_congruencia(item['id'], respuesta_num, emocion)

        st.session_state.respuestas[item['id']] = respuesta_num
        st.session_state.emociones[item['id']] = emocion
        st.session_state.congruencias[item['id']] = congruencia

        if idx + 1 < 10:
            st.session_state.item_actual += 1
        else:
            st.session_state.pagina = 'reporte'
        st.rerun()

# ============================================
# PAGINA: REPORTE
# ============================================
elif st.session_state.pagina == 'reporte':
    puntajes = calcular_ocean(st.session_state.respuestas)

    st.title('Resultados')
    st.markdown(f'**Participante:** {st.session_state.nombre}')

    st.markdown('### Perfil de Personalidad')
    fig, ax = plt.subplots(figsize=(8, 4))
    dims = list(puntajes.keys())
    vals = list(puntajes.values())
    colores_bar = ['#4C9BE8', '#5CB85C', '#F0AD4E', '#E8734C', '#D9534F']
    bars = ax.barh([NOMBRES_DIM[d] for d in dims], vals, color=colores_bar)
    ax.set_xlim(1, 5)
    ax.axvline(x=3, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Puntaje')
    for bar, val in zip(bars, vals):
        ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                f'{val}', va='center', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown('### Tabla de Congruencia')
    datos = []
    for item in ITEMS:
        iid = item['id']
        cong = st.session_state.congruencias[iid]
        simbolo = 'Congruente' if cong == 'congruente' else ('Neutral' if cong == 'neutral' else 'Incongruente')
        datos.append({
            'Item': item['texto'],
            'Respuesta': st.session_state.respuestas[iid],
            'Emocion': CLASES_ES[st.session_state.emociones[iid]],
            'Congruencia': simbolo
        })
    st.table(datos)

    pdf_buffer = generar_pdf(
        st.session_state.nombre,
        st.session_state.respuestas,
        st.session_state.emociones,
        st.session_state.congruencias,
        puntajes
    )

    st.download_button(
        label='Descargar reporte PDF',
        data=pdf_buffer,
        file_name=f'reporte_{st.session_state.nombre}.pdf',
        mime='application/pdf',
        type='primary'
    )

    if st.button('Reiniciar test'):
        for key in ['pagina', 'item_actual', 'respuestas', 'emociones', 'congruencias', 'nombre']:
            del st.session_state[key]
        st.rerun()