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

    with st.expander("Consentimiento informado", expanded=True):
        st.markdown('''
        Antes de comenzar, lee y acepta las siguientes condiciones:

        **Uso de la camara:**
        - La camara se activa unicamente durante el test para capturar
          una fotografia por pregunta.
        - Las imagenes se procesan localmente en el momento de la captura
          y no se almacenan en ningun servidor.

        **Datos personales:**
        - El nombre ingresado se usa unicamente para identificar el reporte generado.
        - No se recopila, almacena ni comparte ningun dato personal.

        **Reporte:**
        - El reporte PDF se genera localmente y solo se descarga
          si el usuario lo solicita explicitamente.
        - Si el usuario no descarga el reporte, ningun resultado queda almacenado.

        **Uso de los resultados:**
        - Los resultados son orientativos y tienen fines exclusivamente academicos.
        - No constituyen un diagnostico psicologico ni clinico.
        ''')

    with st.expander("Limitaciones del sistema"):
        st.markdown('''
        **Del modelo de reconocimiento facial:**
        - Entrenado en FER2013, dataset recolectado en condiciones de laboratorio
          que no siempre refleja expresiones en contextos naturales.
        - Sesgo hacia la clase neutral por desbalance del dataset original
          (neutral: 4,965 imagenes vs disgust: 436 imagenes).
        - Resolucion original de 48x48 pixeles limita la capacidad de
          detectar expresiones sutiles o micro expresiones.

        **Del diseno metodologico:**
        - La foto se captura al presionar Siguiente, por lo que puede no
          coincidir con la expresion espontanea durante la lectura.
        - Las personas tienden a suprimir o neutralizar expresiones al
          saber que estan siendo observadas (efecto del observador).
        - La tabla de congruencia emocion-respuesta fue definida
          teoricamente, no validada empiricamente.

        **Del instrumento:**
        - El BFI-10 es una version reducida del BFI-44, con menor
          precision psicometrica por su brevedad.
        - Los resultados son orientativos y no constituyen
          un diagnostico psicologico.
        ''')

    with st.expander("Referencias"):
        st.markdown('''
        Costa, P. T., & McCrae, R. R. (1992). *Revised NEO Personality Inventory
        (NEO-PI-R) and NEO Five-Factor Inventory (NEO-FFI) professional manual.*
        Psychological Assessment Resources.

        Chen, X., Liang, C., Huang, D., Real, E., Wang, K., Liu, Y., & Le, Q.
        (2023). Symbolic discovery of optimization algorithms. *Advances in
        Neural Information Processing Systems (NeurIPS), 36.*

        Ekman, P. (1992). An argument for basic emotions. *Cognition & Emotion,
        6*(3-4), 169-200.

        Goodfellow, I., Erhan, D., Carrier, P. L., Courville, A., Mirza, M.,
        Hamou, B., & Bengio, Y. (2013). Challenges in representation learning:
        A report on three machine learning contests. *Neural Networks, 64,* 59-71.

        Rammstedt, B., & John, O. P. (2007). Measuring personality in one minute
        or less: A 10-item short version of the Big Five Inventory. *Journal of
        Research in Personality, 41*(1), 203-212.

        Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for
        convolutional neural networks. *International Conference on Machine
        Learning (ICML),* 6105-6114.
        ''')

    acepta = st.checkbox('He leido y acepto las condiciones de uso y consentimiento informado')
    nombre = st.text_input('Ingresa tu nombre:', placeholder='Nombre completo', disabled=not acepta)

    if st.button('Comenzar test', type='primary', disabled=not acepta):
        if not nombre:
            st.warning('Ingresa tu nombre para continuar')
        else:
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