import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)

# --- Configuración ---
st.set_page_config(
    page_title="Gráfica de Curva Implícita",
    page_icon="📈",
    layout="centered"
)

# Transformaciones seguras y amigables
SAFE_TRANSFORMATIONS = (
    standard_transformations
    + (implicit_multiplication_application,)  # xy → x*y
    + (convert_xor,)                          # ^ → **
)

# Estado de sesión
if 'show_plot' not in st.session_state:
    st.session_state.show_plot = False
if 'equation' not in st.session_state:
    st.session_state.equation = "x^2 + y^2 - 25"
if 'parsed_expr' not in st.session_state:
    st.session_state.parsed_expr = None
if 'valid_equation' not in st.session_state:
    st.session_state.valid_equation = False

# --- Título e instrucciones ---
st.title("📈 Gráfica de Curva Implícita")
with st.expander("📖 Instrucciones y ejemplos", expanded=False):
    st.markdown("""
    ### Cómo usar:
    1. **Escribe una ecuación** en términos de `x`, `y` y `t`
    2. **Ajusta los rangos** de X e Y según necesites
    3. **Configura el parámetro** `t` si tu ecuación lo usa
    4. **Haz clic en Graficar**
    
    ### Formatos aceptados:
    - **Forma implícita**: `F(x, y, t) = 0`
    - **Forma de igualdad**: `expresión izquierda = expresión derecha`
    
    ### Ejemplos:
    - **Círculo**: `x^2 + y^2 - t^2` o `x^2 + y^2 = t^2`
    - **Elipse**: `(x/t)^2 + (y/2)^2 - 1`
    - **Lemniscata**: `(x^2 + y^2)^2 - t^2*(x^2 - y^2)`
    - **Curva sinusoidal**: `sin(x) + cos(y) - t/2`
    - **Hipérbola**: `x*y - t`
    """)

# --- Entrada de ecuación (formato natural) ---
col_eq1, col_eq2 = st.columns([3, 1])
with col_eq1:
    user_input = st.text_input(
        "**Ecuación** (usa ^ para potencias, * para multiplicación):",
        value=st.session_state.equation,
        placeholder="Ej: x^2 + y^2 - 25",
        help="Escribe una expresión donde F(x, y, t) = 0"
    )
    
with col_eq2:
    st.markdown("###")
    parse_equation = st.button("🔍 Validar ecuación", use_container_width=True)

# --- Validar ecuación al cargar o cuando se presiona el botón ---
if parse_equation or (st.session_state.show_plot and not st.session_state.valid_equation):
    if user_input.strip():
        try:
            # Preparar expresión
            expr_str = user_input.strip()
            
            # Reemplazar ^ por **
            expr_str = expr_str.replace("^", "**")
            
            # Manejar igualdades
            if '=' in expr_str:
                parts = expr_str.split('=', 1)
                left = parts[0].strip()
                right = parts[1].strip()
                expr_str = f"({left}) - ({right})"
            
            # Parsear expresión
            x, y, t = sp.symbols('x y t')
            parsed_expr = parse_expr(
                expr_str,
                transformations=SAFE_TRANSFORMATIONS,
                local_dict={'x': x, 'y': y, 't': t},
                evaluate=False  # No evaluar para mantener la estructura
            )
            
            # Simplificar
            parsed_expr = sp.simplify(parsed_expr)
            
            # Guardar en estado
            st.session_state.parsed_expr = parsed_expr
            st.session_state.equation = user_input
            st.session_state.valid_equation = True
            
            # Mostrar expresión simplificada
            with col_eq2:
                st.success("✓ Válida")
                
            # Mostrar expresión formateada
            with st.expander("📝 Expresión interpretada", expanded=False):
                st.latex(f"F(x, y, t) = {sp.latex(parsed_expr)} = 0")
                
        except Exception as e:
            st.session_state.valid_equation = False
            st.error(f"❌ Error en la ecuación: {str(e)}")
            st.stop()
    else:
        st.warning("⚠️ Por favor, ingresa una ecuación")
        st.stop()

# --- Parámetros de dominio ---
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📏 Rango de X")
    x_min = st.number_input("X mínimo", value=-10.0, step=1.0, key="x_min")
    x_max = st.number_input("X máximo", value=10.0, step=1.0, key="x_max")
    if x_max <= x_min:
        st.error("❌ X máximo debe ser mayor que X mínimo.")
        st.stop()

with col2:
    st.subheader("📏 Rango de Y")
    y_min = st.number_input("Y mínimo", value=-10.0, step=1.0, key="y_min")
    y_max = st.number_input("Y máximo", value=10.0, step=1.0, key="y_max")
    if y_max <= y_min:
        st.error("❌ Y máximo debe ser mayor que Y mínimo.")
        st.stop()

with col3:
    st.subheader("⚙️ Resolución")
    resolution = st.select_slider(
        "Puntos por eje",
        options=[100, 200, 300, 400, 500, 600],
        value=300,
        help="Mayor resolución = gráfica más precisa pero más lenta"
    )

# --- Parámetro t ---
st.markdown("---")
st.subheader("🎚️ Parámetro $t$")

col_t1, col_t2, col_t3 = st.columns(3)
with col_t1:
    t_min = st.number_input("t mínimo", value=0.0, step=0.5, key="t_min")
with col_t2:
    t_max = st.number_input("t máximo", value=10.0, step=0.5, key="t_max")
with col_t3:
    t_step = st.number_input("Paso de t", value=0.5, min_value=0.01, step=0.1, key="t_step")

if t_max < t_min:
    st.error("❌ t máximo debe ser ≥ t mínimo.")
    st.stop()

# Slider para t con mejor formato
t_val = st.slider(
    f"Valor actual de $t$: **{t_val if 't_val' in locals() else (t_min + t_max)/2:.2f}**",
    min_value=float(t_min),
    max_value=float(t_max),
    value=float((t_min + t_max) / 2),
    step=float(t_step),
    format="%.2f"
)

# --- Botones ---
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])

with col_btn1:
    plot_clicked = st.button("📊 Graficar", type="primary", use_container_width=True)

with col_btn2:
    reset_clicked = st.button("🔄 Resetear", use_container_width=True)

with col_btn3:
    animate = st.checkbox("🎬 Animación automática", value=False)
    if animate:
        # Control de velocidad de animación
        speed = st.slider("Velocidad (ms)", 100, 2000, 500, 100)

if reset_clicked:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# --- Función para evaluar la expresión ---
def evaluate_expression(x_vals, y_vals, t_value):
    """Evalúa la expresión en una malla de puntos"""
    if st.session_state.parsed_expr is None:
        return None
    
    try:
        # Convertir a función numérica
        x_sym, y_sym, t_sym = sp.symbols('x y t')
        func = sp.lambdify(
            (x_sym, y_sym, t_sym),
            st.session_state.parsed_expr,
            modules=['numpy', 'math']
        )
        
        # Evaluar en la malla
        Z = func(x_vals, y_vals, t_value)
        
        # Manejar posibles valores complejos
        if np.iscomplexobj(Z):
            Z = Z.real
            
        return Z
    except Exception as e:
        st.error(f"Error al evaluar: {str(e)}")
        return None

# --- Renderizado del gráfico ---
if plot_clicked or (st.session_state.show_plot and st.session_state.valid_equation):
    if not st.session_state.valid_equation:
        st.error("⚠️ Por favor, valida primero la ecuación.")
        st.stop()
    
    st.session_state.show_plot = True
    
    # Crear malla adaptativa
    res_x = resolution
    res_y = int(resolution * (y_max - y_min) / (x_max - x_min))
    
    x_vals = np.linspace(x_min, x_max, res_x)
    y_vals = np.linspace(y_min, y_max, res_y)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # Contenedor para la animación
    plot_placeholder = st.empty()
    
    if animate:
        # Animación automática
        import time
        
        t_values = np.arange(t_min, t_max + t_step, t_step)
        for t_current in t_values:
            # Evaluar para el valor actual de t
            Z = evaluate_expression(X, Y, t_current)
            
            if Z is not None:
                # Limpiar el eje
                ax.clear()
                
                # Graficar curva de nivel en 0 con mayor precisión
                contour = ax.contour(
                    X, Y, Z,
                    levels=[0],
                    colors='#FF4B4B',
                    linewidths=2.5,
                    linestyles='-'
                )
                
                # Agregar contornos cercanos para mejor visualización
                ax.contour(
                    X, Y, Z,
                    levels=np.linspace(-5, 5, 11),
                    colors='gray',
                    linewidths=0.5,
                    linestyles='--',
                    alpha=0.3
                )
                
                # Estilo de ejes
                ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.7)
                ax.axvline(x=0, color='black', linewidth=0.5, alpha=0.7)
                
                # Configurar límites y aspecto
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_aspect('equal', adjustable='box')
                
                # Etiquetas
                ax.set_xlabel('x', fontsize=12, fontweight='bold')
                ax.set_ylabel('y', fontsize=12, fontweight='bold')
                
                # Título con valor de t
                ax.set_title(f'Curva implícita para $t = {t_current:.2f}$', 
                           fontsize=14, fontweight='bold', pad=20)
                
                # Grid
                ax.grid(True, linestyle='--', alpha=0.3)
                
                # Mostrar en Streamlit
                plot_placeholder.pyplot(fig)
                
                # Pausa para animación
                time.sleep(speed / 1000)
    else:
        # Gráfico estático
        Z = evaluate_expression(X, Y, t_val)
        
        if Z is not None:
            # Gráfico principal: curva en 0
            contour = ax.contour(
                X, Y, Z,
                levels=[0],
                colors='#FF4B4B',
                linewidths=2.5,
                linestyles='-'
            )
            
            # Contornos adicionales para contexto
            ax.contour(
                X, Y, Z,
                levels=np.linspace(-10, 10, 21),
                colors='gray',
                linewidths=0.5,
                linestyles='--',
                alpha=0.3
            )
            
            # Rellenar regiones positivas/negativas
            ax.contourf(
                X, Y, Z,
                levels=[-np.inf, 0, np.inf],
                colors=['#E6F3FF', '#FFE6E6'],
                alpha=0.2
            )
            
            # Ejes cartesianos
            ax.axhline(y=0, color='black', linewidth=1, alpha=0.7)
            ax.axvline(x=0, color='black', linewidth=1, alpha=0.7)
            
            # Configuración del gráfico
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel('x', fontsize=12, fontweight='bold')
            ax.set_ylabel('y', fontsize=12, fontweight='bold')
            
            # Título informativo
            eq_display = st.session_state.equation[:30] + "..." if len(st.session_state.equation) > 30 else st.session_state.equation
            ax.set_title(f'Curva: {eq_display}\n$t = {t_val:.2f}$', 
                       fontsize=13, fontweight='bold', pad=20)
            
            # Grid mejorado
            ax.grid(True, linestyle='--', alpha=0.3, which='both')
            
            # Configurar ticks
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, symmetric=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, symmetric=True))
            
            # Leyenda informativa
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#FFE6E6', alpha=0.3, edgecolor='gray', label='F(x,y,t) < 0'),
                Patch(facecolor='#E6F3FF', alpha=0.3, edgecolor='gray', label='F(x,y,t) > 0'),
                Patch(facecolor='none', edgecolor='#FF4B4B', linewidth=2.5, label='F(x,y,t) = 0')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
            
            # Mostrar métricas
            with st.expander("📊 Métricas de la curva", expanded=False):
                col_met1, col_met2 = st.columns(2)
                with col_met1:
                    if len(contour.allsegs[0]) > 0:
                        curve_points = contour.allsegs[0][0]
                        st.metric("Puntos en la curva", len(curve_points))
                        if len(curve_points) > 0:
                            x_curve = curve_points[:, 0]
                            y_curve = curve_points[:, 1]
                            st.metric("Rango X en curva", f"[{x_curve.min():.2f}, {x_curve.max():.2f}]")
                with col_met2:
                    if len(contour.allsegs[0]) > 0:
                        curve_points = contour.allsegs[0][0]
                        if len(curve_points) > 0:
                            y_curve = curve_points[:, 1]
                            st.metric("Rango Y en curva", f"[{y_curve.min():.2f}, {y_curve.max():.2f}]")
            
            # Mostrar gráfico
            st.pyplot(fig)
            
            # Advertencia si no hay curva
            if len(contour.allsegs[0]) == 0:
                st.warning("""
                ⚠️ **No se detectó curva en el nivel 0**
                
                Esto puede deberse a:
                1. El valor de $t$ no produce una curva real en este rango
                2. La resolución puede ser insuficiente
                3. La ecuación no tiene solución real para estos parámetros
                
                **Sugerencias:**
                - Aumenta la resolución
                - Cambia el rango de X/Y
                - Ajusta el valor de $t$
                - Verifica que la ecuación esté correcta
                """)
            
            # Botón para descargar datos
            if len(contour.allsegs[0]) > 0:
                curve_points = contour.allsegs[0][0]
                if st.button("💾 Descargar puntos de la curva"):
                    import io
                    buffer = io.BytesIO()
                    np.savetxt(buffer, curve_points, delimiter=',', header='x,y', comments='')
                    st.download_button(
                        label="Descargar CSV",
                        data=buffer.getvalue(),
                        file_name=f"curva_t={t_val:.2f}.csv",
                        mime="text/csv"
                    )

else:
    # Estado inicial
    st.markdown("---")
    with st.container():
        st.info("""
        ### 👋 ¡Bienvenido a la app de gráficas implícitas!
        
        1. **Escribe tu ecuación** en el campo superior
        2. **Haz clic en 'Validar ecuación'** para verificar la sintaxis
        3. **Ajusta los parámetros** según necesites
        4. **Presiona 'Graficar'** para visualizar la curva
        
        **Consejo:** Comienza con una ecuación simple como `x^2 + y^2 - 25` para un círculo de radio 5.
        """)
    
    # Ejemplo previsualizado
    with st.expander("🔍 Ver ejemplo predefinido", expanded=True):
        st.code("""
        # Ejemplo: Círculo que cambia con t
        x^2 + y^2 - t^2
        
        # Configuración sugerida:
        X: [-10, 10]
        Y: [-10, 10]
        t: [0, 10]
        
        # Esto mostrará círculos de radio t
        """)