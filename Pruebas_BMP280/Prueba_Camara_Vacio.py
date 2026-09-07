import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. FUNCIÓN PARA CARGAR Y APLICAR LAS MEJORAS 
# =============================================================================
def procesar_senal_avanzada(ruta_archivo, periodos=5, N_padded=65536):
    # Carga básica
    datos = np.loadtxt(ruta_archivo, delimiter=';', skiprows=0,
                       converters={0: lambda s: float(s.replace(',', '.')),
                                   1: lambda s: float(s.replace(',', '.'))})
    t_original = (datos[:, 0] - datos[0, 0]) / 1000.0
    alt_original = datos[:, 1]
    
    dt = np.mean(np.diff(t_original))
    
    # --- MEJORA 1: EVITAR EL ESCALÓN (Espejado y Concatenado) ---
    # Tomamos la señal [1,2,3,4,5] y la convertimos en [1,2,3,4,5,5,4,3,2,1]
    alt_espejada = np.concatenate((alt_original, alt_original[::-1]))
    
    # --- MEJORA 2: HACERLA PERIÓDICA (Repetir muchos períodos) ---
    alt_periodica = np.tile(alt_espejada, periodos)
    
    # Reconstruimos un eje de tiempo ficticio continuo para esta nueva señal larga
    N_total_periodica = len(alt_periodica)
    t_periodica = np.arange(N_total_periodica) * dt
    
    # --- MEJORA 3: ZEROPADDING ---
    # Hacemos la FFT especificando un número grande de puntos (N_padded)
    # Esto mete ceros al final automáticamente e interpola el espectro
    fft_resultado = np.fft.fft(alt_periodica, n=N_padded)
    magnitud = np.abs(fft_resultado)
    frecuencias = np.fft.fftfreq(N_padded, d=dt)
    
    # Recorte de Nyquist (mitad del espectro)
    mitad = N_padded // 2
    frecuencias_nyquist = frecuencias[:mitad]
    magnitud_nyquist = magnitud[:mitad]
    
    # Normalizamos la magnitud por la cantidad de muestras de un solo bloque 
    # para que las escalas visuales sigan teniendo sentido físico
    magnitud_nyquist = magnitud_nyquist / len(alt_original)
    
    return t_original, alt_original, frecuencias_nyquist, magnitud_nyquist

# =============================================================================
# 2. PROCESAMIENTO DE AMBOS ARCHIVOS
# =============================================================================
ruta_con_filtro = r"C:\Users\lucas\OneDrive\Escritorio\Pruebas\despegue_datos_con_filtro.txt"
ruta_sin_filtro = r"C:\Users\lucas\OneDrive\Escritorio\Pruebas\despegue_datos_sin_filtro.txt"

# Ejecutamos el algoritmo avanzado con 5 periodos y padding de 65536 puntos
t_con, alt_con, f_con, mag_con = procesar_senal_avanzada(ruta_con_filtro)
t_sin, alt_sin, f_sin, mag_sin = procesar_senal_avanzada(ruta_sin_filtro)

# =============================================================================
# 3. GRAFICACIÓN DE LOS RESULTADOS OPTIMIZADOS
# =============================================================================
# Cambiamos a un solo "ax" y ajustamos la altura de 9 a 6
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Señales de Altitud (Cruda vs Filtrada)', fontsize=14, fontweight='bold')

# --- Gráfico 1: Tiempo (Se muestra el bloque original analizado) ---
ax.plot(t_sin, alt_sin, color='gray', alpha=0.6, linewidth=1, label='Señal Cruda (Sin Filtro)')
ax.plot(t_con, alt_con, color='blue', linewidth=1.5, label='Señal Filtrada (Alpha-Beta)')
ax.set_title('Dominio del Tiempo: Un Período Base de Muestreo')
ax.set_xlabel('Tiempo (segundos)')
ax.set_ylabel('Altitud (metros)')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()

plt.tight_layout()
plt.show()
