import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import Tuple, List, Optional

class RegresionPolinomial:
    """
    Implementación de Regresión Polinomial utilizando el método matricial de Mínimos Cuadrados Ordinarios.
    
    Esta clase permite inicializar un modelo polinomial paramétrico, computar analíticamente los 
    coeficientes de ajuste óptimo con respecto a un dominio experimental real, y predecir los escalares 
    condicionantes posteriores.

    Attributes:
        grado (int): Denota el grado máximo de expansión de la superficie polinomial predictora.
        coef (Optional[np.ndarray]): Vector algebraico continuo multidimensional de coeficientes calculados.
    """
    def __init__(self, grado: int) -> None:
        self.grado = grado
        self.coef: Optional[np.ndarray] = None

    def construir_phi(self, x: np.ndarray) -> np.ndarray:
        """
        Produce una matriz de diseño de covariabilidad, referida algorítmicamente 
        como Matriz de Vandermonde adaptada al componente continuo.
        
        Args:
            x (np.ndarray): Secuencia del atributo o variable independiente.
            
        Returns:
            np.ndarray: Arreglo de dimensiones ampliadas, evaluando las potencias de X secuencialmente.
        """
        return np.column_stack([x**j for j in range(self.grado + 1)])

    def ajustar(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Solución computacional cerrada mediante álgebra lineal para determinar
        la derivada que minimiza uniformemente el cuadrado de la función de costos.
        
        Aplica la Ecuación Normal: Theta = (Phi^T * Phi)^-1 * (Phi^T * Y)
        
        Args:
            x (np.ndarray): Tensor de observaciones del eje dimensional independiente.
            y (np.ndarray): Tensor de eventos u anotaciones analíticas correspondientes (eje dependiente).
            
        Returns:
            float: El número de condición estocástica asociado a la matriz transpuesta multiplicativa, empleado 
                   como medida preventiva de vulnerabilidad teórica de estabilidad del hardware per-calculus.
        """
        Phi = self.construir_phi(x)
        # Matriz C transpuesta del tensor proyectado y B su correspondencia multivariable
        C = Phi.T @ Phi
        B = Phi.T @ y

        self.coef = np.linalg.solve(C, B)
        condicion = np.linalg.cond(C)
        
        print(f"\n--- Matriz de Coeficientes (Grado {self.grado}) ---")
        print(self.coef)
        print("--------------------------------------------------\n")

        return condicion

    def predecir(self, x: np.ndarray) -> np.ndarray:
        """
        Mapea el recorrido tensorial del dominio pre-entrenado calculando
        el subproducto del ajuste de datos contra el set paramétrico extraído.
        
        Args:
            x (np.ndarray): Datos de pre-procesamiento del espacio X que se intentará calificar.
            
        Returns:
            np.ndarray: Vector univariado correspondiente a las aproximaciones numéricas del ente virtual.
        """
        if self.coef is None:
            raise ValueError("Infracción Crítica: Se intentó generar un vector predecido previo a la calibración del modelo.")
        Phi = self.construir_phi(x)
        return Phi @ self.coef

    def recm(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Raíz del Error Cuadrático Medio ajustado por grados de libertad (RECM / RMSE).
        Aplica el estimador insesgado de la varianza residual según el Teorema de Gauss-Markov,
        usando (n - m - 1) como denominador, donde m es el grado del polinomio.

        Args:
            y (np.ndarray): Target observacional validado del set real.
            y_pred (np.ndarray): Predicción emitida de la máquina de inferencia.

        Returns:
            float: Raíz cuadrada del error cuadrático medio ajustado.

        Raises:
            ValueError: Si los grados de libertad son insuficientes (n <= m + 1).
        """
        n = len(y)
        m = self.grado
        grados_libertad = n - m - 1

        if grados_libertad <= 0:
            raise ValueError(
                f"Grados de libertad insuficientes: n={n}, m={m}. "
                f"Se necesitan al menos {m + 2} observaciones."
            )

        return np.sqrt(np.sum((y - y_pred) ** 2) / grados_libertad)

    def r2(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute el coeficiente analítico de R² equivalente. Útil para expresar porcentualmente la captura
        de varianza representativa lograda desde la estructura polinomial inicializada.
        
        Args:
            y (np.ndarray): Referencia original verídica.
            y_pred (np.ndarray): Exponente inferido.
            
        Returns:
            float: Escalar que refleja la capacidad de entendimiento del modelo en el rango (-inf, 1.0].
        """
        ss_total = np.sum((y - np.mean(y))**2)
        ss_res = np.sum((y - y_pred)**2)
        return 1.0 - (ss_res / ss_total)


# ==========================================
# FUNCIONES DE VISUALIZACIÓN Y ANÁLISIS
# ==========================================

def comparar_modelos(x: np.ndarray, y: np.ndarray) -> int:
    """
    Automatiza la iteración empírica en multi-grado paramétrico probando
    y determinando qué estructura posee mayor retención en datos foráneos sin sobreajustarse.
    
    Técnica de ingeniería: Implementación rudimentaria orientativa de Validation Set Division (80-20 Split Hold-out).
    
    Args:
        x (np.ndarray): Eje X poblacional matricial primario.
        y (np.ndarray): Eje Y poblacional anotado.
        
    Returns:
        int: El término escalar que identificó estadísticamente al modelo óptimo de prueba.
    """
    n = len(x)
    split = int(0.8 * n)

    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    grados = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    resultados: List[Tuple[int, float]] = []

    print("\n=== REPORTE METRICULAR DE ESTUDIO COMPARATIVO ===\n")

    for grado in grados:
        modelo = RegresionPolinomial(grado)
        cond = modelo.ajustar(x_train, y_train)

        y_train_pred = modelo.predecir(x_train)
        y_test_pred = modelo.predecir(x_test)

        ecm_train = modelo.recm(y_train, y_train_pred)
        ecm_test = modelo.recm(y_test, y_test_pred)
        r2_val = modelo.r2(y_test, y_test_pred)

        resultados.append((grado, ecm_test))

        print(f"Propuesta Arquitectónica Grado {grado}")
        print(f"Raíz del Error Cuadrático Medio Local (Train): {ecm_train:.4f}")
        print(f"Raíz del Error Cuadrático Medio Experimental (Test) : {ecm_test:.4f}")
        print(f"Coeficiente R² Explicativo: {r2_val:.4f}")
        print(f"Estabilidad Numérica del Sistema (Condición): {cond:.2e}")
        print("-" * 40)

    mejor_grado = min(resultados, key=lambda item: item[1])[0]
    print(f"\nReporte del selector automático: El grado más estable determinado empíricamente es {mejor_grado}.")

    return mejor_grado


def graficar(x: np.ndarray, y: np.ndarray, modelo: RegresionPolinomial, titulo: str = "Resumen Gráfico Analítico") -> None:
    """
    Rutina que invoca un plot superpuesto comparativo, graficando un vector distribuido discretamente 
    junto al output del polinomial continuado a lo largo del dominio escalar dado para ilustrar proximidad de error.
    """
    x_suave = np.linspace(min(x), max(x), 300)
    y_suave = modelo.predecir(x_suave)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.5, label="Carga Computacional del Operativo")
    plt.plot(x_suave, y_suave, color='red', linewidth=2, label="Traza Computada Predicha")
    plt.title(titulo)
    plt.legend()
    plt.grid(True)
    plt.show()


def animacion_grados(x: np.ndarray, y: np.ndarray) -> None:
    """
    Visualiza dinámicamente un bucle transitorio de aprendizaje, 
    representando el comportamiento gráfico del sesgo o sobreajuste ante aumento progresivo de P-dimensión.
    """
    plt.ion()
    for grado in range(1, 16):
        modelo = RegresionPolinomial(grado)
        modelo.ajustar(x, y)

        x_suave = np.linspace(min(x), max(x), 300)
        y_suave = modelo.predecir(x_suave)

        plt.clf()
        plt.scatter(x, y, alpha=0.5, label="Set Empírico Obtenido")
        plt.plot(x_suave, y_suave, color='red', label="Intento Adaptativo Actual")
        plt.title(f"Visualización Periódica de Inflexiones Polinomiales (K = {grado})")
        plt.legend()
        plt.pause(1)

    plt.ioff()
    plt.show()


def modo_manual(x: np.ndarray, y: np.ndarray) -> None:
    """
    Habilita un bucle algorítmico semi-bloqueante del input humano, concediendo al analista las facultades
    de ascender, detenerse, o declinar gradientes explícitos y atisbar cómo varían las características visuales del modelo.
    """
    grado = 1
    while True:
        modelo = RegresionPolinomial(grado)
        cond = modelo.ajustar(x, y)

        y_pred = modelo.predecir(x)
        ecm_val = modelo.recm(y, y_pred)
        r2_val = modelo.r2(y, y_pred)

        x_suave = np.linspace(min(x), max(x), 300)
        y_suave = modelo.predecir(x_suave)

        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, alpha=0.5, label="Registros del Hardware Físico")
        plt.plot(x_suave, y_suave, color='red', label=f"Extrapolabilidad Cúbica del Modelo P={grado}")
        plt.title(f"Interacción Libre Sandbox en Entorno Paramétrico. Grado P-Actual: {grado}")
        plt.grid(True)
        plt.legend()
        plt.show(block=False)
        plt.pause(0.5)

        print(f"\nReporte de Rendimiento para Arquitectura Polinomial Nivel: {grado}")
        print(f"Registro de Variabilidad RMSE (RECM): {ecm_val:.4f}")
        print(f"Aclaración de Variación Explicada R²: {r2_val:.4f}")
        print(f"Límite Computacional de Condición Múltiple: {cond:.2e}")

        accion = input("\nControles Analista - Tecla [n] para paso expansivo | Tecla [p] para regresión modal previa | Tecla [q] para destruir vista auxiliar: ").strip().lower()
        plt.close()

        if accion == "n":
            grado += 1
        elif accion == "p" and grado > 1:
            grado -= 1
        elif accion == "q":
            break


# ==========================================
# FUNCIONES DE GESTIÓN DE DATOS E/S
# ==========================================

def cargar_datos() -> Tuple[np.ndarray, np.ndarray]:
    """
    Lógica de enlace controladora de las tuberías y protocolos de recolección informativa previas a su procesado I/O.
    Filtra interactivamente los flujos numéricos ausentes en Excel o levanta entornos aleatorios autogenerados.
    """
    print("\nSeleccione el protocolo de acceso al lote de información subyacente para ingestión analítica.")
    print("1. Orquestar motor de inyección de ruido aleatorio en estructura canónica de control.")
    print("2. Deserializar repositorio excel de rendimiento (.xlsx, .xls).")
    
    opc = input("Ingrese la opción sistémica (1/2): ").strip()
    
    if opc == "2":
        ruta = input("Despliegue el directorio absoluto o identificador local tabular (ej: datos_rendimiento_pc.xlsx): ").strip()
        if not os.path.exists(ruta):
            print("Notificación de Sistema: Fallo estructural. Archivo I/O no ubicado. Tránsito degradado a entorno de experimentación aleatorio estándar.")
            return generar_datos_aleatorios()
            
        try:
            df = pd.read_excel(ruta)
            print("\nReconocidas nomenclaturas de campos estructuradas:", list(df.columns))
            col_x = input("Designación del descriptor causal independiente (X Dimensional): ").strip()
            col_y = input("Designación de vector efecto subsecuente (Y Predictoria): ").strip()
            
            # Sanitización analítica pre-fase 1 para prever colapso estructural
            df = df.dropna(subset=[col_x, col_y])
            
            # Reordenamiento estricto posicional por campo temporal para coherencia al procesar splines de ploteo
            df = df.sort_values(by=col_x)
            x = df[col_x].values
            y = df[col_y].values
            print(f"Notificación de Sistema: Ingesta analítica verificada en local completada. Lote N = {len(x)} procesado exitosamente.")
            return x, y
            
        except Exception as e:
            print(f"Violación de Segmentación Lógica o de Formato detectada: Ingesta interrumpida debido a excepción {e}")
            print("Carga de contingencias. Regresando a la tubería de inserción estocástica predefinida.")
            return generar_datos_aleatorios()
    else:
        return generar_datos_aleatorios()

def generar_datos_aleatorios() -> Tuple[np.ndarray, np.ndarray]:
    """
    Procursor estocástico. Ensambla y fabrica ruidosas curvas artificiales emuladoras
    sustrayendo normal distribucional para enriquecer la comprobación heurística base.
    """
    np.random.seed(42)
    x = np.linspace(-2, 2, 100)
    # Función madre estructural (polinómica estándar) distorsionada
    y = 1.5 * x**3 - 0.5 * x**2 + 2 * x + 1 + np.random.normal(0, 2.5, 100)
    return x, y


# ==========================================
# AJUSTE CON GRADO SELECCIONADO POR CONSOLA
# ==========================================

def ajustar_grado_especifico(x: np.ndarray, y: np.ndarray) -> None:
    """
    Permite al analista introducir un grado polinomial concreto por consola,
    ajusta el modelo con ese grado sobre el dataset completo y reporta las
    métricas de calidad (RECM y R²) junto a la gráfica resultante.

    Args:
        x (np.ndarray): Vector de la variable independiente.
        y (np.ndarray): Vector de la variable dependiente (observaciones reales).
    """
    entrada = input("Ingrese el grado polinomial deseado (entero >= 1): ").strip()

    try:
        grado = int(entrada)
        if grado < 1:
            print("Notificación de Sistema: El grado debe ser un entero mayor o igual a 1.")
            return
    except ValueError:
        print("Notificación de Sistema: Entrada inválida. Debe introducir un número entero.")
        return

    modelo = RegresionPolinomial(grado)
    cond = modelo.ajustar(x, y)
    y_pred = modelo.predecir(x)

    try:
        recm_val = modelo.recm(y, y_pred)
    except ValueError as e:
        print(f"Notificación de Sistema: {e}")
        return

    r2_val = modelo.r2(y, y_pred)

    print(f"\n--- Métricas del Modelo Polinomial de Grado {grado} ---")
    print(f"Raíz del Error Cuadrático Medio (RECM): {recm_val:.4f}")
    print(f"Coeficiente R²:                         {r2_val:.4f}")
    print(f"Número de condición de la matriz:       {cond:.2e}")
    print("------------------------------------------------------")

    graficar(x, y, modelo, f"Ajuste Polinomial de Grado {grado} (Selección Manual)")


# ==========================================
# RUTINA DE EJECUCIÓN PRINCIPAL
# ==========================================

def main() -> None:
    """
    Orquesta maestra que encierra el alcance algorítmico y permite la recursividad
    mediante loop operacional. Brindando un puerto aislado y coherente de control top-level de abstracción para ejecución.
    """
    # 1. Pipeline Inicial: Carga única global referencial
    x_global, y_global = cargar_datos()

    # 2. Iteración Continua del Subproceso de Menú de Análisis Paramétrico
    while True:
        print("\n===== MÓDULO COMPUTACIONAL: ENGINE DE REGRESIÓN NO LINEAL =====")
        print("1. Ejecutar batería de modelo completo y auto-determinar óptimo métrico.")
        print("2. Visualizar animación algorítmica de encaje progresivo dinámico.")
        print("3. Modo de interacción granular (Sand-Box manual paso a paso).")
        print("4. Ajustar modelo con grado específico seleccionado por consola.")
        print("5. Invocar rutina de I/O de relectura de datasets para transición causal diferente.")
        print("6. Detener flujo de servicio principal e invocar exit(0).")

        opcion = input("Seleccione el comando estructurado a lanzar: ").strip()

        if opcion == "1":
            mejor_grado = comparar_modelos(x_global, y_global)
            modelo = RegresionPolinomial(mejor_grado)
            modelo.ajustar(x_global, y_global)
            graficar(x_global, y_global, modelo, f"Revisión Perimetral Excitada (Grado Computacional Estático= {mejor_grado})")

        elif opcion == "2":
            animacion_grados(x_global, y_global)

        elif opcion == "3":
            modo_manual(x_global, y_global)

        elif opcion == "4":
            ajustar_grado_especifico(x_global, y_global)

        elif opcion == "5":
            x_global, y_global = cargar_datos()

        elif opcion == "6":
            print("Liberando recursos y subprocesos, cerrando de manera segura el análisis matemático exploratorio...")
            break

        else:
            print("Notificación de Sistema: Infracción de lectura. Operando introducido con sintaxis inválida.")

if __name__ == "__main__":
    main()