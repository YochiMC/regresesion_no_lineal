import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import Tuple, List, Optional


class RegresionPolinomial:
    """
    Regresión polinomial mediante Mínimos Cuadrados Ordinarios (método matricial).

    Attributes:
        grado (int): Grado máximo del polinomio.
        coef (Optional[np.ndarray]): Coeficientes calculados tras el ajuste.
    """

    def __init__(self, grado: int) -> None:
        self.grado = grado
        self.coef: Optional[np.ndarray] = None

    def construir_phi(self, x: np.ndarray) -> np.ndarray:
        """
        Construye la matriz de diseño (Vandermonde) de tamaño (n, grado+1).

        Args:
            x (np.ndarray): Variable independiente.

        Returns:
            np.ndarray: Matriz de diseño con columnas [1, x, x², ..., x^grado].
        """
        return np.column_stack([x**j for j in range(self.grado + 1)])

    def ajustar(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calcula los coeficientes óptimos resolviendo la ecuación normal:
        theta = (Phi^T Phi)^-1 (Phi^T y)

        Args:
            x (np.ndarray): Variable independiente.
            y (np.ndarray): Variable dependiente (observaciones reales).

        Returns:
            float: Número de condición de la matriz (Phi^T Phi), útil para
                   detectar problemas de inestabilidad numérica.
        """
        Phi = self.construir_phi(x)
        C = Phi.T @ Phi
        B = Phi.T @ y

        self.coef = np.linalg.solve(C, B)
        return np.linalg.cond(C)

    def predecir(self, x: np.ndarray) -> np.ndarray:
        """
        Genera predicciones para los valores dados de x.

        Args:
            x (np.ndarray): Valores de entrada a predecir.

        Returns:
            np.ndarray: Valores predichos por el modelo.

        Raises:
            ValueError: Si el modelo no ha sido ajustado previamente.
        """
        if self.coef is None:
            raise ValueError("El modelo no ha sido ajustado. Llame a ajustar() primero.")
        Phi = self.construir_phi(x)
        return Phi @ self.coef

    def recm(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Raíz del Error Cuadrático Medio ajustado por grados de libertad (RECM / RMSE).
        Aplica el estimador insesgado de la varianza residual según el Teorema de Gauss-Markov,
        usando (n - m - 1) como denominador, donde m es el grado del polinomio.

        Args:
            y (np.ndarray): Valores reales.
            y_pred (np.ndarray): Valores predichos por el modelo.

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
        Calcula el coeficiente de determinación R².

        Args:
            y (np.ndarray): Valores reales.
            y_pred (np.ndarray): Valores predichos.

        Returns:
            float: R² en el rango (-inf, 1.0]. Un valor de 1.0 indica ajuste perfecto.
        """
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        return 1.0 - (ss_res / ss_total)

    def mostrar_coeficientes(self) -> None:
        """
        Imprime una tabla formateada con los coeficientes del modelo ajustado.
        Cada fila muestra el término polinomial y su coeficiente correspondiente.

        Raises:
            ValueError: Si el modelo no ha sido ajustado previamente.
        """
        if self.coef is None:
            raise ValueError("El modelo no ha sido ajustado. Llame a ajustar() primero.")

        sep = "-" * 42
        print(f"\nCoeficientes - Grado {self.grado}  (m = {self.grado})")
        print(sep)
        print(f"  {'Indice':<8} {'Termino':<10} {'Coeficiente':>18}")
        print(sep)
        for j, c in enumerate(self.coef):
            indice  = f"a{j}"
            termino = "1" if j == 0 else ("x" if j == 1 else f"x^{j}")
            print(f"  {indice:<8} {termino:<10} {c:>18.6f}")
        print(sep + "\n")


# ==========================================
# FUNCIONES DE VISUALIZACIÓN Y ANÁLISIS
# ==========================================

def obtener_metricas(modelo: RegresionPolinomial, x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Calcula RECM, R2 y número de condición para un modelo y datos dados."""
    y_pred = modelo.predecir(x)
    cond = np.linalg.cond(modelo.construir_phi(x).T @ modelo.construir_phi(x))
    try:
        recm_val = modelo.recm(y, y_pred)
    except ValueError:
        recm_val = np.nan
    r2_val = modelo.r2(y, y_pred)
    return recm_val, r2_val, cond

def comparar_modelos(x: np.ndarray, y: np.ndarray, max_grado: int = 15) -> int:
    """Evalúa modelos de grado 1 a max_grado y retorna el óptimo."""
    split = int(0.8 * len(x))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    resultados = []
    print(f"\n=== COMPARACIÓN DE MODELOS (1 a {max_grado}) ===\n")

    for grado in range(1, max_grado + 1):
        modelo = RegresionPolinomial(grado)
        modelo.ajustar(x_train, y_train)
        
        recm_train, _, _ = obtener_metricas(modelo, x_train, y_train)
        recm_test, r2_test, cond = obtener_metricas(modelo, x_test, y_test)
        
        if not np.isnan(recm_test):
            resultados.append((grado, recm_test))

        print(f"Grado {grado}: RECM Test: {recm_test:.4f} | R²: {r2_test:.4f} | Cond: {cond:.2e}")

    mejor_grado = min(resultados, key=lambda x: x[1])[0]
    print(f"\nGrado óptimo sugerido: {mejor_grado}")
    return mejor_grado


def graficar(x: np.ndarray, y: np.ndarray, modelo: RegresionPolinomial, titulo: str = "Ajuste Polinomial") -> None:
    """
    Grafica los datos originales y la curva ajustada por el modelo.

    Args:
        x (np.ndarray): Variable independiente.
        y (np.ndarray): Valores reales.
        modelo (RegresionPolinomial): Modelo ya ajustado.
        titulo (str): Título del gráfico.
    """
    x_suave = np.linspace(min(x), max(x), 300)
    y_suave = modelo.predecir(x_suave)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.5, label="Datos observados")
    plt.plot(x_suave, y_suave, color='red', linewidth=2, label="Curva ajustada")
    plt.title(titulo)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()


def animacion_grados(x: np.ndarray, y: np.ndarray) -> None:
    """
    Muestra una animación del ajuste polinomial para grados del 1 al 15,
    útil para visualizar el efecto del sobreajuste al aumentar el grado.

    Args:
        x (np.ndarray): Variable independiente.
        y (np.ndarray): Variable dependiente.
    """
    plt.ion()
    for grado in range(1, 16):
        modelo = RegresionPolinomial(grado)
        modelo.ajustar(x, y)

        x_suave = np.linspace(min(x), max(x), 300)
        y_suave = modelo.predecir(x_suave)

        plt.clf()
        plt.scatter(x, y, alpha=0.5, label="Datos observados")
        plt.plot(x_suave, y_suave, color='red', label=f"Grado {grado}")
        plt.title(f"Ajuste polinomial — Grado {grado}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.pause(1)

    plt.ioff()
    plt.show()


def analizar_grado(x: np.ndarray, y: np.ndarray, grado: int, mostrar_tabla: bool = True):
    """Ajusta, calcula métricas e imprime resultados para un grado específico."""
    modelo = RegresionPolinomial(grado)
    cond = modelo.ajustar(x, y)
    recm, r2, _ = obtener_metricas(modelo, x, y)
    
    if mostrar_tabla:
        modelo.mostrar_coeficientes()
        print(f"RECM: {recm:.4f} | R²: {r2:.4f} | Condición: {cond:.2e}")
    
    return modelo, recm, r2, cond

def modo_manual(x: np.ndarray, y: np.ndarray) -> None:
    """Navegación interactiva de grados polinomiales."""
    entrada = input("Grado inicial (Enter para 1): ").strip()
    grado = max(1, int(entrada)) if entrada.isdigit() else 1

    while True:
        modelo, recm, r2, cond = analizar_grado(x, y, grado, mostrar_tabla=False)
        
        # Plot rápido
        x_s = np.linspace(x.min(), x.max(), 300)
        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, alpha=0.5)
        plt.plot(x_s, modelo.predecir(x_s), 'r', label=f"Grado {grado}")
        plt.title(f"Manual: Grado {grado} | RECM: {recm:.4f}")
        plt.legend(); plt.show(block=False); plt.pause(0.5)

        print(f"\nGrado {grado} -> RECM: {recm:.4f} | R²: {r2:.4f} | Cond: {cond:.2e}")
        acc = input("[n] Siguiente | [p] Anterior | [q] Salir: ").lower()
        plt.close()

        if acc == 'n': grado += 1
        elif acc == 'p' and grado > 1: grado -= 1
        elif acc == 'q': break


# ==========================================
# FUNCIONES DE CARGA DE DATOS
# ==========================================

def cargar_datos() -> Tuple[np.ndarray, np.ndarray]:
    """
    Solicita al usuario la fuente de datos: archivo Excel o datos aleatorios.
    En caso de error, cae automáticamente a datos aleatorios.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays (x, y) listos para usar.
    """
    print("\nSeleccione la fuente de datos:")
    print("1. Generar datos aleatorios")
    print("2. Cargar archivo Excel (.xlsx, .xls)")

    opc = input("Opción (1/2): ").strip()

    if opc == "2":
        ruta = input("Ruta del archivo (ej: datos.xlsx): ").strip()
        if not os.path.exists(ruta):
            print("Archivo no encontrado. Se usarán datos aleatorios.")
            return generar_datos_aleatorios()

        try:
            df = pd.read_excel(ruta)
            print("\nColumnas disponibles:", list(df.columns))
            col_x = input("Columna para X: ").strip()
            col_y = input("Columna para Y: ").strip()

            df = df.dropna(subset=[col_x, col_y])
            df = df.sort_values(by=col_x)
            x = df[col_x].values
            y = df[col_y].values
            print(f"Datos cargados correctamente. N = {len(x)}")
            return x, y

        except Exception as e:
            print(f"Error al leer el archivo: {e}")
            print("Se usarán datos aleatorios.")
            return generar_datos_aleatorios()

    return generar_datos_aleatorios()


def generar_datos_aleatorios() -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera datos sintéticos basados en un polinomio cúbico con ruido gaussiano.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays (x, y) con 100 puntos en [-2, 2].
    """
    np.random.seed(42)
    x = np.linspace(-2, 2, 100)
    y = 1.5 * x**3 - 0.5 * x**2 + 2 * x + 1 + np.random.normal(0, 2.5, 100)
    return x, y


# ==========================================
# AJUSTE CON GRADO ESPECIFICO
# ==========================================

def ajustar_grado_especifico(x: np.ndarray, y: np.ndarray) -> None:
    """Solicita un grado y muestra análisis completo."""
    ent = input("Grado deseado (entero >= 1): ").strip()
    if ent.isdigit() and int(ent) >= 1:
        modelo, _, _, _ = analizar_grado(x, y, int(ent))
        graficar(x, y, modelo, f"Grado {ent}")
    else:
        print("Entrada inválida.")


# ==========================================
# RUTINA DE EJECUCION PRINCIPAL
# ==========================================

def main() -> None:
    """
    Punto de entrada principal. Carga los datos y presenta el menú de análisis.
    """
    x_global, y_global = cargar_datos()

    while True:
        print("\n=== REGRESIÓN POLINOMIAL ===")
        print("1. Comparar y seleccionar óptimo")
        print("2. Animación (1-15)")
        print("3. Modo interactivo")
        print("4. Grado específico (Tabla)")
        print("5. Recargar datos")
        print("6. Salir")

        opc = input("\nSelección: ").strip()

        if opc == "1":
            grado = comparar_modelos(x_global, y_global)
            modelo, _, _, _ = analizar_grado(x_global, y_global, grado)
            graficar(x_global, y_global, modelo, f"Óptimo: Grado {grado}")
        elif opc == "2": animacion_grados(x_global, y_global)
        elif opc == "3": modo_manual(x_global, y_global)
        elif opc == "4": ajustar_grado_especifico(x_global, y_global)
        elif opc == "5": x_global, y_global = cargar_datos()
        elif opc == "6": break


if __name__ == "__main__":
    main()