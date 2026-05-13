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


# ==========================================
# FUNCIONES DE VISUALIZACIÓN Y ANÁLISIS
# ==========================================

def comparar_modelos(x: np.ndarray, y: np.ndarray) -> int:
    """
    Evalúa modelos polinomiales de grado 1 a 15 usando un split 80/20 (hold-out),
    e identifica el grado con menor RECM en el conjunto de prueba.

    Args:
        x (np.ndarray): Variable independiente.
        y (np.ndarray): Variable dependiente.

    Returns:
        int: Grado óptimo según la RECM en el conjunto de prueba.
    """
    n = len(x)
    split = int(0.8 * n)

    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    resultados: List[Tuple[int, float]] = []

    print("\n=== COMPARACIÓN DE MODELOS (grados 1 a 15) ===\n")

    for grado in range(1, 16):
        modelo = RegresionPolinomial(grado)
        cond = modelo.ajustar(x_train, y_train)

        y_train_pred = modelo.predecir(x_train)
        y_test_pred = modelo.predecir(x_test)

        try:
            recm_train = modelo.recm(y_train, y_train_pred)
            recm_test = modelo.recm(y_test, y_test_pred)
        except ValueError as e:
            print(f"Grado {grado}: {e}")
            continue

        r2_val = modelo.r2(y_test, y_test_pred)
        resultados.append((grado, recm_test))

        print(f"Grado {grado}")
        print(f"  RECM Train: {recm_train:.4f}")
        print(f"  RECM Test:  {recm_test:.4f}")
        print(f"  R²:         {r2_val:.4f}")
        print(f"  Condición:  {cond:.2e}")
        print("-" * 40)

    mejor_grado = min(resultados, key=lambda item: item[1])[0]
    print(f"\nGrado óptimo seleccionado automáticamente: {mejor_grado}")

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


def modo_manual(x: np.ndarray, y: np.ndarray) -> None:
    """
    Modo interactivo: permite navegar entre grados polinomiales con las teclas
    [n] (subir), [p] (bajar) y [q] (salir). Acepta un grado inicial por consola.

    Args:
        x (np.ndarray): Variable independiente.
        y (np.ndarray): Variable dependiente.
    """
    entrada = input("Ingrese el grado inicial (entero >= 1, Enter para comenzar en 1): ").strip()
    try:
        grado = max(1, int(entrada)) if entrada else 1
    except ValueError:
        print("Entrada inválida. Se usará grado 1.")
        grado = 1

    while True:
        modelo = RegresionPolinomial(grado)
        cond = modelo.ajustar(x, y)

        y_pred = modelo.predecir(x)

        try:
            recm_val = modelo.recm(y, y_pred)
            recm_str = f"{recm_val:.4f}"
        except ValueError as e:
            recm_str = f"N/A ({e})"

        r2_val = modelo.r2(y, y_pred)

        x_suave = np.linspace(min(x), max(x), 300)
        y_suave = modelo.predecir(x_suave)

        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, alpha=0.5, label="Datos observados")
        plt.plot(x_suave, y_suave, color='red', label=f"Grado {grado}")
        plt.title(f"Ajuste polinomial — Grado {grado}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.legend()
        plt.show(block=False)
        plt.pause(0.5)

        print(f"\nGrado: {grado}")
        print(f"  RECM:      {recm_str}")
        print(f"  R²:        {r2_val:.4f}")
        print(f"  Condición: {cond:.2e}")

        accion = input("\n[n] Subir grado  [p] Bajar grado  [q] Salir: ").strip().lower()
        plt.close()

        if accion == "n":
            grado += 1
        elif accion == "p" and grado > 1:
            grado -= 1
        elif accion == "q":
            break


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
# RUTINA DE EJECUCIÓN PRINCIPAL
# ==========================================

def main() -> None:
    """
    Punto de entrada principal. Carga los datos y presenta el menú de análisis.
    """
    x_global, y_global = cargar_datos()

    while True:
        print("\n===== REGRESIÓN POLINOMIAL =====")
        print("1. Comparar grados y seleccionar el óptimo automáticamente.")
        print("2. Animación del ajuste para grados 1 a 15.")
        print("3. Modo interactivo (navegar entre grados manualmente).")
        print("4. Cargar nuevo dataset.")
        print("5. Salir.")

        opcion = input("\nOpción: ").strip()

        if opcion == "1":
            mejor_grado = comparar_modelos(x_global, y_global)
            modelo = RegresionPolinomial(mejor_grado)
            modelo.ajustar(x_global, y_global)
            graficar(x_global, y_global, modelo, f"Ajuste óptimo — Grado {mejor_grado}")

        elif opcion == "2":
            animacion_grados(x_global, y_global)

        elif opcion == "3":
            modo_manual(x_global, y_global)

        elif opcion == "4":
            x_global, y_global = cargar_datos()

        elif opcion == "5":
            print("Saliendo...")
            break

        else:
            print("Opción no válida. Ingrese un número del 1 al 5.")


if __name__ == "__main__":
    main()