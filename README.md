# La mejor recta como un problema de optimización

Proyecto final de **Cálculo Multivariable**, Universidad Cooperativa de Colombia (2026B), San Juan de Pasto.

**Pregunta:** ¿cuánto baja la temperatura media al subir en Colombia?
Ajustamos el modelo **T = a·h + b** a 13 estaciones del IDEAM. Para hacerlo, minimizamos la función de dos variables

$$S(a,b)=\sum_{i=1}^{13}\left(T_i - a\,h_i - b\right)^2$$

## Resultado

| Concepto | Valor |
|---|---|
| Pendiente óptima a* | −6,1677 °C/km |
| Intercepto óptimo b* | 29,9003 °C |
| Error mínimo S(a\*, b\*) | 8,8249 °C² |
| R² | 0,9785 (la altitud explica el 97,9 % de la variación) |
| RMSE | 0,82 °C |
| Hessiano | constante, det H = 549,59 > 0 y S_aa = 88,34 > 0, así que el punto es un mínimo global |

En palabras: cada 1000 m de subida, la temperatura media baja unos 6,2 °C.

## Contenido del repositorio

```
index.html                     Laboratorio interactivo (página web)
codigo/
  regresion_altitud_temperatura.py   Programa en Python: método propio + comparación con librerías
  diagrama_flujo.dot                 Diagrama de flujo del algoritmo (Graphviz)
  figuras/                           Las 14 figuras del informe
  resultados.json                    Todos los números calculados
datos/
  datos_ciudades.csv                 13 estaciones: altitud (m) y temperatura media anual (°C)
requirements.txt
```

## Laboratorio interactivo (`index.html`)

Es una sola página, sin instalación y sin dependencias: se abre con doble clic en cualquier navegador. Cada recta es un punto (a, b), y al moverlo se actualiza todo al mismo tiempo:

1. **Datos y recta:** residuos y sus cuadrados eᵢ². El área total de los cuadrados es S.
2. **Función de error:**
   - curvas de nivel, con el vector −∇S perpendicular a la curva de nivel;
   - superficie 3D (paraboloide elíptico) que se puede rotar, con las dos trazas que pasan por el punto.
3. **El cálculo en vivo:**
   - derivadas parciales evaluadas en el punto;
   - ecuaciones normales y solución por Cramer;
   - Hessiano, sus autovalores y por qué las elipses son alargadas.
4. **Trazas** z = S(a, b fijo) y z = S(a fijo, b), con la recta tangente cuya pendiente es la derivada parcial.
5. **Descenso por gradiente** con paso η = 1/λ_max. Llega al mismo mínimo que la solución analítica.
6. **Predicción** de la temperatura para cualquier altitud, y la tabla de residuos.

Para publicarla gratis con GitHub Pages: *Settings → Pages → Deploy from a branch → `main` / root*.

## Ejecutar el código en Python

```bash
pip install -r requirements.txt
cd codigo
python regresion_altitud_temperatura.py
```

El programa:
- calcula a\* y b\* con las ecuaciones normales deducidas a mano;
- verifica el Hessiano de forma analítica y numérica;
- compara con `numpy.polyfit`, `scipy.stats.linregress` y `numpy.linalg.lstsq` (la diferencia es del orden de 10⁻¹⁴);
- analiza los residuos;
- hace los análisis adicionales A–D: sensibilidad, influencia de puntos, modelo cuadrático y número de datos;
- regenera las figuras.

## Datos

Normales climatológicas del IDEAM, periodo 1991–2020. Bucaramanga, Manizales, Neiva, Tunja e Ipiales usan 1981–2010. Las altitudes son las de cada estación.

## Integrantes

Juan Manuel Gallardo Valenzuela · David Esteban Campiño Igua · Jaider Julian Chindoy Caicedo · Santiago Alejandro Campoverde · Manuel Alejandro Villareal Gil

Docente: M.Sc. Alejandro Molina
