# Comparación RAG vs No-RAG

Fecha: 2025-09-02 16:24:24

## Comparativa Global de Métricas

| Modelo     | Método     | Rouge-1   | Rouge-2   | Rouge-L   | Similitud Semántica   | Tiempo (s)   |
|:-----------|:-----------|:----------|:----------|:----------|:----------------------|:-------------|
| gpt-4o     | CON RAG    | 0.2861    | 0.1834    | 0.2322    | 0.7318                | 19.65        |
| gpt-4o     | SIN RAG    | 0.1577    | 0.0614    | 0.1128    | 0.6236                | 8.27         |
| gpt-4o     | DIFERENCIA | +0.1283   | +0.1220   | +0.1194   | +0.1081               | +11.38       |
|            |            |           |           |           |                       |              |
| gpt-5-mini | CON RAG    | 0.2900    | 0.1787    | 0.2304    | 0.6542                | 32.11        |
| gpt-5-mini | SIN RAG    | 0.1077    | 0.0403    | 0.0761    | 0.6403                | 24.44        |
| gpt-5-mini | DIFERENCIA | +0.1823   | +0.1384   | +0.1543   | +0.0138               | +7.67        |
|            |            |           |           |           |                       |              |
| o4-mini    | CON RAG    | 0.3131    | 0.1802    | 0.2405    | 0.6810                | 23.36        |
| o4-mini    | SIN RAG    | 0.1657    | 0.0587    | 0.1133    | 0.6496                | 11.42        |
| o4-mini    | DIFERENCIA | +0.1474   | +0.1215   | +0.1273   | +0.0314               | +11.95       |
|            |            |           |           |           |                       |              |

## Análisis de Sobre-Respuesta del RAG

### Modelo: gpt-4o

- **Total de preguntas evaluadas:** 100
- **Casos donde Sin RAG > Con RAG:** 37 (37.0%)
- **Diferencia promedio en estos casos:** 0.1143

**Top 3 casos donde Sin RAG supera significativamente a RAG:**

**Pregunta:** ¿Puede un procesador ejecutar directamente un programa en lenguaje ensamblador?

**Ground Truth:** No, un procesador (CPU) puede interpretar y ejecutar directamente las instrucciones de un programa solo en lenguaje máquina. El lenguaje ensamblador requiere un ensamblador para ser traducido a lenguaje máquina antes de su ejecución.

**Respuesta RAG:** Para abordar tu pregunta sobre si un procesador puede ejecutar directamente un programa en lenguaje ensamblador, consideremos algunos puntos clave sobre cómo funcionan los procesadores y los lenguajes...

**Respuesta Sin RAG:** No, un procesador no puede ejecutar directamente un programa en lenguaje ensamblador. El lenguaje ensamblador es una representación simbólica de las instrucciones de máquina, diseñada para ser más com...

**Similitud RAG:** 0.7283
**Similitud Sin RAG:** 0.9113

---

**Pregunta:** ¿Por qué un planificador de procesos tiene sentido en un SO multiprogramado?

**Ground Truth:** Un planificador de procesos tiene sentido en un SO multiprogramado porque es responsable de decidir qué proceso se ejecutará en la CPU y por cuánto tiempo, gestionando el uso concurrente de los recursos y optimizando el rendimiento del sistema.

**Respuesta RAG:** Para entender por qué un planificador de procesos tiene sentido en un sistema operativo multiprogramado, es útil reflexionar sobre el papel que juega este tipo de sistema operativo. En un entorno mult...

**Respuesta Sin RAG:** En un sistema operativo multiprogramado, un planificador de procesos es esencial para gestionar eficientemente el uso del procesador entre múltiples programas en ejecución. Aquí tienes algunas razones...

**Similitud RAG:** 0.8853
**Similitud Sin RAG:** 0.8857

---

**Pregunta:** Según el ejercicio 12c del Tema 2, ¿suspender o bloquear un proceso implica un cambio de contexto?

**Ground Truth:** Sí, suspender o bloquear un proceso (c) implica un cambio de contexto, ya que el estado del proceso actual debe guardarse y se debe cargar el contexto de otro proceso (o del sistema operativo).

**Respuesta RAG:** Para entender si suspender o bloquear un proceso implica un cambio de contexto, vamos a explorar algunos conceptos fundamentales relacionados con el manejo de procesos en un sistema operativo.

1. **D...

**Respuesta Sin RAG:** En el contexto de los sistemas operativos y de la gestión de procesos, suspender o bloquear un proceso puede implicar un cambio de contexto, pero esto depende de los detalles específicos de cómo se ma...

**Similitud RAG:** 0.8522
**Similitud Sin RAG:** 0.8837

---

### Modelo: gpt-5-mini

- **Total de preguntas evaluadas:** 100
- **Casos donde Sin RAG > Con RAG:** 52 (52.0%)
- **Diferencia promedio en estos casos:** 0.1729

**Top 3 casos donde Sin RAG supera significativamente a RAG:**

**Pregunta:** ¿Por qué un planificador de procesos tiene sentido en un SO multiprogramado?

**Ground Truth:** Un planificador de procesos tiene sentido en un SO multiprogramado porque es responsable de decidir qué proceso se ejecutará en la CPU y por cuánto tiempo, gestionando el uso concurrente de los recursos y optimizando el rendimiento del sistema.

**Respuesta RAG:** Buena pregunta. No voy a darte la respuesta final: en cambio te propongo preguntas y una estructura para que construyas el razonamiento por ti mismo y verifique si tiene sentido un planificador en cad...

**Respuesta Sin RAG:** Respuesta clara y pedagógica:

Qué es un planificador de procesos (scheduler)
- Es el componente del sistema operativo que decide qué proceso debe ejecutar el procesador en cada momento y por cuánto t...

**Similitud RAG:** 0.6701
**Similitud Sin RAG:** 0.8757

---

**Pregunta:** ¿Qué es la fragmentación interna en un sistema paginado?

**Ground Truth:** La fragmentación interna se produce en un sistema de gestión de memoria paginado cuando el tamaño de los datos de un proceso no es un múltiplo exacto del tamaño de la página, dejando espacio sin usar dentro de la última página asignada a ese proceso. El ejercicio 21 del Tema 2 pide calcularla.

**Respuesta RAG:** No encontré en los materiales disponibles una definición de “fragmentación interna” en un sistema paginado. Puedo, eso sí, ayudarte a construirla paso a paso sin darte la solución final.

Preguntas pa...

**Respuesta Sin RAG:** Respuesta clara y pedagógica:

- Definición básica
  La fragmentación interna en un sistema paginado es la memoria desperdiciada dentro de los marcos (frames) o páginas asignadas porque los procesos s...

**Similitud RAG:** 0.7670
**Similitud Sin RAG:** 0.8570

---

**Pregunta:** ¿Qué es un enlazador y por qué es necesario?

**Ground Truth:** El enlazador es una utilidad que combina diferentes módulos de código (objetos) y bibliotecas para producir un programa ejecutable final. Es necesario para resolver referencias a símbolos externos y funciones que pueden estar definidas en otros archivos o bibliotecas.

**Respuesta RAG:** No encontré información sobre qué es un “enlazador” ni su definición en los materiales disponibles. Puedo ayudarte a razonar y guiarte para que construyas tu propia respuesta. ¿Te parece si intentas p...

**Respuesta Sin RAG:** Respuesta clara y pedagógica:

Qué es un enlazador (linker)
- El enlazador es una herramienta del proceso de construcción de software que toma varios archivos objeto (generados por el compilador) y bi...

**Similitud RAG:** 0.5844
**Similitud Sin RAG:** 0.8457

---

### Modelo: o4-mini

- **Total de preguntas evaluadas:** 100
- **Casos donde Sin RAG > Con RAG:** 53 (53.0%)
- **Diferencia promedio en estos casos:** 0.1344

**Top 3 casos donde Sin RAG supera significativamente a RAG:**

**Pregunta:** ¿Qué es la fragmentación interna en un sistema paginado?

**Ground Truth:** La fragmentación interna se produce en un sistema de gestión de memoria paginado cuando el tamaño de los datos de un proceso no es un múltiplo exacto del tamaño de la página, dejando espacio sin usar dentro de la última página asignada a ese proceso. El ejercicio 21 del Tema 2 pide calcularla.

**Respuesta RAG:** Para que formules tú mismo qué es la fragmentación interna en paginación, te propongo pensar en estos puntos:

1. Asignación por páginas de tamaño fijo  
   - Imagina que la memoria se divide en bloqu...

**Respuesta Sin RAG:** La fragmentación interna en un sistema paginado es el espacio desperdiciado que queda dentro de un marco (frame) de memoria porque la página que se le asigna no rellena exactamente todo el espacio dis...

**Similitud RAG:** 0.7910
**Similitud Sin RAG:** 0.9121

---

**Pregunta:** ¿Puede un procesador ejecutar directamente un programa en lenguaje ensamblador?

**Ground Truth:** No, un procesador (CPU) puede interpretar y ejecutar directamente las instrucciones de un programa solo en lenguaje máquina. El lenguaje ensamblador requiere un ensamblador para ser traducido a lenguaje máquina antes de su ejecución.

**Respuesta RAG:** Bien, antes de llegar a una conclusión directa, conviene que reflexiones sobre estos puntos:

1. ¿En qué formato “habla” realmente el procesador?  
   - Piensa: ¿prefiere texto con mnemónicos (por eje...

**Respuesta Sin RAG:** No. Un procesador sólo entiende instrucciones en forma binaria (código máquina), no textos en lenguaje ensamblador. El ensamblador (assembler) es la herramienta que traduce cada mnemónico de ensamblad...

**Similitud RAG:** 0.6208
**Similitud Sin RAG:** 0.8682

---

**Pregunta:** ¿Qué son los tokens en el contexto de la compilación?

**Ground Truth:** Los tokens son los componentes léxicos en los que se descompone el programa fuente durante la fase de análisis léxico de la compilación. El ejercicio 2 del Tema 3 pregunta si es lo mismo que un lexema.

**Respuesta RAG:** Veamos cómo ir construyendo por ti mismo la idea de “token” en un compilador. Te sugiero reflexionar sobre estas pistas y preguntas:

1. Papel del analizador léxico  
   - ¿Qué parte del compilador se...

**Respuesta Sin RAG:** En compilación, los tokens son las unidades mínimas de significado que produce el análisis léxico (o “scanner”) a partir del código fuente. A grandes rasgos:

1. Rol en el flujo de compilación  
   - ...

**Similitud RAG:** 0.5553
**Similitud Sin RAG:** 0.8524

---

## Conclusiones

1. **Impacto del RAG**: Compare las métricas globales para ver si el RAG mejora o perjudica el rendimiento general.
2. **Casos de sobre-respuesta**: Analice los casos donde Sin RAG supera a RAG para identificar patrones.
3. **Eficiencia**: Compare los tiempos de respuesta para evaluar el costo computacional del RAG.
