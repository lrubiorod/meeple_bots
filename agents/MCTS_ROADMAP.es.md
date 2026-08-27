# Hoja de ruta para el desarrollo de MCTS

[English](MCTS_ROADMAP.md) | [**Español**](MCTS_ROADMAP.es.md)

Este documento recoge posibles direcciones para evolucionar los agentes de búsqueda de Meeple
Bots. Es un horizonte, no un calendario comprometido: cada etapa debe justificarse con mediciones
antes de incorporarse al proyecto.

La hoja de ruta avanza deliberadamente desde mejoras observables del MCTS clásico hasta búsquedas
con azar e información imperfecta, y solo después llega a políticas y funciones de valor
aprendidas. Así, cada concepto nuevo resulta lo bastante pequeño como para entenderlo y probarlo
de manera independiente.

## Principios rectores

- Mantener las reglas de los juegos independientes de los algoritmos de búsqueda.
- No exponer nunca el estado oculto autoritativo a un agente de información imperfecta.
- Comparar agentes tanto por su fuerza de juego como por su coste en tiempo real. Una iteración no
  cuesta lo mismo en distintos juegos o variantes de búsqueda.
- Añadir una sola idea de búsqueda cada vez y conservar un baseline sencillo para comparar.
- Preferir capacidades explícitas frente a suposiciones en tiempo de ejecución. Las combinaciones
  no compatibles deben fallar claramente, idealmente en tiempo de compilación.
- Hacer que los experimentos deterministas sean reproducibles a partir de su semilla y
  configuración.
- Tratar los algoritmos avanzados como agentes o políticas opcionales, en lugar de complicar
  continuamente la implementación básica de MCTS.

## Etapa 0: establecer baselines fiables

### Objetivo

Hacer medible el comportamiento del MCTS actual antes de modificar su algoritmo.

### Posible trabajo

- Registrar por decisión el número de nodos creados, profundidad máxima del árbol, acciones de
  rollout, rollouts terminales y cortes evaluados mediante heurística.
- Registrar visitas y utilidades medias de las acciones raíz, además de la acción seleccionada.
- Permitir presupuestos de búsqueda expresados en iteraciones y en tiempo real.
- Añadir configuraciones de benchmark para Tres en raya, Connect Four y Boop.
- Comparar agentes por tasa de victoria, asiento, tiempo de decisión, nodos por segundo y memoria.
- Validar las decisiones de Tres en raya frente a su árbol de juego resuelto.

### Señal de finalización

Los experimentos pueden explicar no solo qué agente gana, sino también cómo emplea su presupuesto
de búsqueda.

## Etapa 1: hacer modulares las políticas del MCTS clásico

### Objetivo

Crear puntos de extensión explícitos sin cambiar el comportamiento UCT predeterminado.

### Posible trabajo

- Separar la política de selección del árbol del almacenamiento de nodos.
- Separar la política de rollout del evaluador de corte.
- Separar de la selección UCT la regla que elige la acción final en la raíz.
- Conservar como implementaciones baseline el rollout aleatorio uniforme, la evaluación neutral
  en el corte y la selección final de la acción más visitada.
- Permitir que cada política consulte únicamente las capacidades del juego que realmente necesita.

### Conceptos que explorar

- `TreePolicy`: selecciona un hijo durante el recorrido del árbol.
- `ExpansionPolicy`: selecciona qué acción aún no expandida se añade.
- `RolloutPolicy`: selecciona acciones simuladas fuera del árbol.
- `CutoffEvaluator`: estima estados no terminales al alcanzar el límite del rollout.
- `RootSelectionPolicy`: elige la acción real después de la búsqueda.

### Señal de finalización

Se pueden probar políticas alternativas sin duplicar el bucle MCTS completo.

## Etapa 2: mejorar la calidad de las simulaciones

### Objetivo

Obtener información más útil de cada iteración sin recurrir a aprendizaje automático.

### Posible trabajo

- Añadir políticas de rollout heurísticas o basadas en reglas.
- Permitir mezclas epsilon: elegir normalmente una acción heurística, pero algunas veces una al
  azar.
- Añadir progressive bias para que el conocimiento del juego influya en la selección inicial y se
  desvanezca al aumentar las visitas. (Implementado con evaluadores condicionales opcionales y
  valores de hijos almacenados en caché.)
- Comparar rollouts completos con rollouts más cortos evaluados mediante heurística.
- Investigar implicit minimax backups para juegos tácticos.
- Ajustar la exploración de forma independiente para cada juego y política de búsqueda.

### Riesgos

- Una política de rollout sesgada puede ignorar repetidamente líneas de victoria poco habituales.
- Las heurísticas fuertes pueden ocultar errores al hacer que agentes pequeños parezcan
  competentes.
- Una iteración más cara puede rendir peor con el mismo presupuesto temporal aunque necesite menos
  iteraciones.

### Señal de finalización

Al menos una política informada supera al rollout aleatorio uniforme con el mismo coste temporal y
el baseline sigue disponible.

## Etapa 3: reutilizar y compartir información de búsqueda

### Objetivo

Evitar descartar trabajo válido dentro de una partida y entre caminos de búsqueda equivalentes.

### Posible trabajo

- Implementado: conservar y podar el subárbol seleccionado después de una acción aceptada.
- Implementado: cambiar la raíz después de observar la acción real del oponente.
- Implementado: verificar juego y estado, y reiniciar de forma segura si no coinciden.
- Implementado: usar un ciclo de vida explícito para que una partida nueva no herede un árbol.
- Añadir tablas de transposición para caminos que llegan al mismo estado completo.
- Añadir límites configurables de memoria y políticas de poda.

### Restricciones de diseño

- La reutilización del árbol debe seguir siendo opcional porque no todas las variantes de búsqueda
  asocian las estadísticas a estados exactos.
- La reutilización en juegos de información perfecta no debe imponer `State: Eq + Hash` a agentes
  no relacionados, salvo que el beneficio justifique la restricción.
- Los agentes de información imperfecta deben identificar la reutilización mediante observaciones
  legales, conjuntos de información o historiales públicos, nunca mediante el estado oculto
  autoritativo.
- Los cambios de perspectiva de utilidad, reglas, política de rollout o evaluador de corte deben
  invalidar las estadísticas incompatibles.

### Señal de finalización

La búsqueda reutilizada produce las mismas decisiones legales que una búsqueda nueva, se reinicia
de forma segura ante una discrepancia y aporta una mejora medida de fuerza o latencia.

## Etapa 4: permitir juegos estocásticos

### Objetivo

Representar explícitamente las transiciones aleatorias del entorno y buscar correctamente a través
de ellas.

### Posible trabajo

- Distinguir nodos de decisión del jugador, nodos de azar y nodos terminales.
- Extender el contrato del juego para enumerar resultados aleatorios con sus probabilidades cuando
  resulte práctico.
- Permitir también muestrear un resultado aleatorio mediante un modelo generativo.
- Validar que las probabilidades sean finitas, no negativas y estén normalizadas.
- Retropropagar valores esperados a través de los nodos de azar, en lugar de maximizarlos o
  minimizarlos.
- Añadir progressive widening o muestreo de resultados cuando el espacio de azar sea muy grande.
- Mantener separado el azar del entorno del azar del agente para que las partidas sigan siendo
  reproducibles.
- Añadir un juego de dados pequeño como banco de pruebas específico para búsqueda estocástica.

### Preguntas que responder

- ¿Debe resolver el azar el ejecutor de simulaciones o una API de transiciones propiedad del juego?
- ¿Cuándo debe la búsqueda enumerar todos los resultados y cuándo debe muestrearlos?
- ¿Cómo se representan los resultados aleatorios en las trazas de acciones y en el análisis de
  repeticiones?

### Señal de finalización

Un juego estocástico de referencia ofrece partidas reproducibles, frecuencias de resultados
estadísticamente correctas y decisiones MCTS que coinciden con las expectativas exactas en
posiciones pequeñas.

## Etapa 5: introducir búsqueda con información imperfecta

### Objetivo

Permitir que los agentes razonen a partir de observaciones sin obtener acceso al estado oculto.

### Posible trabajo

- Definir observaciones públicas, observaciones privadas e historiales de acciones observables.
- Definir cómo muestrea un agente un estado completo compatible con su información.
- Añadir validación de creencias o determinizaciones para impedir estados muestreados imposibles.
- Implementar determinización simple como un baseline deliberadamente sencillo.
- Implementar Information Set MCTS como agente o backend de búsqueda independiente.
- Almacenar estadísticas por conjunto de información o historial observable, en lugar de por estado
  oculto completo.
- Añadir Kuhn Poker como juego de prueba compacto con azar, cartas ocultas, faroles y un equilibrio
  conocido.
- Medir la explotabilidad o la distancia a un equilibrio conocido, además de la tasa de victoria.

### Conceptos que estudiar

- Conjunto de información: estados que un jugador no puede distinguir entre sí.
- Estado de creencia: distribución de probabilidad sobre posibles estados subyacentes.
- Strategy fusion: elegir incorrectamente acciones distintas en estados ocultos que el jugador no
  puede distinguir.
- Non-locality: el valor de una decisión puede depender de elecciones estratégicas realizadas en
  otras partes del juego.
- Estrategia mixta: aleatorizar acciones intencionadamente para evitar ser explotado.

### Riesgos

- Una búsqueda determinista de información perfecta puede actuar accidentalmente como si conociera
  hechos ocultos.
- La tasa de victoria frente a un solo oponente no demuestra que una estrategia de información
  oculta sea sólida.
- Reutilizar estadísticas obtenidas bajo una distribución de creencias antigua puede sesgar una
  búsqueda posterior.

### Señal de finalización

Las pruebas demuestran que el agente no puede inspeccionar el estado oculto, se rechazan las
determinizaciones imposibles y el agente se aproxima a una estrategia razonable en un juego con
solución conocida.

## Etapa 6: añadir búsqueda guiada por políticas

### Objetivo

Guiar la expansión mediante preferencias previas sobre las acciones, conservando una simulación
exacta del juego.

### Posible trabajo

- Introducir una interfaz de priors que devuelva una distribución normalizada sobre las acciones
  legales.
- Implementar selección PUCT junto a UCT.
- Empezar con priors uniformes para establecer la equivalencia con el baseline.
- Añadir priors escritos manualmente para un juego.
- Entrenar priors tabulares o lineales sencillos a partir de partidas registradas.
- Utilizar ruido de exploración en la raíz y temperatura para seleccionar acciones únicamente en
  configuraciones de self-play.
- Guardar las distribuciones de visitas como objetivos de políticas mejoradas.

### Progresión del aprendizaje

1. Priors uniformes.
2. Priors escritos manualmente.
3. Tablas de frecuencias aprendidas de partidas MCTS.
4. Un modelo estadístico pequeño.
5. Una política neuronal si los enfoques más sencillos han alcanzado sus límites.

### Señal de finalización

La búsqueda guiada mejora la fuerza de juego o reduce las simulaciones necesarias para alcanzar la
fuerza del baseline, sin hacer posibles acciones ilegales.

## Etapa 7: aprender valores de posiciones mediante self-play

### Objetivo

Sustituir rollouts largos o débiles por una estimación de valor entrenada a partir de experiencia.

### Posible trabajo

- Definir una codificación estable y específica de cada juego para sus estados.
- Registrar ejemplos de entrenamiento que contengan el estado, la distribución de visitas en la
  raíz, el jugador actual y la utilidad final.
- Entrenar un modelo de valor que prediga el resultado final.
- Combinar las predicciones de política y valor en un solo modelo cuando resulte útil.
- Añadir una prueba de promoción: un modelo nuevo solo sustituye al actual tras superar una batería
  reproducible de partidas.
- Medir la calibración además del error de predicción; un valor de `0.8` debe tener una
  interpretación significativa.
- Evitar que las partidas de entrenamiento y evaluación compartan accidentalmente flujos
  aleatorios o datos.

### Bucle al estilo AlphaZero

```text
política y valor actuales
          |
          v
      MCTS guiado
          |
          v
       self-play
          |
          v
estados + distribuciones de visitas + utilidades finales
          |
          v
 entrenar una política y un valor mejores
```

### Riesgos

- El self-play puede amplificar sus propios puntos ciegos.
- La infraestructura de entrenamiento puede dominar la complejidad del código del juego y la
  búsqueda.
- La inferencia neuronal puede costar más que las simulaciones que sustituye en juegos pequeños.
- Las mejoras de fuerza deben compararse con el mismo cómputo total, no solo con las mismas
  iteraciones MCTS.

### Señal de finalización

Un agente experimental al estilo AlphaZero aprende mediante self-play y supera a su baseline MCTS
no guiado con un presupuesto de cómputo documentado.

## Etapa 8: investigación avanzada en teoría de juegos y modelos aprendidos

### Objetivo

Mantener visibles líneas de investigación a largo plazo sin tratarlas como requisitos inmediatos.

### Posibles direcciones

- Counterfactual Regret Minimization como baseline para juegos de información imperfecta, suma
  cero y dos jugadores.
- Estados de creencia pública y resolución con profundidad limitada inspirados en ReBeL.
- Búsqueda que combine mejora de políticas con razonamiento de teoría de juegos, inspirada en
  Student of Games.
- Selección de acciones mediante Gumbel para mejorar políticas de manera fiable con presupuestos
  pequeños de simulaciones.
- Búsqueda con muestreo de acciones y progressive widening para espacios de acciones extremadamente
  grandes o continuos.
- Evaluación de hojas por lotes y búsqueda paralela en el árbol.
- Dinámicas aprendidas inspiradas en MuZero cuando no se disponga de reglas exactas o resulte
  prohibitivamente caro simularlas.

### Advertencia sobre el alcance

MuZero no es automáticamente una mejora respecto a una búsqueda al estilo AlphaZero para Meeple
Bots. El proyecto ya dispone de simuladores exactos escritos en Rust, por lo que sustituirlos por
dinámicas aprendidas introduciría errores del modelo y una complejidad de entrenamiento
considerable. Solo resulta relevante para entornos cuyas reglas sean desconocidas, inaccesibles o
demasiado caras de ejecutar durante la búsqueda.

La búsqueda basada en teoría de juegos también es una rama distinta, no una pequeña modificación
de UCT. Debe construirse sobre un modelo de información imperfecta bien probado e incluir
evaluaciones orientadas a la explotabilidad.

## Orden de implementación sugerido

Las etapas no tienen que completarse como un único proyecto lineal. Un orden práctico sería:

1. Diagnósticos de búsqueda y comparaciones basadas en tiempo.
2. Políticas modulares de rollout y árbol.
3. Rollouts informados y progressive bias.
4. Reutilización opcional del árbol y transposiciones para juegos de información perfecta.
5. Nodos de azar explícitos y un juego de dados pequeño.
6. Historiales de observaciones, determinización y Kuhn Poker.
7. Information Set MCTS.
8. PUCT con priors uniformes y escritos manualmente.
9. Modelos tabulares o ligeros de política y valor.
10. Un experimento de self-play al estilo AlphaZero.
11. Búsqueda basada en teoría de juegos o dinámicas aprendidas únicamente cuando las requiera un
    juego concreto.

## Resumen de prioridades

| Dirección | Valor didáctico | Valor esperado | Complejidad | Prioridad cercana |
| --- | --- | --- | --- | --- |
| Diagnósticos de búsqueda | Alto | Alto | Baja | Muy alta |
| Políticas de búsqueda modulares | Alto | Alto | Media | Muy alta |
| Mejores políticas de rollout | Alto | Alto | Baja a media | Alta |
| Reutilización del árbol | Medio | Medio a alto | Media | Media |
| Tablas de transposición | Medio | Depende del juego | Media | Media |
| Nodos de azar | Alto | Alto | Media | Alta |
| Baseline por determinización | Alto | Medio | Media | Alta tras azar |
| Information Set MCTS | Muy alto | Alto | Alta | Alta tras determinización |
| PUCT y priors de acciones | Muy alto | Alto | Media | Media |
| Función de valor aprendida | Muy alto | Potencialmente alto | Alta | Posterior |
| Self-play al estilo AlphaZero | Muy alto | Depende del juego | Muy alta | Posterior |
| Ideas de ReBeL o Student of Games | Muy alto | Especializado | Muy alta | Investigación |
| Dinámicas al estilo MuZero | Alto | Bajo con simuladores exactos | Muy alta | Investigación |

## Checklist de evaluación para cada etapa

Antes de aceptar una técnica de búsqueda nueva, responder:

- ¿Conserva el juego legal y los límites de información del agente?
- ¿Se puede reproducir el experimento a partir de su configuración y semilla?
- ¿Mejora la fuerza con el mismo tiempo real?
- ¿Cuánta memoria adicional necesita?
- ¿Funciona de forma consistente en ambos asientos y contra distintos oponentes?
- ¿Puede aislarse su contribución mediante una ablación o comparación con un baseline?
- ¿Introduce conocimiento específico de un juego en un crate genérico?
- ¿Existe un fallback seguro cuando el juego no ofrece la capacidad necesaria?
- ¿Afecta a la configuración, trazas, bindings de Python o documentación de usuario?

## Recomendación actual

El camino inmediato más útil consiste en instrumentar el MCTS existente, modularizar sus políticas
y experimentar con rollouts informados. En paralelo, se pueden diseñar los contratos de juego para
transiciones de azar explícitas sin debilitar el límite actual de información perfecta. Un juego
estocástico pequeño seguido de Kuhn Poker proporcionaría una progresión controlada desde el azar
hasta la información oculta.

El MCTS guiado por políticas debe comenzar con priors manuales o tabulares antes de introducir
redes neuronales. Así se expone la idea esencial de AlphaZero mientras los experimentos siguen
siendo comprensibles. Las dinámicas aprendidas y los solucionadores avanzados de teoría de juegos
deben permanecer como opciones a largo plazo, activadas por los requisitos de un juego concreto y
no únicamente por esta hoja de ruta.
