# Hoja de ruta para el desarrollo de agentes y búsqueda

[English](MCTS_ROADMAP.md) | [**Español**](MCTS_ROADMAP.es.md)

Se conserva el nombre histórico del archivo para no romper los enlaces existentes. Meeple Bots
busca construir agentes genéricos y fuertes para juegos de mesa modernos, incluidos los de azar
público e información imperfecta, **y extraer información estratégica útil de sus decisiones**.
Esta es una hoja de ruta de investigación, no el compromiso de implementar cada algoritmo. La
fuerza, la propiedad genérica de los componentes, la reproducibilidad y la evidencia estratégica
interpretable importan más que acumular técnicas.

## Base actual

Las capacidades siguientes ya están implementadas, aunque su beneficio estratégico depende del
juego y del presupuesto medido:

| Área | Capacidad actual |
| --- | --- |
| Medición | Diagnósticos de búsqueda y presupuestos de tiempo/iteraciones; Analyze, Probe, Study, partidas con semillas y experimentos emparejados. |
| Búsqueda genérica | Selección compartida UCT y UCB1-Tuned; rollout y evaluación de corte configurables, rollouts heurísticos/epsilon, MAST, Progressive Bias, RAVE, Progressive Widening y admisión guiada por RAVE. |
| Memoria de búsqueda | Reutilización opcional del árbol y transposiciones por estado exacto en búsqueda compatible de información perfecta; reutilización por trayectoria de observaciones en SO-ISMCTS de Lost Cities. |
| Familias de juegos | MCTS determinista de información perfecta, MCTS de azar público para juegos compatibles y SO-ISMCTS basado en observaciones para Lost Cities. |
| Propiedad de evaluación | `StateEvaluator` se define en Rust core y MCTS lo consume; las heurísticas específicas pertenecen a los juegos. |

Por tanto, los mecanismos principales de las antiguas etapas de modularidad, búsqueda informada,
reutilización, azar y observaciones son **base implementada**, no primeros pasos pendientes.
Algunas señales antiguas de
finalización, como mejoras universales de fuerza a igual tiempo, benchmarks de referencia
exhaustivos, límites de memoria configurables o un juego de información oculta con equilibrio
conocido, **no** se han establecido. La [guía de agentes](README.md) describe los mecanismos y
configuraciones disponibles; las guías de [Probe](../python/probes.md),
[Analyze](../crates/evaluation/README.md) y [Study](../python/studies.md) describen la medición.
Actualmente no hay modelos aprendidos de valor o política, PUCT, Alpha-Beta, un bucle de
entrenamiento AlphaZero ni un agente CFR.

## Reglas arquitectónicas y experimentales

- Los juegos poseen reglas, legalidad, azar, utilidad terminal y características específicas.
  Las búsquedas genéricas en Rust consumen contratos del juego y evaluación opcional; no deben
  ramificar por nombre de juego. El catálogo y los adaptadores componen combinaciones admitidas.
  Rust posee reglas, búsqueda y simulación; Python posee orquestación experimental, entrenamiento,
  publicación de datasets, análisis e informes. La elección del backend de inferencia sigue
  abierta; una llamada a Python en cada hoja de búsqueda Rust no es el diseño predeterminado.
- Mantener separadas las familias semánticas determinista de información perfecta, de azar público
  y de información imperfecta. Cada algoritmo declara las capacidades que necesita; no hay una
  búsqueda universal ni obligación de que todos los juegos admitan todos los agentes. Los
  resultados del azar no son acciones del jugador. Random, MCTS, MCTS estocástico, SO-ISMCTS y los
  futuros algoritmos siguen siendo consumidores independientes de pequeños contratos compartidos;
  la infraestructura neuronal es opcional.
- Orientar el valor según el jugador solicitado explícitamente y seleccionar según el **actor
  real**, no la paridad de profundidad. En juegos actuales existen acciones consecutivas de un
  mismo jugador. Un modelo puede asesorar a la búsqueda, pero nunca sustituye la legalidad, las
  transiciones ni los resultados terminales autoritativos.
- El lifecycle general de `Agent` es trusted y puede recibir estado autoritativo. Lost Cities usa
  un adaptador trusted que filtra eventos antes de entregarlos a la búsqueda SO, que solo recibe
  observaciones. Los futuros evaluadores y agentes de información oculta necesitan el mismo
  límite explícito; el lifecycle genérico por sí solo no es un sandbox.
- Evaluar con semillas nuevas de partida, asientos equilibrados, configuraciones guardadas y
  procedencia de modelos/datos. Las iteraciones fijas ayudan a diagnosticar reproducibilidad;
  **el tiempo real medido e igualado** es la comparación principal entre algoritmos con distinto
  coste por iteración. Usar datos de evaluación separados, ablaciones y probes de comportamiento.
  No convertir una preferencia estratégica no forzada en una aserción CI; reservar las pruebas de
  corrección para invariantes y resultados matemáticamente forzados.

## Siguiente paso: diagnosticar cuellos de botella estratégicos

**Objetivo y motivo.** Averiguar por qué las búsquedas actuales toman decisiones cuestionables
antes de construir un sistema grande de aprendizaje. Lost Cities SO-ISMCTS es el primer caso:
las jugadas aparentemente obvias que se han señalado aportan una hipótesis concreta que
comprobar, no un diagnóstico demostrado.

**Posible trabajo.** Usar los Probes existentes sobre posiciones fijas, varios presupuestos de
iteraciones y semillas de búsqueda. Observar acción elegida, visitas raíz, Q desde la perspectiva
del jugador raíz, disponibilidad, estabilidad y convergencia. Separar presupuesto/eficiencia de
muestreo insuficientes, señal débil de rollout o evaluación de corte y límites estructurales de
la determinización/SO-ISMCTS. Repetir los hallazgos decisivos con posiciones nuevas y tiempo
medido. Los Probes describen comportamiento; torneos y estudios miden fuerza.

**Riesgo y señal de finalización.** Más iteraciones pueden reforzar una estimación sesgada; una
elección estable no es necesariamente sólida. Terminar con capturas reproducibles y una
clasificación clara de si los errores desaparecen, persisten o siguen siendo inciertos al crecer
el presupuesto. No se necesita un oráculo estratégico en CI.

## Aprendizaje I: valor escalar antes que política

**Objetivo y motivo.** Comprobar si una evaluación estratégica mejor cambia las decisiones de
agentes que ya buscan. La función de valor es la cantidad estimada; una red neuronal es uno de
sus posibles aproximadores. Comparar un baseline neutral, una señal manual donde exista y un
valor aprendido. Un modelo lineal/estadístico puede servir de control; un MLP pequeño es un
primer experimento neuronal concreto. GPU, transformers, batching, entrenamiento de políticas y
un bucle AlphaZero no son requisitos previos.

**Posible trabajo.** Mantener genérico el contrato de evaluación escalar (`StateEvaluator` ya
tiene un owner neutral en Rust). Situar la codificación de características específica del juego
y la adaptación del modelo en un límite que conoce el juego, no en búsqueda genérica. No existe
un tensor universal de tablero. El primer dataset puede vincular observación/estado legítimo de
una decisión, perspectiva del jugador activo y resultado final. En Lost Cities, usar solo la
observación del jugador activo: puede codificar su mano, expediciones/descarte/puntuaciones
públicos, cartas restantes del mazo y fase/historial observados legítimamente; no puede incluir
la mano oculta rival ni el orden real futuro del mazo. El objetivo estima el resultado esperado
condicionado a la información disponible y la política de datos, no la verdad oculta. Los
resultados ocultos producen objetivos ruidosos. Mantener autoritativa la utilidad terminal.
SO-ISMCTS no integra hoy valor aprendido; su futura vía de evaluación segura respecto a
observaciones debe establecerse explícitamente, no deducirse del trait que consume MCTS.
Versionar codificador/modelo y registrar procedencia; no reutilizar silenciosamente estadísticas
de búsqueda con un modelo distinto.

**Riesgos.** Un self-play débil puede copiar sus propios puntos ciegos; partidas correlacionadas
entre entrenamiento y evaluación pueden exagerar el progreso; una fuga de estado oculto puede
hacer que un modelo inválido parezca excelente; precisión predictiva no implica mejor juego; el
coste de inferencia puede borrar la ganancia de búsqueda. Separar partidas de
entrenamiento/validación/test, usar semillas nuevas de evaluación e informar de calibración y
fuerza a igual tiempo.

**Señal de finalización.** El codificador supera comprobaciones del límite de información; se
informa de predicción y calibración en datos separados; una búsqueda adecuada consume el valor
sin ramificar por juego; las comparaciones neutral/manual/aprendido igualan el tiempo real medido
y equilibran asientos; se repiten los Probes de Lost Cities. Un resultado negativo bien medido
es investigación válida.

**Punto de decisión.** Si un valor mejor corrige sustancialmente las decisiones SO observadas,
continuar hacia búsqueda guiada por política y segura respecto a observaciones. Si persisten los
errores con presupuestos grandes pese a mejorar el valor, priorizar métodos de conjuntos de
información/teoría de juegos o de creencias. Un valor aprendido no elimina por sí solo strategy
fusion, sesgo de determinización, límites del modelado del oponente ni desajustes de creencias.

## Aprendizaje II: política sobre acciones legales

**Objetivo y motivo.** Aprender `P(acción | información/estado)` para mejorar orden de acciones,
expansión y eficiencia de muestreo, y revelar preferencias estratégicas. Fases anchas como la
colocación de gemas en SPOTF motivan este trabajo sin enseñar a la búsqueda genérica qué es una
gema.

**Posible trabajo.** Asociar puntuaciones o probabilidades con las **acciones legales** que
proporciona el juego; una salida conceptual es `[(acción, puntuación), ...]`. Un vector global
fijo es opcional y puede no encajar con acciones dinámicas o dependientes de la fase. Los
codificadores/adaptadores propiedad del juego pueden proporcionar índices estables cuando sea
útil. Con información oculta, la entrada de la política es una observación legítima. Mantener
separados el consejo de política, la legalidad del juego y el valor escalar.

**Riesgo y señal de finalización.** Identidad de acciones, normalización, máscaras y versión del
modelo deben ser estables entre entrenamiento e inferencia. Mostrar asociación con acciones
legales, medidas de política en datos separados y comparación de agentes a igual tiempo sin
ramificar por juego concreto en la búsqueda genérica.

## Búsqueda guiada: PUCT después del contrato de política

**Objetivo y motivo.** Comprobar si priors sobre acciones legales más valor mejoran la búsqueda,
y exportar visitas raíz como señal de política analizable.

**Posible trabajo.** Añadir priors en aristas de decisión, normalización/validación genéricas,
selección PUCT y distribuciones de visitas raíz vinculadas a acciones legales y al contexto del
modelo. Empezar en Tres en raya o Connect Four con priors/valor sencillos. Más adelante,
Progressive Widening guiado por política puede decidir qué acción pendiente entra al árbol; la
admisión guiada por RAVE ya existe, pero sus puntuaciones AMAF **no** son priors de política
aprendida. Integrar RAVE/MAST, PUCT estocástico o SO-PUCT y batching no es necesario para el
primer experimento.

**Riesgo y señal de finalización.** Conservar orientación por actor real y distinguir aristas de
decisión de resultados de azar; no confundir el diagnóstico de hijos admitidos con una política
completa de acciones legales. Mostrar acciones legales, priors validados, visitas raíz
reproducibles y comparaciones a igual tiempo.

## Rama de self-play: experimentos al estilo AlphaZero

**Objetivo y motivo.** Estudiar la mejora de política/valor con simuladores exactos de Rust en
juegos donde encajan los supuestos clásicos de determinismo, secuencia e información perfecta.
Tres en raya y Connect Four son las primeras comprobaciones del pipeline; Connect6, Boop y SPOTF
requieren codificación y asociación explícita de acciones, incluidas las fases/acciones
consecutivas del mismo jugador. Esta rama no es el destino de todos los juegos.

**Posible trabajo.** Combinar inferencia de política/valor, PUCT, ruido exploratorio en raíz,
muestreo de acciones de self-play según temperatura, objetivos de visitas legales en raíz,
resultados finales orientados por jugador, versiones congeladas del modelo durante la partida y
una puerta de evaluación. Rust expone decisiones de búsqueda y simulación exacta; Python publica
datasets, entrena/selecciona modelos y orquesta experimentos. Juegos de azar como Can't Stop y
Splendor necesitan una extensión explícita consciente del azar; PUCT ordinario no se aplica sin
cambios. Lost Cities necesita un tratamiento de observaciones/creencias, no una red clásica de
estado omnisciente.

**Riesgo y señal de finalización.** El self-play puede reforzar puntos ciegos y los cambios de
modelo pueden invalidar datos de búsqueda reutilizados. Demostrar un bucle reproducible completo,
registros de entrenamiento legales y versionados, evaluación separada/a igual tiempo y salidas
de decisión interpretables. Ganar una sola partida no es evidencia suficiente.

## Rama de investigación de información imperfecta

**Objetivo y motivo.** Investigar estrategias mixtas intencionales y razonamiento por conjuntos
de información cuando los errores SO-ISMCTS parezcan estructurales. Ganar a un solo oponente no
demuestra robustez; la explotabilidad o la distancia a un equilibrio conocido pueden informar más
en juegos pequeños adecuados. La familia CFR interesa tanto para la fuerza como para extraer
estrategia, sin presumir que supera a SO-ISMCTS.

**Posible trabajo.** Reubicar Kuhn Poker como futuro banco de pruebas compacto con equilibrio
conocido para CFR y estrategias mixtas; estudiar después variantes de CFR/Deep CFR o regret
neuronal y métodos de creencia pública. Comparar con SO-ISMCTS en problemas ocultos más ricos
solo tras validar los límites de información y la medición. Los métodos tipo ReBeL son una
dirección posterior consciente de creencias, no un sinónimo de SO-ISMCTS ni AlphaZero.

**Riesgo y señal de finalización.** Distinguir strategy fusion, sesgo de determinización, modelo
del oponente y desajuste de creencias; un valor o política aprendidos no resuelven ninguno
automáticamente. Un pequeño experimento de referencia debe aportar evidencia reproducible de
estrategia/explotabilidad antes de escalar a un juego moderno.

## Ramas de referencia opcionales y horizonte de investigación

- **Alpha-Beta:** Búsqueda opcional didáctica/de referencia para juegos deterministas, de
  información perfecta, suma cero y dos jugadores. Comparar minimax sistemático con MCTS,
  resolver posiciones pequeñas y probar el evaluador escalar compartido. La identidad del actor,
  no la paridad de profundidad, debe controlar maximizar/minimizar. No es la ruta crítica hacia
  agentes aprendidos ni un atajo para juegos de información oculta.
- **Expectiminimax:** Referencia opcional de azar público para juegos pequeños; MCTS estocástico ya
  proporciona la ruta principal de búsqueda con azar.
- **Trabajo avanzado:** Deep CFR, métodos de creencia pública/tipo ReBeL, ideas de Student of
  Games, búsqueda Gumbel, muestreo de acciones, inferencia por lotes y búsqueda paralela siguen
  siendo opciones. Las dinámicas aprendidas tipo MuZero tienen baja prioridad mientras exista
  simulación exacta en Rust; reconsiderarlas solo si faltan reglas o simularlas resulta demasiado
  caro.

## Extracción de estrategia e interpretabilidad

Es un resultado transversal, no un efecto secundario de una tasa de victoria mayor. Conservar
visitas raíz, estimaciones Q/valor, distribuciones sobre acciones legales, disponibilidad SO
cuando proceda, fase/contexto, características de observación legítimas, versión del
modelo/codificador y estabilidad entre semillas. Preguntar dónde hay grandes cambios de valor,
qué acciones son robustas entre determinizaciones, qué patrones se relacionan con el valor,
dónde una política se concentra o mezcla, qué fases cuestan búsqueda y dónde discrepan los
agentes. Distinguir incertidumbre de preferencia fuerte; una red no se explica sola.

Las trayectorias de valor y comparaciones condicionadas a acciones son útiles, pero
`V(después) - V(antes)` no es automáticamente el valor estratégico de una acción: perspectiva,
azar, información oculta y respuesta rival pueden cambiar su interpretación. Preferir
comparaciones controladas en la raíz/contrafactuales y anotar sus supuestos. Los Probes describen
decisiones, los datasets separados evalúan modelos y los estudios/torneos emparejados miden fuerza.

## Orden de implementación sugerido

1. Usar la infraestructura existente Probe/Analyze/Study para diagnosticar decisiones SO de Lost
   Cities en posiciones fijas, presupuestos y semillas; registrar tiempo medido.
2. Usar el contrato escalar ya alojado en core; crear un codificador seguro de observación o
   específico del estado y un pequeño modelo de valor aprendido.
3. Integrar el valor en una búsqueda adecuada sin fuga de estado oculto; comparar señales
   neutrales, manuales (donde existan) y aprendidas a igual tiempo y con semillas nuevas.
4. Repetir Probes y dejar que la evidencia elija entre experimentos SO guiados por política o una
   rama de conjuntos de información/creencias.
5. Añadir una política aprendida sobre acciones legales; después, priors genéricos, PUCT y salida
   de distribución de visitas raíz en un juego sencillo de información perfecta.
6. Validar un bucle de self-play al estilo AlphaZero en ese juego adecuado; extenderlo luego a
   juegos compatibles más complejos mediante adaptadores propiedad de los juegos.
7. Construir un baseline compacto CFR/de equilibrio y comparar métodos de información oculta
   cuando lo justifique la evidencia; abordar técnicas avanzadas solo ante límites concretos.

Los pasos son condicionales, no una jerarquía algorítmica obligatoria. Alpha-Beta y
expectiminimax siguen como ramas opcionales de referencia.

## Resumen de prioridades

| Dirección | Estado | Prioridad cercana | Dependencia clave |
| --- | --- | --- | --- |
| Diagnóstico de comportamiento y extracción de estrategia | Siguiente / transversal | Muy alta | Probes existentes y posiciones reproducibles |
| Codificación segura y valor aprendido | Siguiente | Muy alta | Entrada legítima, datos separados, evaluador neutral |
| Política aprendida sobre acciones legales | Después del valor | Alta | Asociación de acciones y procedencia del modelo |
| PUCT y widening guiado por política | Después de la política | Alta | Priors y política raíz sobre acciones legales |
| Self-play tipo AlphaZero | Rama de juegos adecuados | Media/alta tras las bases | PUCT, datos versionados y puerta de evaluación |
| CFR y métodos de información/creencias | Rama de investigación | Alta si persisten errores SO | Referencia compacta y medidas de teoría de juegos |
| Alpha-Beta / expectiminimax | Referencia opcional | Opcional / didáctica | Semántica elegible del juego |
| Búsqueda por lotes/paralela y métodos avanzados | Horizonte de investigación | Según evidencia | Cuello de botella medido |
| Dinámicas tipo MuZero | Horizonte de investigación | Baja con simuladores exactos | Reglas exactas ausentes o costosas |

## Checklist de evaluación

En cada experimento, preguntar si se respetan legalidad, orientación por actor y límites de
información; si quedan registradas configuración, procedencia de código/modelo/codificador y
datos; si se separan partidas de entrenamiento/validación/test y semillas de comparación; si se
equilibran asientos y oponentes; si se informan calibración en datos separados, fuerza a igual
tiempo y coste de inferencia; si baseline y ablación aíslan la señal nueva; y si las salidas
permiten analizar estrategia con cuidado. Las iteraciones siguen sirviendo de diagnóstico, no
de medida principal de equidad entre métodos de distinto coste.

## Recomendación actual

Determinar primero si los errores estratégicos actuales, en especial los de Lost Cities
SO-ISMCTS, se deben a presupuesto/eficiencia de muestreo insuficientes, evaluación estratégica
débil o límites más profundos de conjuntos de información. Usar Probes de posiciones fijas con
varios presupuestos y semillas. Después entrenar un modelo pequeño de valor con codificación
legítima de observación/estado, integrarlo mediante el contrato escalar genérico y compararlo
con los baselines actuales a igual tiempo. Si el valor mejor corrige los errores observados,
avanzar hacia política aprendida y búsqueda guiada; si persisten, priorizar CFR o métodos de
creencias antes de limitarse a ampliar la red o el presupuesto.
