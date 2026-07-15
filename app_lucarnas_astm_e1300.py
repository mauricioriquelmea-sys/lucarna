# -*- coding: utf-8 -*-
"""
=====================================================================
CÁLCULO ESTRUCTURAL DE LUCARNAS VIDRIADAS (SKYLIGHTS)
Vidrios simples y termopaneles (DVH) inclinados
Norma de referencia: ASTM E1300-24

Autor: Proyectos Estructurales EIRL
Ejecución: streamlit run app_lucarnas_astm_e1300.py

-------------------------------------------------------------------
BASES DE CÁLCULO IMPLEMENTADAS
-------------------------------------------------------------------
1. Espesor efectivo laminado : Modelo de Wölfel-Bennison (ASTM E1300, Anexo X),
                               con módulo de corte G del interlayer conmutado
                               según la duración dominante de cada combinación.
2. Deflexión                 : Formulación no lineal de gran deformación
                               (ASTM E1300, Anexo X), evaluada con la CARGA REAL.
                               Fallback a Timoshenko lineal fuera del dominio
                               de validez del ajuste.
3. Tensión de trabajo        : Coeficientes de Timoshenko para placa rectangular
                               simplemente apoyada en 4 bordes, evaluada con la
                               CARGA EQUIVALENTE A 3 s.
4. Duración de carga         : Método de la carga equivalente a 3 segundos
                               (ASTM E1300): q_3s = q_d * (d/3)^(1/16).
                               Permite combinar cargas de duraciones distintas
                               contra una única tensión admisible de 3 s.
5. Proyección de cargas      : Las cargas gravitacionales (D, S, Lr) actúan
                               verticalmente y se proyectan perpendicularmente
                               al plano del vidrio mediante cos(theta).
                               El viento actúa perpendicular al plano.
6. Reparto de carga (LSF)    : Proporcional a la rigidez relativa (t_ef^3),
                               aplicado SOLO a las cargas externas transmitidas
                               a través de la cámara (W, S, Lr). El peso propio
                               de cada lámina es soportado por ella misma.
=====================================================================
"""

import math
from dataclasses import dataclass, field

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Polygon

matplotlib.use("Agg")

# =====================================================================
# 1. CONFIGURACIÓN DE PÁGINA Y ESTÉTICA CORPORATIVA
# =====================================================================
st.set_page_config(
    page_title="ASTM E1300-24 | Cálculo de Lucarnas Vidriadas",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main > div { padding-left: 2rem; padding-right: 2rem; max-width: 100%; }
    .stMetric {
        background-color: #f8f9fa; padding: 15px; border-radius: 10px;
        border: 1px solid #dee2e6;
    }
    .verdict-ok {
        background-color: #e7f6ec; border-left: 8px solid #1e7e34;
        padding: 22px; border-radius: 8px; margin: 15px 0;
        font-size: 1.55em; font-weight: 700; color: #14532d;
    }
    .verdict-fail {
        background-color: #fdecea; border-left: 8px solid #b02a37;
        padding: 22px; border-radius: 8px; margin: 15px 0;
        font-size: 1.55em; font-weight: 700; color: #6b1219;
    }
    .info-box {
        background-color: #eef2f7; border-left: 6px solid #0056b3;
        padding: 18px; border-radius: 6px; margin: 15px 0;
        font-size: 0.90em; line-height: 1.5;
    }
    .lite-header {
        background-color: #343a40; color: #ffffff; padding: 8px 14px;
        border-radius: 5px; font-weight: 600; margin-bottom: 10px;
    }
    .govern-box {
        background-color: #fff8e1; border-left: 6px solid #f0a500;
        padding: 14px; border-radius: 6px; margin: 10px 0; font-size: 0.95em;
    }
    .sidebar-help { font-size: 0.83em; color: #555; line-height: 1.35; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# 2. CONSTANTES NORMATIVAS Y TABLAS DE REFERENCIA
# =====================================================================

E_GLASS = 71_700e6          # Módulo de elasticidad del vidrio [Pa] (ASTM E1300)
NU_GLASS = 0.22             # Coeficiente de Poisson del vidrio [-]
GAMMA_GLASS = 25_000.0      # Peso específico del vidrio [N/m3] (~25 kN/m3)
WEIBULL_EXP = 1.0 / 16.0    # Exponente de duración de carga (ASTM E1300)
REF_DURATION_S = 3.0        # Duración de referencia de la tensión base [s]

# Duraciones de carga características [s]
DUR_WIND_3S = 3.0
DUR_WIND_60S = 60.0
DUR_ROOF_LIVE = 600.0            # Sobrecarga de techo (mantención): 10 min
DUR_SNOW_DEFAULT = 30 * 86400.0  # Nieve: 30 días
DUR_DEAD = 50 * 365 * 86400.0    # Peso propio: permanente (50 años)

# Tensión admisible base para duración de 3 s y probabilidad de rotura
# pb = 8 lites / 1000 (ASTM E1300) [Pa]
ALLOWABLE_STRESS_3S = {
    "Crudo (Annealed)": 23.3e6,
    "Termoendurecido (Heat Strengthened)": 46.6e6,
    "Templado (Tempered)": 93.1e6,
}

# Colores de representación gráfica según tratamiento térmico (convención Saflex)
GLASS_COLORS = {
    "Crudo (Annealed)": "#F2E205",                      # Amarillo
    "Termoendurecido (Heat Strengthened)": "#F28C0F",   # Naranjo
    "Templado (Tempered)": "#D62828",                   # Rojo
}

# Espesores nominales comerciales [mm] y su espesor mínimo de cálculo
# (ASTM E1300, Tabla 4: nominal vs. minimum thickness)
NOMINAL_TO_MIN_THICKNESS = {
    2.5: 2.16, 3.0: 2.92, 4.0: 3.78, 5.0: 4.57, 6.0: 5.56,
    8.0: 7.42, 10.0: 9.02, 12.0: 11.91, 16.0: 15.09,
    19.0: 18.26, 22.0: 21.44, 25.0: 24.61,
}

# Módulo de corte G del interlayer [Pa] para CORTA duración (viento, 3 s).
# Valores indicativos: verificar SIEMPRE contra la tabla del fabricante.
INTERLAYER_SHORT = {
    "PVB - 3 s @ 20 °C": 8.06e6,
    "PVB - 3 s @ 30 °C": 1.60e6,
    "PVB - 3 s @ 40 °C": 0.62e6,
    "PVB - 3 s @ 50 °C": 0.44e6,
    "SGP (Ionoplast) - 3 s @ 30 °C": 141.0e6,
    "SGP (Ionoplast) - 3 s @ 50 °C": 25.0e6,
    "Definido por el usuario": None,
}

# Módulo de corte G del interlayer [Pa] para LARGA duración (nieve, peso propio).
# El PVB pierde prácticamente toda su capacidad de transferencia de corte.
INTERLAYER_LONG = {
    "PVB - 1 mes @ 20 °C": 0.052e6,
    "PVB - 1 mes @ 30 °C": 0.0281e6,
    "PVB - 1 mes @ 40 °C": 0.0234e6,
    "PVB - permanente (G -> 0, límite estratificado)": 0.0001e6,
    "SGP (Ionoplast) - 1 mes @ 30 °C": 11.6e6,
    "SGP (Ionoplast) - 1 mes @ 50 °C": 3.3e6,
    "Definido por el usuario": None,
}

# Tablas de Timoshenko: placa rectangular simplemente apoyada en 4 bordes,
# carga uniforme.   w_max = alpha * q * b^4 / D   ;   sigma_max = beta * q * b^2 / t^2
# (b = lado corto, a = lado largo, nu = 0.3)
TIMO_RATIO = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 3.0, 4.0, 5.0, 1e3])
TIMO_ALPHA = np.array([0.00406, 0.00564, 0.00705, 0.00830, 0.00931,
                       0.01013, 0.01223, 0.01282, 0.01297, 0.01302])
TIMO_BETA = np.array([0.28740, 0.37620, 0.45300, 0.51720, 0.56880,
                      0.61020, 0.71340, 0.74100, 0.74760, 0.75000])

PONDING_ANGLE_LIMIT = 5.0   # Ángulo bajo el cual se revisa empozamiento [°]


# =====================================================================
# 3. ESTRUCTURAS DE DATOS
# =====================================================================

@dataclass
class LiteConfig:
    """Configuración de entrada de una lámina de vidrio."""
    name: str
    construction: str            # "Monolítico" | "Laminado"
    treatment: str               # Clave de ALLOWABLE_STRESS_3S
    t_nom: float                 # Espesor nominal de CADA ply [mm]
    t_interlayer: float = 0.0    # Espesor del interlayer [mm]
    g_short: float = 0.0         # G del interlayer, corta duración [Pa]
    g_long: float = 0.0          # G del interlayer, larga duración [Pa]
    lsf_strength: float = 1.0    # Laminate Strength Factor [-]


@dataclass
class LiteGeometry:
    """Propiedades geométricas derivadas de una lámina, en unidades SI [m]."""
    cfg: LiteConfig
    h_ef_w_short: float          # Espesor efectivo a deflexión, G corto [m]
    h_ef_s_short: float          # Espesor efectivo a tensión, G corto [m]
    h_ef_w_long: float           # Espesor efectivo a deflexión, G largo [m]
    h_ef_s_long: float           # Espesor efectivo a tensión, G largo [m]
    gamma_short: float           # Coef. transferencia de corte, corta duración
    gamma_long: float            # Coef. transferencia de corte, larga duración
    t_glass_total: float         # Espesor total de vidrio (sin interlayer) [m]
    t_build_nom: float           # Espesor nominal construido [mm]

    def h_ef_w(self, long_term: bool) -> float:
        return self.h_ef_w_long if long_term else self.h_ef_w_short

    def h_ef_s(self, long_term: bool) -> float:
        return self.h_ef_s_long if long_term else self.h_ef_s_short

    def gamma(self, long_term: bool):
        if self.cfg.construction == "Monolítico":
            return None
        return self.gamma_long if long_term else self.gamma_short

    def dead_load(self) -> float:
        """Peso propio de la lámina por unidad de área [Pa] (vertical)."""
        return GAMMA_GLASS * self.t_glass_total


@dataclass
class LoadCombo:
    """
    Combinación de carga de servicio (nivel ASD, sin mayoración).

    factors : multiplicadores de D, W, S, Lr
    long_term : True si la combinación está gobernada por cargas de larga
                duración (define el G del interlayer a utilizar).
    """
    name: str
    f_D: float = 0.0
    f_W: float = 0.0
    f_S: float = 0.0
    f_Lr: float = 0.0
    long_term: bool = False


@dataclass
class ComboResult:
    """Resultado de la verificación de una lámina bajo una combinación."""
    combo: str
    lite: str
    q_real: float                # Carga neta perpendicular real [Pa]
    q_eq3: float                 # Carga neta perpendicular equivalente 3 s [Pa]
    sigma: float                 # Tensión de trabajo [Pa]
    sigma_adm: float             # Tensión admisible [Pa]
    fu_stress: float
    delta: float                 # Deflexión [m]
    fu_defl: float
    method: str
    h_ef_w: float
    h_ef_s: float
    gamma: float = None
    lsf: float = 1.0
    notes: str = ""


# =====================================================================
# 4. MOTOR MATEMÁTICO — PLACAS
# =====================================================================

def timoshenko_coefficients(aspect_ratio: float) -> tuple:
    """
    Interpola los coeficientes alpha (deflexión) y beta (tensión) de Timoshenko
    para una placa simplemente apoyada en sus 4 bordes bajo carga uniforme.
    """
    ar = max(1.0, float(aspect_ratio))
    alpha = float(np.interp(ar, TIMO_RATIO, TIMO_ALPHA))
    beta = float(np.interp(ar, TIMO_RATIO, TIMO_BETA))
    return alpha, beta


def plate_flexural_rigidity(t: float) -> float:
    """Rigidez flexural D = E*t^3 / (12*(1 - nu^2))  [N*m]."""
    return E_GLASS * t ** 3 / (12.0 * (1.0 - NU_GLASS ** 2))


def linear_deflection(q: float, a: float, b: float, t: float) -> float:
    """Deflexión lineal de Timoshenko [m]. q [Pa]; a, b, t [m]."""
    if t <= 0 or q <= 0:
        return 0.0
    alpha, _ = timoshenko_coefficients(a / b)
    return alpha * q * b ** 4 / plate_flexural_rigidity(t)


def astm_nonlinear_deflection(q: float, a: float, b: float, t: float) -> tuple:
    """
    Deflexión de gran deformación según ASTM E1300 (Anexo X):

        x = ln( ln( q*(a*b)^2 / (E*t^4) ) )
        w = t * exp( r0 + r1*x + r2*x^2 )

    con r_i función de la relación de aspecto. Válido para 1 <= a/b <= 5 y
    carga adimensional > 1. Fuera de ese dominio se usa la solución lineal.

    Retorna (deflexión [m], método [str]).
    """
    if t <= 0 or q <= 0:
        return 0.0, "N/A"

    ar = a / b
    q_hat = q * (a * b) ** 2 / (E_GLASS * t ** 4)  # carga adimensional

    if q_hat <= 1.0 or not (1.0 <= ar <= 5.0):
        return linear_deflection(q, a, b, t), "Timoshenko lineal"

    x = math.log(math.log(q_hat))

    r0 = 0.553 - 3.830 * ar + 1.110 * ar ** 2 - 0.0969 * ar ** 3
    r1 = -2.290 + 5.830 * ar - 2.170 * ar ** 2 + 0.2067 * ar ** 3
    r2 = 1.485 - 1.908 * ar + 0.815 * ar ** 2 - 0.0822 * ar ** 3

    w = t * math.exp(r0 + r1 * x + r2 * x ** 2)
    return w, "E1300 no lineal"


def timoshenko_stress(q: float, a: float, b: float, t: float) -> float:
    """Tensión principal máxima de flexión [Pa] (teoría lineal, conservadora)."""
    if t <= 0 or q <= 0:
        return 0.0
    _, beta = timoshenko_coefficients(a / b)
    return beta * q * b ** 2 / t ** 2


# =====================================================================
# 5. MOTOR MATEMÁTICO — DURACIÓN DE CARGA
# =====================================================================

def duration_factor_to_3s(duration_s: float) -> float:
    """
    Factor de conversión de una carga de duración d a su equivalente de 3 s:

        q_3s = q_d * (d / 3)^(1/16)

    Es el procedimiento de ASTM E1300 para combinar cargas de distinta duración
    contra una única tensión admisible de referencia (3 s).
    """
    return (duration_s / REF_DURATION_S) ** WEIBULL_EXP


def allowable_stress_3s(treatment: str) -> float:
    """Tensión admisible de referencia a 3 s [Pa], pb = 8/1000."""
    return ALLOWABLE_STRESS_3S[treatment]


# =====================================================================
# 6. MOTOR MATEMÁTICO — LAMINADO Y GEOMETRÍA
# =====================================================================

def laminated_effective_thickness(h1: float, h2: float, hv: float,
                                  g_int: float, a_min: float) -> dict:
    """
    Espesor efectivo de un vidrio laminado de 2 láminas.
    Modelo de Wölfel-Bennison (ASTM E1300, Anexo X). Unidades en [m]; g_int [Pa].

        hs    = 0.5*(h1 + h2) + hv
        hs1   = hs*h1/(h1+h2)   ;   hs2 = hs*h2/(h1+h2)
        Is    = h1*hs2^2 + h2*hs1^2
        Gamma = 1 / (1 + 9.6*E*Is*hv / (G*hs^2*a_min^2))
        h_ef,w      = (h1^3 + h2^3 + 12*Gamma*Is)^(1/3)
        h_ef,sigma,i= sqrt( h_ef,w^3 / (h_i + 2*Gamma*hs_j) )
    """
    hs = 0.5 * (h1 + h2) + hv
    hs1 = hs * h1 / (h1 + h2)
    hs2 = hs * h2 / (h1 + h2)
    i_s = h1 * hs2 ** 2 + h2 * hs1 ** 2

    denom = g_int * hs ** 2 * a_min ** 2
    gamma = 0.0 if denom <= 0 else 1.0 / (1.0 + 9.6 * E_GLASS * i_s * hv / denom)
    gamma = min(max(gamma, 0.0), 1.0)

    h_ef_w = (h1 ** 3 + h2 ** 3 + 12.0 * gamma * i_s) ** (1.0 / 3.0)
    h_ef_s1 = math.sqrt(h_ef_w ** 3 / (h1 + 2.0 * gamma * hs2))
    h_ef_s2 = math.sqrt(h_ef_w ** 3 / (h2 + 2.0 * gamma * hs1))

    return {"h_ef_w": h_ef_w, "h_ef_s1": h_ef_s1, "h_ef_s2": h_ef_s2, "gamma": gamma}


def build_lite_geometry(cfg: LiteConfig, a_min: float) -> LiteGeometry:
    """
    Construye las propiedades geométricas de una lámina, calculando los espesores
    efectivos tanto para corta como para larga duración (el PVB pierde rigidez
    al corte bajo cargas sostenidas).
    """
    if cfg.construction == "Monolítico":
        t_min = NOMINAL_TO_MIN_THICKNESS[cfg.t_nom] / 1000.0
        return LiteGeometry(
            cfg=cfg,
            h_ef_w_short=t_min, h_ef_s_short=t_min,
            h_ef_w_long=t_min, h_ef_s_long=t_min,
            gamma_short=1.0, gamma_long=1.0,
            t_glass_total=t_min, t_build_nom=cfg.t_nom,
        )

    # Laminado simétrico de 2 plies del mismo espesor nominal
    h_ply = NOMINAL_TO_MIN_THICKNESS[cfg.t_nom] / 1000.0
    hv = cfg.t_interlayer / 1000.0

    res_s = laminated_effective_thickness(h_ply, h_ply, hv, cfg.g_short, a_min)
    res_l = laminated_effective_thickness(h_ply, h_ply, hv, cfg.g_long, a_min)

    return LiteGeometry(
        cfg=cfg,
        h_ef_w_short=res_s["h_ef_w"], h_ef_s_short=res_s["h_ef_s1"],
        h_ef_w_long=res_l["h_ef_w"], h_ef_s_long=res_l["h_ef_s1"],
        gamma_short=res_s["gamma"], gamma_long=res_l["gamma"],
        t_glass_total=2 * h_ply,
        t_build_nom=2 * cfg.t_nom + cfg.t_interlayer,
    )


def load_share_factors(t1_ef: float, t2_ef: float) -> tuple:
    """
    Factores de reparto de carga (LSF) de un termopanel, proporcionales a la
    rigidez relativa de cada lámina:  f1 = t1^3 / (t1^3 + t2^3);  f2 = 1 - f1.
    """
    s1, s2 = t1_ef ** 3, t2_ef ** 3
    total = s1 + s2
    if total <= 0:
        return 0.5, 0.5
    return s1 / total, s2 / total


# =====================================================================
# 7. MOTOR MATEMÁTICO — COMBINACIONES Y VERIFICACIÓN
# =====================================================================

def build_load_combos(has_wind: bool, has_snow: bool, has_live: bool,
                      combine_s_lr: bool = False) -> list:
    """
    Genera las combinaciones de carga de servicio a evaluar.

    Las cargas gravitacionales (D, S, Lr) se proyectan perpendicularmente al
    vidrio con cos(theta); el viento (W) ya actúa perpendicular al plano.
    Se evalúan viento en presión y en succión (esta última puede aliviar o
    revertir el efecto del peso propio).

    combine_s_lr : si es False (por defecto), la nieve (S) y la sobrecarga de
        techo (Lr) se tratan como acciones ALTERNATIVAS y NO concurrentes,
        criterio de ASCE 7 (D + (Lr o S)). Si es True, se agrega la combinación
        D + S + Lr para cubrir escenarios de concurrencia explícita
        (p. ej. mantención sobre nieve residual).
    """
    combos = [LoadCombo("D·cosθ (permanente)", f_D=1.0, long_term=True)]

    if has_wind:
        combos.append(LoadCombo("D·cosθ + W (presión)", f_D=1.0, f_W=1.0))
        combos.append(LoadCombo("D·cosθ + W (succión)", f_D=1.0, f_W=-1.0))
    if has_snow:
        combos.append(LoadCombo("D·cosθ + S·cosθ", f_D=1.0, f_S=1.0, long_term=True))
    if has_live:
        combos.append(LoadCombo("D·cosθ + Lr·cosθ", f_D=1.0, f_Lr=1.0))
    if has_wind and has_snow:
        # Coexistencia parcial (análoga a ASCE 7 ASD: D + 0.75S + 0.75W)
        combos.append(LoadCombo("D·cosθ + 0.75·S·cosθ + 0.75·W",
                                f_D=1.0, f_S=0.75, f_W=0.75, long_term=True))
    if combine_s_lr and has_snow and has_live:
        # Concurrencia explícita S + Lr (fuera del criterio por defecto de ASCE 7)
        combos.append(LoadCombo("D·cosθ + S·cosθ + Lr·cosθ",
                                f_D=1.0, f_S=1.0, f_Lr=1.0, long_term=True))
    return combos


def combo_lite_loads(combo: LoadCombo, geo: LiteGeometry, lsf: float,
                     w_kpa: float, s_kpa: float, lr_kpa: float,
                     cos_t: float, dur_wind: float, dur_snow: float) -> tuple:
    """
    Calcula la carga neta perpendicular sobre UNA lámina, real y equivalente 3 s.

    Criterio de reparto:
      - Cargas externas (W, S, Lr) se transmiten a través de la cámara y se
        reparten según el LSF.
      - El peso propio (D) de cada lámina es soportado íntegramente por ella
        misma: NO se reparte.

    Retorna (q_real [Pa], q_eq3 [Pa]).
    """
    d_pa = geo.dead_load() * cos_t                 # peso propio proyectado [Pa]
    w_pa = w_kpa * 1000.0                          # viento (ya perpendicular)
    s_pa = s_kpa * 1000.0 * cos_t                  # nieve proyectada
    lr_pa = lr_kpa * 1000.0 * cos_t                # sobrecarga de techo proyectada

    # --- Carga real -------------------------------------------------
    q_ext = combo.f_W * w_pa + combo.f_S * s_pa + combo.f_Lr * lr_pa
    q_self = combo.f_D * d_pa
    q_real = lsf * q_ext + q_self

    # --- Carga equivalente a 3 s (para verificación de tensiones) ----
    k_w = duration_factor_to_3s(dur_wind)
    k_s = duration_factor_to_3s(dur_snow)
    k_lr = duration_factor_to_3s(DUR_ROOF_LIVE)
    k_d = duration_factor_to_3s(DUR_DEAD)

    q_ext_eq3 = combo.f_W * w_pa * k_w + combo.f_S * s_pa * k_s + combo.f_Lr * lr_pa * k_lr
    q_self_eq3 = combo.f_D * d_pa * k_d
    q_eq3 = lsf * q_ext_eq3 + q_self_eq3

    return q_real, q_eq3


def analyze_combo_lite(combo: LoadCombo, geo: LiteGeometry, lsf: float,
                       a: float, b: float, q_real: float, q_eq3: float) -> ComboResult:
    """
    Verifica una lámina bajo una combinación:
      - Tensión  : con la carga EQUIVALENTE A 3 s, contra sigma_adm(3 s).
      - Deflexión: con la carga REAL (la conversión de duración no aplica
                   a la rigidez).
    Se trabaja con el valor absoluto de la carga neta: la placa se verifica
    igual en presión que en succión.
    """
    long_term = combo.long_term
    h_ef_w = geo.h_ef_w(long_term)
    h_ef_s = geo.h_ef_s(long_term)

    q_abs_real = abs(q_real)
    q_abs_eq3 = abs(q_eq3)

    # Envolvente de diseño para tensiones.
    # El factor de duración k >= 1 siempre, de modo que |q_eq3| >= |q_real| cuando
    # todas las cargas de la combinación actúan en el mismo sentido. Si los signos
    # se oponen (p. ej. succión que revierte el peso propio), la superposición de
    # equivalentes puede "acreditar" una amplificación inexistente y arrojar una
    # carga neta menor que la real. Se toma la envolvente para recuperar en ese
    # caso la verificación instantánea contra sigma_adm(3 s).
    q_design = max(q_abs_eq3, q_abs_real)
    envelope_note = " · gobierna q real (signos opuestos)" if q_abs_real > q_abs_eq3 else ""

    sigma = timoshenko_stress(q_design, a, b, h_ef_s)
    sigma /= max(geo.cfg.lsf_strength, 1e-6)   # Laminate Strength Factor

    sigma_adm = allowable_stress_3s(geo.cfg.treatment)
    delta, method = astm_nonlinear_deflection(q_abs_real, a, b, h_ef_w)

    return ComboResult(
        combo=combo.name,
        lite=geo.cfg.name,
        q_real=q_real,
        q_eq3=q_eq3,
        sigma=sigma,
        sigma_adm=sigma_adm,
        fu_stress=sigma / sigma_adm if sigma_adm > 0 else np.inf,
        delta=delta,
        fu_defl=0.0,   # se completa afuera (requiere delta admisible)
        method=method,
        h_ef_w=h_ef_w,
        h_ef_s=h_ef_s,
        gamma=geo.gamma(long_term),
        lsf=lsf,
        notes=("G largo plazo" if long_term else "G corto plazo") + envelope_note,
    )


# =====================================================================
# 8. REPRESENTACIÓN GRÁFICA — SECCIÓN INCLINADA
# =====================================================================

def _rot(points, theta_rad, origin=(0.0, 0.0)):
    """Rota una lista de puntos (x, y) un ángulo theta alrededor de un origen."""
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    ox, oy = origin
    out = []
    for x, y in points:
        dx, dy = x - ox, y - oy
        out.append((ox + dx * c - dy * s, oy + dx * s + dy * c))
    return out


def draw_inclined_section(cfg1: LiteConfig, cfg2: LiteConfig = None,
                          air_gap: float = 0.0, theta_deg: float = 30.0):
    """
    Dibuja el esquema transversal de la lucarna inclinada un ángulo theta,
    con vectores de gravedad/nieve (vertical) y viento (perpendicular al vidrio).

    Convención: theta = 0° es horizontal; theta = 90° es vertical.
    El eje local "u" recorre el largo del vidrio; el eje local "v" es el espesor.
    """
    fig, ax = plt.subplots(figsize=(6.0, 6.4))

    theta = math.radians(theta_deg)
    length = 100.0     # longitud representada del vidrio [unidades gráficas]
    v = 0.0            # coordenada local de espesor acumulado

    def add_layer(v0: float, thickness: float, color: str, edge="black", lw=1.3):
        """Agrega una capa (rectángulo en ejes locales) ya rotada a ejes globales."""
        pts = [(0, v0), (length, v0), (length, v0 + thickness), (0, v0 + thickness)]
        ax.add_patch(Polygon(_rot(pts, theta), closed=True, facecolor=color,
                             edgecolor=edge, linewidth=lw))

    def add_lite(v0: float, cfg: LiteConfig) -> float:
        """Dibuja una lámina completa; retorna la coordenada local final."""
        color = GLASS_COLORS[cfg.treatment]
        if cfg.construction == "Monolítico":
            add_layer(v0, cfg.t_nom, color)
            return v0 + cfg.t_nom
        t_ply, t_int = cfg.t_nom, cfg.t_interlayer
        add_layer(v0, t_ply, color)
        add_layer(v0 + t_ply, t_int, "black", edge="black", lw=0.6)  # interlayer
        add_layer(v0 + t_ply + t_int, t_ply, color)
        return v0 + 2 * t_ply + t_int

    v = add_lite(v, cfg1)
    if cfg2 is not None:
        add_layer(v, air_gap, "white", edge="gray", lw=0.9)          # cámara de aire
        v += air_gap
        v = add_lite(v, cfg2)
    total_thk = v

    # --- Vectores de carga ------------------------------------------
    # Punto de aplicación: centro de la cara exterior del vidrio
    face_pt = _rot([(length / 2.0, total_thk)], theta)[0]
    arrow_len = 42.0

    # Gravedad / Nieve: siempre vertical hacia abajo
    ax.annotate("", xy=(face_pt[0], face_pt[1] - 3),
                xytext=(face_pt[0], face_pt[1] + arrow_len),
                arrowprops=dict(arrowstyle="-|>", linewidth=2.4, color="#2b6cb0"))
    ax.text(face_pt[0] + 2, face_pt[1] + arrow_len + 3, "D + S + Lr\n(vertical)",
            fontsize=9, fontweight="bold", color="#2b6cb0", ha="left", va="bottom")

    # Viento: perpendicular al plano del vidrio (dirección normal local +v)
    nx, ny = -math.sin(theta), math.cos(theta)
    wind_pt = _rot([(length * 0.22, total_thk)], theta)[0]
    ax.annotate("", xy=(wind_pt[0], wind_pt[1]),
                xytext=(wind_pt[0] + nx * arrow_len, wind_pt[1] + ny * arrow_len),
                arrowprops=dict(arrowstyle="-|>", linewidth=2.4, color="#b02a37"))
    ax.text(wind_pt[0] + nx * (arrow_len + 6), wind_pt[1] + ny * (arrow_len + 6),
            "W\n(⊥ al vidrio)", fontsize=9, fontweight="bold",
            color="#b02a37", ha="center", va="center")

    # --- Línea horizontal de referencia y cota angular ---------------
    ax.plot([0, length * 0.62], [0, 0], color="gray", linestyle="--", linewidth=1.0)
    arc_r = length * 0.30
    arc = np.linspace(0, theta, 60)
    ax.plot(arc_r * np.cos(arc), arc_r * np.sin(arc), color="gray", linewidth=1.2)
    ax.text(arc_r * 1.12 * math.cos(theta / 2), arc_r * 1.12 * math.sin(theta / 2),
            f"θ = {theta_deg:.0f}°", fontsize=11, fontweight="bold", color="#333")

    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    ax.margins(0.28)
    ax.set_title(f"Sección de lucarna — Espesor total: {total_thk:.2f} mm",
                 fontsize=11, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


# =====================================================================
# 9. BARRA LATERAL — ENTRADA DE DATOS
# =====================================================================

def lite_input_panel(prefix: str, label: str, name: str) -> LiteConfig:
    """Genera los controles de configuración de una lámina en la sidebar."""
    st.sidebar.markdown(f'<div class="lite-header">{label}</div>',
                        unsafe_allow_html=True)

    construction = st.sidebar.selectbox(
        "Tipo de construcción", ["Monolítico", "Laminado"], key=f"{prefix}_constr"
    )
    treatment = st.sidebar.selectbox(
        "Tratamiento térmico", list(ALLOWABLE_STRESS_3S.keys()),
        index=2, key=f"{prefix}_treat"
    )
    t_nom = st.sidebar.selectbox(
        "Espesor nominal de la lámina [mm]",
        list(NOMINAL_TO_MIN_THICKNESS.keys()), index=4, key=f"{prefix}_tnom",
        help="Para vidrio laminado corresponde al espesor de CADA lámina (ply).",
    )

    cfg = LiteConfig(name=name, construction=construction, treatment=treatment,
                     t_nom=float(t_nom))

    if construction == "Laminado":
        cfg.t_interlayer = st.sidebar.number_input(
            "Espesor del interlayer [mm]", min_value=0.10, max_value=6.00,
            value=0.76, step=0.38, format="%.2f", key=f"{prefix}_tint",
        )

        # --- G para cargas de CORTA duración (viento) ----------------
        p_short = st.sidebar.selectbox(
            "Interlayer — G corta duración (viento)", list(INTERLAYER_SHORT.keys()),
            index=1, key=f"{prefix}_ps",
        )
        g_s = INTERLAYER_SHORT[p_short]
        if g_s is None:
            g_s = st.sidebar.number_input(
                "G corto plazo [MPa]", min_value=0.001, max_value=500.0,
                value=1.60, step=0.10, format="%.3f", key=f"{prefix}_gs") * 1e6
        else:
            st.sidebar.caption(f"G_corto = {g_s / 1e6:.3f} MPa")
        cfg.g_short = g_s

        # --- G para cargas de LARGA duración (nieve, peso propio) ----
        p_long = st.sidebar.selectbox(
            "Interlayer — G larga duración (nieve/D)", list(INTERLAYER_LONG.keys()),
            index=1, key=f"{prefix}_pl",
            help="El PVB pierde casi toda su transferencia de corte bajo carga "
                 "sostenida: el laminado tiende al límite estratificado.",
        )
        g_l = INTERLAYER_LONG[p_long]
        if g_l is None:
            g_l = st.sidebar.number_input(
                "G largo plazo [MPa]", min_value=0.0001, max_value=500.0,
                value=0.0281, step=0.01, format="%.4f", key=f"{prefix}_gl") * 1e6
        else:
            st.sidebar.caption(f"G_largo = {g_l / 1e6:.4f} MPa")
        cfg.g_long = g_l

        cfg.lsf_strength = st.sidebar.number_input(
            "Laminate Strength Factor [-]", min_value=0.10, max_value=1.00,
            value=1.00, step=0.05, format="%.2f", key=f"{prefix}_lsf",
            help="Factor de reducción de resistencia del laminado (Saflex).",
        )

    return cfg


st.sidebar.title("⚙️ Datos de Entrada")
st.sidebar.markdown(
    '<div class="sidebar-help">Verificación de lucarnas vidriadas inclinadas '
    '(vidrio simple y DVH) según ASTM E1300-24. Apoyo en 4 bordes '
    'simplemente apoyados.</div>', unsafe_allow_html=True
)
st.sidebar.divider()

# ---- Geometría y posición -------------------------------------------
st.sidebar.subheader("1. Geometría y posición")
a_long_mm = st.sidebar.number_input("Largo del vano, a [mm]", min_value=100.0,
                                    max_value=6000.0, value=1800.0, step=50.0)
b_short_mm = st.sidebar.number_input("Ancho del vano, b [mm]", min_value=100.0,
                                     max_value=6000.0, value=1200.0, step=50.0)
theta_deg = st.sidebar.slider("Inclinación de la lucarna, θ [°]",
                              min_value=0.0, max_value=90.0, value=15.0, step=1.0,
                              help="0° = horizontal (cubierta plana); "
                                   "90° = vertical (fachada).")

# ---- Cargas de diseño ------------------------------------------------
st.sidebar.subheader("2. Cargas de diseño")
w_kpa = st.sidebar.number_input(
    "Presión de viento, W [kPa]", min_value=-10.0, max_value=10.0,
    value=1.00, step=0.05, format="%.2f",
    help="Positivo = presión (hacia el interior). Negativo = succión. "
         "Se evalúan ambos sentidos automáticamente.",
)
dur_wind_label = st.sidebar.selectbox("Duración del viento", ["3 seg", "60 seg"])
dur_wind = DUR_WIND_3S if dur_wind_label == "3 seg" else DUR_WIND_60S

s_kpa = st.sidebar.number_input("Sobrecarga de nieve, S [kPa]", min_value=0.0,
                                max_value=15.0, value=0.50, step=0.05, format="%.2f")
snow_days = st.sidebar.number_input(
    "Duración de la nieve [días]", min_value=1.0, max_value=365.0,
    value=30.0, step=1.0,
    help="ASTM E1300 emplea habitualmente 30 días para nieve.",
)
dur_snow = snow_days * 86400.0

lr_kpa = st.sidebar.number_input("Sobrecarga de techo, Lr [kPa]", min_value=0.0,
                                 max_value=10.0, value=1.00, step=0.05, format="%.2f")

combine_s_lr = st.sidebar.checkbox(
    "Combinar nieve (S) con sobrecarga de techo (Lr)", value=False,
    help="Por defecto DESMARCADO: S y Lr se tratan como acciones alternativas "
         "y no concurrentes, según el criterio de ASCE 7 — D + (Lr o S). "
         "Marcar solo si el proyecto exige concurrencia explícita, p. ej. "
         "mantención sobre nieve residual.",
)
if combine_s_lr and s_kpa > 1e-9 and lr_kpa > 1e-9:
    st.sidebar.caption("➕ Se agrega la combinación D·cosθ + S·cosθ + Lr·cosθ")
else:
    st.sidebar.caption("S y Lr se evalúan por separado (alternativas)")

# ---- Deformación admisible -------------------------------------------
st.sidebar.subheader("3. Deformación admisible")
defl_criterion = st.sidebar.radio(
    "Criterio", ["L/60 (lado corto)", "L/175 (lado corto)", "Valor fijo [mm]"]
)
if defl_criterion == "Valor fijo [mm]":
    defl_adm_mm = st.sidebar.number_input("Deflexión admisible [mm]", min_value=1.0,
                                          max_value=100.0, value=19.05, step=0.5,
                                          format="%.2f")
elif defl_criterion == "L/60 (lado corto)":
    defl_adm_mm = min(a_long_mm, b_short_mm) / 60.0
else:
    defl_adm_mm = min(a_long_mm, b_short_mm) / 175.0

if defl_criterion != "Valor fijo [mm]":
    st.sidebar.caption(f"Δ_adm = {defl_adm_mm:.2f} mm")

st.sidebar.divider()

# ---- Tipo de sistema -------------------------------------------------
st.sidebar.subheader("4. Tipo de sistema")
system_type = st.sidebar.radio(
    "Glass Construction",
    ["Vidrio Simple (Single Lite)", "Termopanel (Insulating Unit)"],
)
is_igu = system_type.startswith("Termopanel")
st.sidebar.divider()

# ---- Configuración de láminas ---------------------------------------
cfg1 = lite_input_panel("l1", "Vidrio 1 — Exterior", "Lite 1 (Exterior)")
air_gap_mm = 0.0
cfg2 = None
if is_igu:
    st.sidebar.divider()
    air_gap_mm = st.sidebar.number_input("Espesor de cámara de aire [mm]",
                                         min_value=6.0, max_value=30.0,
                                         value=12.0, step=1.0)
    st.sidebar.divider()
    cfg2 = lite_input_panel("l2", "Vidrio 2 — Interior", "Lite 2 (Interior)")


# =====================================================================
# 10. CÁLCULO
# =====================================================================

# Normalización geométrica: a = lado largo, b = lado corto [m]
a_m = max(a_long_mm, b_short_mm) / 1000.0
b_m = min(a_long_mm, b_short_mm) / 1000.0
aspect_ratio = a_m / b_m
cos_t = math.cos(math.radians(theta_deg))
defl_adm_m = defl_adm_mm / 1000.0

geo1 = build_lite_geometry(cfg1, b_m)
geo2 = build_lite_geometry(cfg2, b_m) if is_igu else None
geometries = [geo1] + ([geo2] if is_igu else [])

combos = build_load_combos(has_wind=abs(w_kpa) > 1e-9,
                           has_snow=s_kpa > 1e-9,
                           has_live=lr_kpa > 1e-9,
                           combine_s_lr=combine_s_lr)

results = []
for combo in combos:
    long_term = combo.long_term
    if is_igu:
        f1, f2 = load_share_factors(geo1.h_ef_w(long_term), geo2.h_ef_w(long_term))
        shares = [f1, f2]
    else:
        shares = [1.0]

    for geo, lsf in zip(geometries, shares):
        q_real, q_eq3 = combo_lite_loads(combo, geo, lsf, w_kpa, s_kpa, lr_kpa,
                                         cos_t, dur_wind, dur_snow)
        res = analyze_combo_lite(combo, geo, lsf, a_m, b_m, q_real, q_eq3)
        res.fu_defl = res.delta / defl_adm_m if defl_adm_m > 0 else np.inf
        results.append(res)

# ---- Envolventes -----------------------------------------------------
gov_stress = max(results, key=lambda r: r.fu_stress)
gov_defl = max(results, key=lambda r: r.fu_defl)
gov_load = max(results, key=lambda r: abs(r.q_real))
design_ok = (gov_stress.fu_stress <= 1.0) and (gov_defl.fu_defl <= 1.0)
weight_total = sum(g.dead_load() * a_m * b_m / 9.81 for g in geometries)  # [kg]


# =====================================================================
# 11. PANEL PRINCIPAL — RESULTADOS
# =====================================================================

st.title("🔆 Cálculo Estructural de Lucarnas Vidriadas")
st.caption("ASTM E1300-24 — Vidrios simples y termopaneles (DVH) inclinados | "
           "Proyectos Estructurales EIRL")

if design_ok:
    st.markdown('<div class="verdict-ok">✅ DISEÑO ACEPTABLE — CUMPLE</div>',
                unsafe_allow_html=True)
else:
    motivos = []
    if gov_stress.fu_stress > 1.0:
        motivos.append("tensión")
    if gov_defl.fu_defl > 1.0:
        motivos.append("deflexión")
    st.markdown(
        f'<div class="verdict-fail">❌ DISEÑO NO ACEPTABLE — NO CUMPLE '
        f'({" y ".join(motivos)})</div>', unsafe_allow_html=True
    )

# ---- Métricas destacadas ----
m1, m2, m3, m4 = st.columns(4)
m1.metric("Carga neta crítica ⊥", f"{abs(gov_load.q_real) / 1000:.3f} kPa",
          help=f"Combinación: {gov_load.combo} — {gov_load.lite}")
m2.metric("σ máx / σ adm",
          f"{gov_stress.sigma / 1e6:.2f} / {gov_stress.sigma_adm / 1e6:.1f} MPa",
          help="Tensión evaluada con la carga equivalente a 3 s.")
m3.metric("Δ máx / Δ adm", f"{gov_defl.delta * 1000:.2f} / {defl_adm_mm:.2f} mm")
m4.metric("Peso del vidrio", f"{weight_total:.1f} kg")

f1c, f2c = st.columns(2)
f1c.metric("FU a flexión (envolvente)", f"{gov_stress.fu_stress:.3f}",
           delta="OK" if gov_stress.fu_stress <= 1.0 else "EXCEDE",
           delta_color="normal" if gov_stress.fu_stress <= 1.0 else "inverse")
f2c.metric("FU a deflexión (envolvente)", f"{gov_defl.fu_defl:.3f}",
           delta="OK" if gov_defl.fu_defl <= 1.0 else "EXCEDE",
           delta_color="normal" if gov_defl.fu_defl <= 1.0 else "inverse")

st.markdown(
    f'<div class="govern-box"><b>Combinación gobernante a flexión:</b> '
    f'{gov_stress.combo} — {gov_stress.lite} (FU = {gov_stress.fu_stress:.3f})<br>'
    f'<b>Combinación gobernante a deflexión:</b> {gov_defl.combo} — '
    f'{gov_defl.lite} (FU = {gov_defl.fu_defl:.3f})</div>',
    unsafe_allow_html=True,
)

# ---- Control de empozamiento (ponding) ----
if theta_deg < PONDING_ANGLE_LIMIT:
    st.warning(
        f"⚠️ **Riesgo de empozamiento (ponding).** La inclinación θ = {theta_deg:.0f}° "
        f"es menor a {PONDING_ANGLE_LIMIT:.0f}° y la deflexión máxima alcanza "
        f"{gov_defl.delta * 1000:.1f} mm. El agua tiende a acumularse en la zona "
        "deformada, incrementando la carga y la deflexión en un proceso que puede "
        "ser inestable. Verificar la pendiente efectiva bajo carga y el drenaje; "
        "esta aplicación NO modela la iteración carga-deflexión del empozamiento."
    )

col_main, col_fig = st.columns([1.75, 1.0], gap="large")

with col_main:
    st.subheader("Verificación por Combinación y Lámina")
    tabla = pd.DataFrame([{
        "Combinación": r.combo,
        "Lámina": r.lite,
        "LSF": round(r.lsf, 3),
        "q real [kPa]": round(r.q_real / 1000, 3),
        "q eq. 3s [kPa]": round(r.q_eq3 / 1000, 3),
        "h_ef,w [mm]": round(r.h_ef_w * 1000, 2),
        "h_ef,σ [mm]": round(r.h_ef_s * 1000, 2),
        "Γ": round(r.gamma, 4) if r.gamma is not None else "—",
        "σ [MPa]": round(r.sigma / 1e6, 2),
        "σ adm [MPa]": round(r.sigma_adm / 1e6, 1),
        "FU σ": round(r.fu_stress, 3),
        "Δ [mm]": round(r.delta * 1000, 2),
        "FU Δ": round(r.fu_defl, 3),
    } for r in results])

    def _highlight(row):
        crit = max(row["FU σ"], row["FU Δ"])
        if crit > 1.0:
            return ["background-color: #fdecea"] * len(row)
        if crit > 0.85:
            return ["background-color: #fff8e1"] * len(row)
        return [""] * len(row)

    st.dataframe(tabla.style.apply(_highlight, axis=1),
                 use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Descargar verificación completa (CSV)",
        data=tabla.to_csv(index=False).encode("utf-8-sig"),
        file_name="verificacion_lucarna_astm_e1300.csv",
        mime="text/csv",
    )

    # ---- Detalle por lámina ----
    st.subheader("Detalle de Configuración")
    for geo in geometries:
        with st.container(border=True):
            c = geo.cfg
            st.markdown(f"**{c.name} — {c.construction} / {c.treatment}**")
            detalle = [
                f"Espesor nominal construido: **{geo.t_build_nom:.2f} mm**",
                f"Peso propio: **{geo.dead_load() / 1000:.3f} kPa** "
                f"(proyectado: {geo.dead_load() * cos_t / 1000:.3f} kPa)",
                f"σ admisible (3 s): **{allowable_stress_3s(c.treatment) / 1e6:.1f} MPa**",
            ]
            if c.construction == "Laminado":
                detalle += [
                    f"Corta duración: Γ = **{geo.gamma_short:.4f}** · "
                    f"h_ef,w = **{geo.h_ef_w_short * 1000:.2f} mm** · "
                    f"h_ef,σ = **{geo.h_ef_s_short * 1000:.2f} mm**",
                    f"Larga duración: Γ = **{geo.gamma_long:.4f}** · "
                    f"h_ef,w = **{geo.h_ef_w_long * 1000:.2f} mm** · "
                    f"h_ef,σ = **{geo.h_ef_s_long * 1000:.2f} mm**",
                ]
            st.markdown("- " + "\n- ".join(detalle))

    # ---- Factores de duración aplicados ----
    st.subheader("Factores de Conversión a 3 s")
    st.markdown(
        f"""
        | Carga | Duración | k = (d/3)^(1/16) |
        |---|---|---|
        | Viento (W) | {dur_wind:.0f} s | {duration_factor_to_3s(dur_wind):.3f} |
        | Sobrecarga de techo (Lr) | {DUR_ROOF_LIVE:.0f} s | {duration_factor_to_3s(DUR_ROOF_LIVE):.3f} |
        | Nieve (S) | {snow_days:.0f} días | {duration_factor_to_3s(dur_snow):.3f} |
        | Peso propio (D) | permanente | {duration_factor_to_3s(DUR_DEAD):.3f} |
        """
    )

with col_fig:
    st.subheader("Sección Transversal")
    fig = draw_inclined_section(cfg1, cfg2 if is_igu else None, air_gap_mm, theta_deg)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("**Datos generales**")
    st.markdown(
        f"""
        - Dimensiones: **{a_long_mm:.0f} × {b_short_mm:.0f} mm**
        - Relación de aspecto a/b: **{aspect_ratio:.2f}**
        - Inclinación: **θ = {theta_deg:.0f}°** (cos θ = {cos_t:.3f})
        - Viento: **{w_kpa:.2f} kPa** ({dur_wind_label})
        - Nieve: **{s_kpa:.2f} kPa** · Lr: **{lr_kpa:.2f} kPa**
        - Apoyo: **4 bordes simplemente apoyados**
        - E = **{E_GLASS / 1e6:.0f} MPa** · ν = **{NU_GLASS}** · γ = **{GAMMA_GLASS / 1000:.0f} kN/m³**
        """
    )

# ---- Advertencias de validez ----
if aspect_ratio > 5.0:
    st.warning(
        f"Relación de aspecto a/b = {aspect_ratio:.2f} > 5.0: el ajuste no lineal "
        "de deflexión de ASTM E1300 está fuera de su dominio de validez; se aplicó "
        "la solución lineal de Timoshenko."
    )

st.markdown(
    """
    <div class="info-box">
    <b>Bases y limitaciones del cálculo</b><br>
    • <b>Duración de carga:</b> se emplea el método de la carga equivalente a 3 s
      de ASTM E1300 (q₃ = q·(d/3)<sup>1/16</sup>). Las tensiones se verifican con la
      carga equivalente contra σ<sub>adm</sub>(3 s); las <u>deflexiones se verifican
      con la carga real</u>, ya que la conversión de duración es un artificio de
      daño acumulado por fatiga estática y no afecta la rigidez.<br>
    • <b>Tensión:</b> coeficientes de Timoshenko (teoría lineal de placas). Es
      conservadora respecto de la formulación no lineal al no considerar el efecto
      membrana.<br>
    • <b>Deflexión:</b> formulación de gran deformación de ASTM E1300 (Anexo X),
      válida para 1 ≤ a/b ≤ 5 y q(ab)²/Et⁴ &gt; 1.<br>
    • <b>Laminado:</b> modelo de Wölfel-Bennison, dos láminas simétricas. El G del
      interlayer se conmuta según la duración dominante de cada combinación: bajo
      nieve o peso propio el PVB tiende al límite estratificado. Los valores de G
      son indicativos: verificar contra la tabla del fabricante para la
      temperatura y duración del proyecto.<br>
    • <b>Termopanel:</b> el LSF (t<sub>ef</sub>³) se aplica solo a las cargas
      externas transmitidas por la cámara (W, S, Lr). El <u>peso propio de cada
      lámina es soportado por ella misma</u> y no se reparte. No se incluyen cargas
      climáticas de la cámara (presión isócora, ΔT, Δaltitud entre fabricación y
      obra), que en lucarnas de vano pequeño pueden gobernar.<br>
    • <b>Nieve vs. sobrecarga de techo:</b> por defecto S y Lr se tratan como
      acciones <u>alternativas y no concurrentes</u> — D + (Lr o S) — según el
      criterio de ASCE 7. La concurrencia S + Lr se agrega solo si se activa
      explícitamente en el panel de cargas.<br>
    • <b>Cartas NFL:</b> no se reproducen (derivan del GFPM); el chequeo se realiza
      por tensión admisible, criterio equivalente en filosofía pero no idéntico.<br>
    • <b>Empozamiento:</b> se emite advertencia bajo θ = 5°, sin modelar la
      iteración carga–deflexión.<br>
    • <b>Fuera de alcance:</b> impacto y carga de rotura (requisito habitual de
      vidriado sobre cabeza), retención post-rotura del laminado, y verificación
      de la carpintería y sus fijaciones.<br>
    Los resultados requieren revisión y validación por un ingeniero responsable.
    </div>
    """,
    unsafe_allow_html=True,
)
