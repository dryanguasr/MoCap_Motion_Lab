"""Reproducible exploratory kinetics for six explicitly selected short clips.

Run from repository root: .venv/Scripts/python scripts/estimate_planar_kinetics.py
No pose re-inference, downloads, or modification of source videos/old analyses.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from dataclasses import dataclass, asdict, replace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from seima_mocap.planar_dynamics import (
    G, local_polynomial, whole_body_wrench, double_support, segment_proximal_wrench,
)
from seima_mocap.output_layout import artifact_path, ensure_output_layout

CLIPS = (
    "20251212_132025_1", "20251212_133639_1", "20251212_134838_1",
    "20251212_135118_1", "20251212_135431_1", "20251212_140101_1",
)
OUT = ROOT / "data/processed"
PIPELINE = "planar_kinetics"
HEIGHT = 1.84
USED = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
CRITICAL = [11, 12, 23, 24, 25, 26, 27, 28]
FOOT = [29, 30, 31, 32]
BONES = [(11, 23), (12, 24), (23, 25), (24, 26), (25, 27), (26, 28)]
JOINTS = [f"{side}_{joint}" for side in ("left", "right") for joint in ("ankle", "knee", "hip")]


@dataclass(frozen=True)
class Assumptions:
    name: str = "base"
    body_mass_kg: float = 90.0
    racket_mass_kg: float = 0.18
    scale_multiplier: float = 1.0
    smoothing_window_s: float = 0.4
    cop_fraction_heel_to_toe: float = 0.5
    friction_coefficient: float = 0.7
    limb_mass_multiplier: float = 1.0
    inertia_multiplier: float = 1.0


def scenarios():
    base = Assumptions()
    yield base
    for field, values in (
        ("body_mass_kg", (85.0, 95.0)), ("scale_multiplier", (0.9, 1.1)),
        ("smoothing_window_s", (0.3, 0.5)), ("cop_fraction_heel_to_toe", (0.2, 0.8)),
        ("friction_coefficient", (0.5, 0.9)), ("limb_mass_multiplier", (0.9, 1.1)),
        ("inertia_multiplier", (0.8, 1.2)),
    ):
        for value in values:
            yield replace(base, name=f"{field}_{value:g}", **{field: value})


def read_timestamps(video, n):
    raw = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "frame=best_effort_timestamp_time", "-of", "json", str(video),
    ], text=True)
    t = np.array([float(f["best_effort_timestamp_time"]) for f in json.loads(raw)["frames"]])
    if len(t) != n or not np.all(np.diff(t) > 0):
        raise ValueError(f"Invalid or mismatched video timestamps: {video.name}")
    t -= t[0]
    if t[-1] > 15:
        raise ValueError(f"Refusing non-short video: {video.name}")
    return t


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_clip(stem):
    pose_cache = artifact_path(OUT, "arrays", stem, "left_player", "pose_landmarks", ".npz")
    event_file = artifact_path(OUT, "events", stem, "left_player", "semantic_events", ".csv")
    matches = list((ROOT / "data/raw").glob(f"{stem}.mp4"))
    # Search source recordings only, never generated videos.
    if len(matches) != 1:
        raise ValueError(f"Expected one source video for {stem}, got {len(matches)}")
    video = matches[0]
    with np.load(pose_cache) as data:
        norm = data["normalized"].copy()
        width, height = int(data["width"]), int(data["height"])
    t = read_timestamps(video, len(norm))
    pix = norm[..., :2] * [width, -height]
    quality = np.minimum(norm[..., 3], norm[..., 4])
    finite = np.all(np.isfinite(pix[:, USED]), axis=(1, 2))
    inside = np.all((norm[:, USED, :2] >= 0.005) & (norm[:, USED, :2] <= 0.995), axis=(1, 2))
    conf = (np.min(quality[:, CRITICAL], axis=1) >= 0.4)
    conf &= np.min(quality[:, FOOT], axis=1) >= 0.3
    conf &= np.min(quality[:, [0, 13, 14, 15, 16]], axis=1) >= 0.1
    conf &= np.mean(quality[:, [0, 13, 14, 15, 16]], axis=1) >= 0.5
    bone_ok = np.ones(len(t), dtype=bool)
    for a, b in BONES:
        length = np.linalg.norm(pix[:, a] - pix[:, b], axis=-1)
        med = np.nanmedian(length[finite & inside])
        bone_ok &= (length >= 0.6 * med) & (length <= 1.4 * med)
    target_ok = (norm[:, 23, 0] + norm[:, 24, 0]) / 2 < 0.58
    raw_ok = finite & inside & conf & bone_ok & target_ok
    torso = np.linalg.norm((pix[:, 11] + pix[:, 12] - pix[:, 23] - pix[:, 24]) / 2, axis=-1)
    if not np.any(raw_ok):
        raise ValueError(f"No usable scale reference observations: {stem}")
    scale = HEIGHT * 0.288 / np.median(torso[raw_ok])
    event_times = []
    with event_file.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            # Convert stored peak FRAME to actual timestamp, not old constant-FPS time.
            event_times.append(float(t[int(row["frame_peak"])]))
    event_near = np.zeros(len(t), dtype=bool)
    for event_t in event_times:
        event_near |= np.abs(t - event_t) <= 0.1
    flags = {"finite": finite, "inside_image": inside, "landmark_quality": conf,
             "bone_consistency": bone_ok, "left_target": target_ok, "raw_pose_valid": raw_ok,
             "near_heuristic_stroke": event_near}
    manifest = {
        "video": str(video.relative_to(ROOT)), "video_sha256": sha256(video),
        "pose_cache": str(pose_cache.relative_to(ROOT)),
        "pose_cache_sha256": sha256(pose_cache),
        "frames": len(t), "last_frame_time_s": float(t[-1]), "timestamp_source": "ffprobe best_effort_timestamp_time",
        "fixed_scale_m_per_px": float(scale), "scale_method": "0.288 * 1.84 / median reliable projected torso pixels",
        "heuristic_stroke_times_s": event_times,
    }
    return pix, quality, t, scale, flags, manifest


def make_segments(points, spec):
    """14 body segments + racket point mass; intentionally explicit priors.

    Male reference body mass fractions inspired by de Leva (1996). COM positions
    below are simplified assumed fractions, NOT subject-specific de Leva data.
    Every body inertia uses m*Lmedian**2/12 (uniform rod), NOT a 3-D inertia tensor.
    Hands are extrapolated 20% of forearm length; head/neck proxy ends at nose.
    """
    shoulder = (points[:, 11] + points[:, 12]) / 2
    hip = (points[:, 23] + points[:, 24]) / 2
    descriptions = [("trunk", hip, shoulder, 0.4346, 0.5),
                    ("head_neck", shoulder, points[:, 0], 0.0694, 0.7)]
    for side, offset in (("left", 0), ("right", 1)):
        p = lambda i: points[:, i + offset]
        descriptions.extend([
            (f"{side}_upper_arm", p(11), p(13), 0.0271, 0.45),
            (f"{side}_forearm", p(13), p(15), 0.0162, 0.43),
            (f"{side}_hand", p(15), p(15) + 0.2 * (p(15) - p(13)), 0.0061, 0.5),
            (f"{side}_thigh", p(23), p(25), 0.1416, 0.41),
            (f"{side}_shank", p(25), p(27), 0.0433, 0.44),
            (f"{side}_foot", p(29), p(31), 0.0137, 0.5),
        ])
    fractions = np.array([d[3] * (spec.limb_mass_multiplier if d[0].startswith(("left_", "right_")) else 1)
                          for d in descriptions])
    masses = spec.body_mass_kg * fractions / fractions.sum()
    coms, angles, lengths = [], [], []
    for name, a, b, fraction, com_fraction in descriptions:
        coms.append(a + com_fraction * (b - a))
        angle = np.arctan2((b - a)[:, 1], (b - a)[:, 0])
        good = np.isfinite(angle)
        # Fill only to make unwrap defined. Filtering rejects any invalid window.
        angle = np.interp(np.arange(len(angle)), np.flatnonzero(good), angle[good])
        angles.append(np.unwrap(angle))
        lengths.append(np.nanmedian(np.linalg.norm(b - a, axis=-1)))
    inertias = masses * np.array(lengths) ** 2 / 12 * spec.inertia_multiplier
    names = [d[0] for d in descriptions] + ["racket_point_at_right_wrist"]
    coms.append(points[:, 16])
    angles.append(np.zeros(len(points)))
    return (names, np.r_[masses, spec.racket_mass_kg], np.stack(coms, axis=1),
            np.stack(angles, axis=1), np.r_[inertias, 0.0])


def estimate(pix, t, scale, flags, spec):
    points = pix * scale * spec.scale_multiplier
    # Fixed coordinate origin for each clip, never a moving pelvis origin.
    ground_z = np.nanpercentile(points[:, FOOT, 1], 10)
    points[..., 1] -= ground_z
    points[..., 0] -= np.nanmedian((points[:, 23, 0] + points[:, 24, 0]) / 2)
    names, masses, com_raw, angles, inertias = make_segments(points, spec)
    # Only USED landmarks enter fit, so an unused missing landmark cannot reject it.
    joined = np.concatenate([points[:, USED].reshape(len(t), -1), com_raw.reshape(len(t), -1), angles], axis=1)
    # Exclude full derivative windows touching a heuristic impact neighborhood.
    observed = flags["raw_pose_valid"] & ~flags["near_heuristic_stroke"]
    fitted, vel, acc, smooth_ok = local_polynomial(t, joined, observed, spec.smoothing_window_s)
    cut = 2 * len(USED)
    end = cut + 2 * len(names)
    p = np.full_like(points, np.nan)
    p[:, USED] = fitted[:, :cut].reshape(len(t), len(USED), 2)
    pvel = np.full_like(points, np.nan)
    pvel[:, USED] = vel[:, :cut].reshape(len(t), len(USED), 2)
    coms = fitted[:, cut:end].reshape(len(t), len(names), 2)
    accelerations = acc[:, cut:end].reshape(len(t), len(names), 2)
    alpha = acc[:, end:]
    center, center_acc, force, hdot = whole_body_wrench(masses, coms, accelerations, inertias, alpha)
    bodyweight = spec.body_mass_kg * G
    normal_ok = (force[:, 1] >= 0.2 * bodyweight) & (force[:, 1] <= 2.5 * bodyweight)
    friction_ok = np.abs(force[:, 0]) <= spec.friction_coefficient * force[:, 1]
    total_ok = smooth_ok & normal_ok & friction_ok
    fraction = spec.cop_fraction_heel_to_toe
    cop_l = p[:, 29] + fraction * (p[:, 31] - p[:, 29])
    cop_r = p[:, 30] + fraction * (p[:, 32] - p[:, 30])
    # These COP positions are projected heel-to-toe points, NOT measured floor COP.
    fl, fr, load, support_ok, residual, lever = double_support(center, force, hdot, cop_l, cop_r)
    contact_ok = np.ones(len(t), dtype=bool)
    for heel, toe in ((29, 31), (30, 32)):
        foot_z = (p[:, heel, 1] + p[:, toe, 1]) / 2
        raw_z = (points[:, heel, 1] + points[:, toe, 1]) / 2
        baseline = np.nanpercentile(raw_z[flags["raw_pose_valid"]], 20)
        foot_vel = (pvel[:, heel] + pvel[:, toe]) / 2
        # Heuristic consistency with the DOUBLE SUPPORT assumption, not a classifier.
        contact_ok &= np.abs(foot_vel[:, 1]) <= 0.45
        contact_ok &= np.linalg.norm(foot_vel, axis=1) <= 1.0
        contact_ok &= foot_z - baseline <= 0.15
    torque_ok = total_ok & support_ok & contact_ok & (np.abs(residual) < 1e-6)
    result = {"time_s": t, "smooth_window_valid": smooth_ok, "normal_force_plausible": normal_ok,
              "friction_compatible": friction_ok, "double_support_compatible": support_ok,
              "feet_quasistationary": contact_ok, "total_force_valid": total_ok, "torque_valid": torque_ok,
              "com_x_m": center[:, 0], "com_z_projected_m": center[:, 1],
              "com_ax_m_s2": center_acc[:, 0], "com_az_m_s2": center_acc[:, 1],
              "required_hdot_Nm_diagnostic": hdot, "load_fraction_left_diagnostic": load,
              "support_lever_m_diagnostic": lever, "moment_balance_residual_Nm_diagnostic": residual}
    for field, value in (("total_Fx_N", force[:, 0]), ("total_Fz_N", force[:, 1]),
                         ("total_Fz_BW", force[:, 1] / bodyweight)):
        result[field] = np.where(total_ok, value, np.nan)
    for side, foot_force, cop, offset in (("left", fl, cop_l, 0), ("right", fr, cop_r, 1)):
        for axis, index in (("x", 0), ("z", 1)):
            result[f"{side}_F{axis}_N"] = np.where(torque_ok, foot_force[:, index], np.nan)
            result[f"{side}_cop_{axis}_m_assumed"] = np.where(torque_ok, cop[:, index], np.nan)
        distal_force, distal_moment, distal_point = foot_force, np.zeros(len(t)), cop
        for segment, joint, landmark in (("foot", "ankle", 27), ("shank", "knee", 25), ("thigh", "hip", 23)):
            i = names.index(f"{side}_{segment}")
            proximal_point = p[:, landmark + offset]
            proximal_force, proximal_moment = segment_proximal_wrench(
                masses[i], coms[:, i], accelerations[:, i], inertias[i], alpha[:, i],
                distal_point, distal_force, distal_moment, proximal_point,
            )
            result[f"{side}_{joint}_Nm"] = np.where(torque_ok, proximal_moment, np.nan)
            result[f"{side}_{joint}_Nm_kg"] = result[f"{side}_{joint}_Nm"] / spec.body_mass_kg
            distal_force, distal_moment, distal_point = -proximal_force, -proximal_moment, proximal_point
    segment_table = [{"name": name, "mass_kg": float(mass), "projected_rod_inertia_kg_m2": float(inertia)}
                     for name, mass, inertia in zip(names, masses, inertias)]
    return result, segment_table


def write_csv(path, columns):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for row in zip(*columns.values()):
            writer.writerow([int(x) if isinstance(x, (bool, np.bool_)) else
                             ("" if isinstance(x, (float, np.floating)) and not np.isfinite(x) else x)
                             for x in row])


def stats(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    return {"n": len(values), "median": float(np.median(values)), "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)), "min": float(values.min()), "max": float(values.max()),
            "p95_abs": float(np.percentile(np.abs(values), 95))}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def draw_clip(stem, baseline, envelope, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, layout="constrained")
    t = baseline["time_s"]
    axes[0].set_title(f"{stem} | PILOTO CONDICIONAL 2D · NO MEDICIÓN\n90 kg · 1.84 m · huecos = intervalos rechazados", fontsize=12)
    for key, label, color in (("total_Fz_N", "Vertical proyectada", "#167a8a"), ("total_Fx_N", "Horizontal imagen", "#b96516")):
        axes[0].plot(t, baseline[key], label=label, color=color)
        axes[0].fill_between(t, envelope[key + "_min"], envelope[key + "_max"], color=color, alpha=.16)
    axes[0].axhline(90 * G, color="gray", ls=":", label="Peso corporal 882.9 N")
    axes[0].set_ylabel("Fuerza total (N)")
    for ax, joint, title in zip(axes[1:], ("ankle", "knee", "hip"), ("Tobillo", "Rodilla", "Cadera")):
        for side, label, color in (("left", "Izquierdo", "#2867ad"), ("right", "Derecho", "#d1663a")):
            key = f"{side}_{joint}_Nm"
            ax.plot(t, baseline[key], label=label, color=color)
            ax.fill_between(t, envelope[key + "_min"], envelope[key + "_max"], alpha=.16, color=color)
        ax.set_ylabel(f"{title} (N·m)")
        ax.axhline(0, color="gray", lw=.6)
    for ax in axes:
        ax.grid(alpha=.18)
        ax.legend(loc="upper right", fontsize=8)
    # Preserve rejected starts/ends: matplotlib otherwise crops all-NaN tails.
    axes[-1].set_xlim(0, t[-1])
    axes[-1].set_xlabel("Tiempo real del video (s). Bandas: sensibilidad un-factor-a-la-vez; NO IC estadístico.")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_report(summaries):
    frames = sum(s["frames"] for s in summaries)
    force_n = sum(s["force_frames"] for s in summaries)
    torque_n = sum(s["torque_frames"] for s in summaries)
    lines = [
        "# Piloto computacional de fuerzas y momentos — seis clips cortos", "",
        "**Resultados condicionales de un modelo 2D, NO mediciones de fuerza ni torques anatómicos validados.**", "",
        f"Analizados {frames} frames del jugador principal situado a la izquierda. "
        "El video largo `20251212_132025.mp4` queda excluido y no se ha borrado ni alterado. "
        "Izquierdo/derecho en las variables identifica las extremidades del jugador principal, no al oponente.", "",
        f"El escenario base admite fuerza total en {force_n}/{frames} frames ({100*force_n/frames:.1f} %) "
        f"y fuerzas por pie/momentos en {torque_n}/{frames} ({100*torque_n/frames:.1f} %). "
        "Cobertura significa pasar filtros internos, NO exactitud o confianza estadística. "
        "Las regiones omitidas no son aleatorias: no se pueden generalizar estos resúmenes a todos los golpes.", "",
        "## Resultados base", "",
        "Fz es la componente vertical **supuesta en el plano de imagen**, no una plataforma de fuerza. "
        "Los percentiles usan exclusivamente frames admitidos; no son picos reales del gesto.", "",
        "| Clip | Fuerza admitida | Momentos admitidos | Mediana Fz (N) | P95 Fz (N) |",
        "|---|---:|---:|---:|---:|",
    ]
    for s in summaries:
        fz = s["force_Fz_N"]
        lines.append(f"| {s['video']} | {s['force_coverage_pct']:.1f} % | {s['torque_coverage_pct']:.1f} % | "
                     f"{fz.get('median', float('nan')):.0f} | {fz.get('p95', float('nan')):.0f} |")
    lines.extend(["", "Magnitud P95 del momento neto proyectado (N·m), no torque muscular ni indicador de lesión:", "",
                  "| Clip | Tobillo I / D | Rodilla I / D | Cadera I / D |", "|---|---:|---:|---:|"])
    for s in summaries:
        m = s["moments_Nm"]
        pairs = [f"{m['left_' + joint].get('p95_abs', float('nan')):.1f} / "
                 f"{m['right_' + joint].get('p95_abs', float('nan')):.1f}" for joint in ("ankle", "knee", "hip")]
        lines.append(f"| {s['video']} | " + " | ".join(pairs) + " |")
    lines.extend(["", "## Hipótesis y ecuaciones", "",
        "- Masa corporal 90 kg (peso 882.9 N), talla 1.84 m; raqueta supuesta de 0.18 kg como masa puntual en la muñeca derecha. "
        "El sistema completo pesa 884.67 N en reposo. No hay medición de raqueta ni datos de InBody incorporados.",
        "- Se utilizan x/y de imagen, NO las coordenadas world de MediaPipe centradas en pelvis. "
        "Eje x a la derecha, z hacia arriba en imagen, gravedad asumida paralela a ese eje. Se ignoran profundidad, "
        "inclinación real de cámara, rotación fuera del plano y perspectiva. No es un modelo sagital 3D.",
        "- Una escala FIJA por clip: 0.288 × 1.84 m dividido por la mediana del torso proyectado en frames fiables. "
        "Ese 28.8 % es un supuesto heredado, no una medida anatómica. No derivamos una escala variable frame a frame. "
        "Origen fijo; la altura del COM es proyectada respecto a un suelo aproximado y no altura física calibrada.",
        "- 14 segmentos corporales: tronco 43.46 %, cabeza/cuello 6.94 %; POR LADO brazo 2.71 %, antebrazo 1.62 %, "
        "mano 0.61 %, muslo 14.16 %, pierna 4.33 %, pie 1.37 %. Priors de referencia masculina inspirados en de Leva, "
        "no ajuste individual ni confirmación de su adecuación al sujeto. La suma se normaliza a la masa corporal.",
        "- COM supuesto a fracción proximal→distal: tronco 0.50, cabeza/cuello 0.70 hacia nariz desde hombros, brazo 0.45, "
        "antebrazo 0.43, mano 0.50, muslo 0.41, pierna 0.44, pie 0.50 talón→punta. La mano se extrapola "
        "20 % de la longitud de antebrazo. Son simplificaciones explícitas, NO una tabla completa de de Leva.",
        "- Inercias planares de barras homogéneas I=m×L_mediana²/12. Sin inercia rotacional de raqueta. "
        "No hay tensores inerciales 3D, masas segmentarias medidas ni estimación de músculo individual.",
        "- Ajuste polinomial local cúbico centrado de aproximadamente 0.4 s (ventana impar según fps), usando timestamps "
        "reales de ffprobe. Primera/segunda derivadas analíticas del ajuste. No se interpolan ventanas rechazadas.", "",
        "Balance global: F_suelo = Σ m_i (a_i − g); COM de cuerpo + raqueta. "
        "dH_COM/dt = Σ[I_i α_i + (c_i − COM) × m_i a_i]. Se suponen nulas otras fuerzas externas.", "",
        "Reparto entre pies: se supone doble apoyo, COP de cada pie en el 50 % talón→punta, momento libre nulo "
        "y fuerzas izquierda/derecha paralelas. F_I=λF, F_D=(1−λ)F. Se resuelve λ con el balance de momento global; "
        "sin estas restricciones no hay una solución única. COP es un punto proyectado SUPUESTO, no medido ni limitado "
        "por una huella real. No se calcula una alternativa de apoyo simple para completar huecos.", "",
        "Recursión distal→proximal en pie, pierna y muslo: F_prox=m(a−g)−F_dist; "
        "τ_prox=Iα−τ_dist−(r_dist−c)×F_dist−(r_prox−c)×F_prox. "
        "En el siguiente segmento se invierten fuerza y momento por acción/reacción. "
        "Signo positivo: r_x F_z − r_z F_x, antihorario en el gráfico x/z; momento que actúa sobre el segmento distal. "
        "No interpretar el signo como flexión/extensión o abducción anatómica.", "",
        "## Filtros de admisión (heurísticos, no umbrales clínicos)", "",
        "1. Coordenadas finitas y puntos usados dentro de un margen de imagen de 0.5 %. Pelvis del objetivo en x<0.58. "
        "Calidad=min(visibility,presence): tronco/rodillas/tobillos ≥0.4; talones/puntas ≥0.3; nariz/codos/muñecas mínimo ≥0.1 "
        "y promedio ≥0.5. Longitudes proyectadas de torso, muslos y piernas entre 0.6 y 1.4 veces su mediana.",
        "2. Todos los frames de la ventana de derivación deben ser admisibles. Se descartan bordes y discontinuidades "
        "temporales >1.75 veces dt mediano. Se excluye ±0.10 s alrededor de los picos semánticos de muñeca previos, "
        "y toda ventana que los toque. Estos son candidatos de golpe, NO detecciones verificadas de contacto de pelota; "
        "pueden quedar impactos no detectados. Tampoco se detectan automáticamente apoyos de mano en mesa.",
        "3. Fuerza total: 0.2≤Fz/peso_corporal≤2.5 y |Fx|≤μFz, con μ=0.7 supuesto. "
        "No se recortan valores a esos límites. La fricción real y la vertical real no están medidas.",
        "4. Para reparto/torques: fracción de carga 0≤λ≤1 sin clipping, brazo proyectado efectivo entre apoyos ≥0.06 m. "
        "Ambos pies deben ser compatibles con cuasi-apoyo: |velocidad vertical media talón/punta|≤0.45 m/s, "
        "velocidad proyectada≤1 m/s y altura≤0.15 m por encima del P20 del pie correspondiente. "
        "Es consistencia con un supuesto, no un detector validado de contacto.",
        "5. Residuo de momento del reparto <10⁻⁶ N·m como comprobación algebraica. "
        "Ese cierre se obtiene por construcción y no demuestra fidelidad biomecánica.", "",
        "## Sensibilidad e interpretabilidad", "",
        "Quince escenarios: base y variaciones UN FACTOR A LA VEZ de masa 85/95 kg, escala ±10 %, ventana 0.3/0.5 s, "
        "COP 20/80 % talón→punta, μ=0.5/0.9, masas de extremidades ±10 % renormalizando a masa corporal, "
        "e inercias ±20 %. Son escenarios elegidos para explorar sensibilidad, no distribuciones calibradas de error.", "",
        "Las bandas muestran mínimo/máximo de escenarios admitidos en cada instante con base admitida. "
        "Se registra cuántos escenarios sobreviven: cuando desaparece un escenario NO sabemos qué habría ocurrido allí. "
        "No son intervalos de confianza, no combinan simultáneamente perturbaciones y no cubren todos los errores "
        "de profundidad, COM, contacto, pose o cámara. Una banda estrecha de fuerza no implica una medida precisa.", "",
        "La cobertura baja y la sensibilidad de los momentos al COP impiden conclusiones sobre asimetrías, "
        "sobrecarga o riesgo de lesión. Medianas cercanas al peso corporal son compatibles con gravedad dominante, "
        "no validación. El muestreo cercano a 30 fps y el suavizado excluyen resolver picos rápidos o impactos.", "",
        "No se estiman torques de muñeca/codo/hombro en este piloto: la raqueta se incluye solo como masa puntual "
        "para el balance corporal. Faltan orientación/inercia de la raqueta y carga de impacto para un análisis defendible.", "",
        "## Archivos y reproducción", "",
        "- Nombres: `fuente__planar_kinetics__artefacto.ext`; los agregados usan `batch` como fuente.",
        "- `summaries/`: resumen por clip, escenarios y resumen de lote; `metadata/`: supuestos y manifiesto.",
        "- `metrics/`: cinética por frame y envolvente; `arrays/`: series de escenarios; `diagnostics/`: gráficas.",
        "- Los campos `diagnostic` pueden contener hipótesis inadmisibles; no son resultados cinéticos aceptados. "
        "Los campos COM describen el modelo proyectado; `total_force_valid` y `torque_valid` gobiernan la admisión.",
        "- Ejecutar desde la raíz del repositorio: `.venv/Scripts/python scripts/estimate_planar_kinetics.py`. "
        "Requiere numpy, matplotlib y ffprobe; reutiliza poses existentes. No necesita OpenSim ni nuevas dependencias instaladas.",
        "- Tests analíticos en `tests/test_planar_dynamics.py`: balance estático/dinámico, signos, reparto inadmisible "
        "y derivación con timestamps irregulares/huecos. Son controles de implementación, NO validación experimental.", "",
        "Protocolo para datos reales: `docs/PROTOCOLO_ADQUISICION_CINETICA.md`. "
        "Los artefactos pesados permanecen locales según `.gitignore`; los resúmenes e informes compactos pueden versionarse.", "",
        "## Referencias", "",
        "[OpenSim: entradas y alcance de dinámica inversa](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53090063). "
        "[De Leva: parámetros antropométricos de referencia](https://pubmed.ncbi.nlm.nih.gov/8872282/). "
        "La implementación es un modelo planar propio simplificado, no una ejecución de esos paquetes ni una réplica completa del artículo.", "",
    ])
    artifact_path(OUT, "reports", "batch", PIPELINE, "report", ".md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main():
    ensure_output_layout(OUT)
    assumptions = list(scenarios())
    write_json(artifact_path(OUT, "metadata", "batch", PIPELINE, "assumptions", ".json"),
               {"height_m": HEIGHT, "scenarios": [asdict(s) for s in assumptions],
               "excluded_video": "20251212_132025.mp4", "excluded_reason": "Long video explicitly excluded by user",
               "scope": "Main player at image left only; anatomical left/right refer to that player's limbs"})
    summaries, all_manifest = [], []
    fields = ["total_Fx_N", "total_Fz_N"] + [j + "_Nm" for j in JOINTS]
    for stem in CLIPS:
        pix, quality, t, scale, flags, manifest = load_clip(stem)
        results, scenario_summaries = [], []
        for spec in assumptions:
            result, segment_table = estimate(pix, t, scale, flags, spec)
            results.append(result)
            scenario_summaries.append({"assumptions": asdict(spec), "total_force_frames": int(result["total_force_valid"].sum()),
                "torque_frames": int(result["torque_valid"].sum()), "metrics": {f: stats(result[f]) for f in fields},
                "segments": segment_table})
        baseline = results[0]
        envelope = {"time_s": t}
        for f in fields:
            stack = np.stack([r[f] for r in results])
            count = np.isfinite(stack).sum(axis=0)
            minimum = np.min(np.where(np.isfinite(stack), stack, np.inf), axis=0)
            maximum = np.max(np.where(np.isfinite(stack), stack, -np.inf), axis=0)
            # Bands only at baseline-accepted frames; count says how many scenarios survive.
            valid = np.isfinite(baseline[f])
            envelope[f + "_min"] = np.where(valid & (count > 0), minimum, np.nan)
            envelope[f + "_max"] = np.where(valid & (count > 0), maximum, np.nan)
            envelope[f + "_valid_scenarios"] = count
        write_csv(artifact_path(OUT, "metrics", stem, PIPELINE, "frame_kinetics", ".csv"),
                  {"frame": np.arange(len(t)), **flags,
                  "landmark_quality_min_critical": np.nan_to_num(np.min(quality[:, CRITICAL], axis=1)), **baseline})
        write_csv(artifact_path(OUT, "metrics", stem, PIPELINE, "sensitivity_envelope", ".csv"), envelope)
        np.savez_compressed(artifact_path(OUT, "arrays", stem, PIPELINE, "scenario_series", ".npz"),
                            time_s=t, scenario_names=np.array([s.name for s in assumptions]),
                            **{f: np.stack([r[f] for r in results]) for f in fields})
        write_json(artifact_path(OUT, "summaries", stem, PIPELINE, "scenario_summaries", ".json"), scenario_summaries)
        draw_clip(stem, baseline, envelope,
                  artifact_path(OUT, "diagnostics", stem, PIPELINE, "kinetics", ".png"))
        summary = {"video": stem + ".mp4", "frames": len(t),
                   "force_frames": int(baseline["total_force_valid"].sum()), "torque_frames": int(baseline["torque_valid"].sum()),
                   "force_coverage_pct": float(100 * baseline["total_force_valid"].mean()),
                   "torque_coverage_pct": float(100 * baseline["torque_valid"].mean()),
                   "force_Fz_N": stats(baseline["total_Fz_N"]), "force_Fx_N": stats(baseline["total_Fx_N"]),
                   "moments_Nm": {j: stats(baseline[j + "_Nm"]) for j in JOINTS},
                   "gate_pass_counts_nonexclusive": {k: int(v.sum()) for k, v in {**flags, **baseline}.items()
                                                       if np.asarray(v).dtype == bool},
                   "maximum_valid_moment_balance_residual_Nm":
                       stats(baseline["moment_balance_residual_Nm_diagnostic"][baseline["torque_valid"]])}
        summaries.append(summary)
        all_manifest.append(manifest)
        write_json(artifact_path(OUT, "summaries", stem, PIPELINE, "summary", ".json"), summary)
        print(f"{stem}: force {summary['force_frames']}/{len(t)}, torques {summary['torque_frames']}/{len(t)}", flush=True)
    write_json(artifact_path(OUT, "metadata", "batch", PIPELINE, "manifest", ".json"),
        {"clips": all_manifest, "script_sha256": sha256(Path(__file__)),
        "model_sha256": sha256(ROOT / "src/seima_mocap/planar_dynamics.py"),
        "numpy_version": np.__version__, "python_version": sys.version})
    write_json(artifact_path(OUT, "summaries", "batch", PIPELINE, "summary", ".json"), summaries)
    write_report(summaries)
    write_csv(artifact_path(OUT, "summaries", "batch", PIPELINE, "summary", ".csv"), {
        "video": [s["video"] for s in summaries], "frames": [s["frames"] for s in summaries],
        "force_coverage_pct": [s["force_coverage_pct"] for s in summaries],
        "torque_coverage_pct": [s["torque_coverage_pct"] for s in summaries],
        "median_Fz_N": [s["force_Fz_N"].get("median", np.nan) for s in summaries],
        "p95_Fz_N": [s["force_Fz_N"].get("p95", np.nan) for s in summaries],
        **{j + "_p95_abs_Nm": [s["moments_Nm"][j].get("p95_abs", np.nan) for s in summaries] for j in JOINTS},
    })
    print(f"Completed six clips. Results: {OUT}")


if __name__ == "__main__":
    main()
