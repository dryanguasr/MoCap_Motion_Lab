"""One fixed, three-window H1 experiment. Never launches annotation or later stages.

Run once, inspect its checkpoint and videos, then stop for human review.
Local pickle files are generated here from trusted local project data; they are
not an input interchange format and must not be replaced by downloaded files.
"""
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import subprocess
import sys
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def save_pickle(path, value):
    path.write_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def load_pickle(path):
    return pickle.loads(path.read_bytes())


def worker(output, variant):
    library = output / "baseline_source/src" if variant == "baseline" else ROOT / "src"
    sys.path.insert(0, str(library))
    from seima_mocap.assisted_tracking import replay_tracking
    from seima_mocap.video_io import atomic_json
    plan = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    expected_bundles = json.loads((output / "bundle_hashes.json").read_text())
    worker_sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((library / "seima_mocap").glob("*.py"))}
    if variant != "baseline":
        from seima_mocap.stroke_guidance import StrokeGuide, StrokePriorConfig
    for case in plan["cases"]:
        assert hashlib.sha256((output / f"{case['id']}_inputs.pkl").read_bytes()).hexdigest() == expected_bundles[case['id']]
        data = load_pickle(output / f"{case['id']}_inputs.pkl")
        priors, traces = [], {}
        kwargs = {}
        if variant != "baseline":
            guide = StrokeGuide(data["scene"].table_polygon_xy, StrokePriorConfig(**plan["stroke_parameters"]))
            priors = [guide.step(c) for c in data["contexts"]]

            def candidate_prior(index, candidates):
                adjusted = priors[index].apply(candidates)
                traces[index] = [{"xy": c.observation.pixel_xy.tolist(), "source": c.observation.source,
                                  "score": c.observation.confidence,
                                  "weight": c.reason_scores["stroke_prior_weight"]} for c in adjusted]
                return adjusted

            kwargs["candidate_prior"] = candidate_prior
        result = replay_tracking(data["times"], data["evidence"], data["models"], data["contexts"],
                                 data["scene"], data["params"], [data["seed"]], stop_on_request=False,
                                 recovery_candidates=lambda i: data["recovery"][i], **kwargs)
        save_pickle(output / f"{case['id']}_{variant}.pkl", result)
        atomic_json(output / f"{case['id']}_{variant}_details.json",
                    {"requests": result.requests, "priors": [p.record() for p in priors],
                     "candidate_trace": traces, "frames": len(result.tracks), "worker_source_sha256": worker_sources})
        print(f"{variant} {case['id']}: {len(result.tracks)} frames; {len(result.requests)} loss requests", flush=True)


def prepare(output, resume=False):
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    from dataclasses import asdict
    from seima_mocap.rally_clips import validate_manifest, clip_parts
    from seima_mocap.rally_pipeline import prepare_evidence
    from seima_mocap.racketvision_adapter import load_cache
    from seima_mocap.pose_context import load_body_frames
    from seima_mocap.ball_tracking import BallPhysicalParams
    from seima_mocap.table_geometry import sha256_file
    from seima_mocap.stroke_guidance import StrokePriorConfig
    from seima_mocap.video_io import atomic_json
    output.mkdir(parents=True, exist_ok=resume)  # resume is explicit and verifies the frozen protocol
    path = ROOT / "data/annotations/rallies/professionals.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    plan = json.loads((ROOT / "config/stroke_guidance_stage1.json").read_text(encoding="utf-8"))
    plan["stroke_parameters"] = asdict(StrokePriorConfig(**plan["stroke_parameters"]))
    frozen = json.loads((output / "protocol.json").read_text(encoding="utf-8")) if resume else None
    if frozen:
        for key in ("cases", "stroke_parameters", "baseline_commit", "assistance"):
            assert plan[key] == frozen[key], f"Cannot resume with changed {key}"
        for key in ("manifest", "journal", "video", "geometry"):
            original = frozen["input_provenance"][key]
            assert sha256_file(original["path"]) == original["sha256"], f"Cannot resume: {key} changed"
    journal_path = path.parent / "interventions" / (plan["pilot"] + ".json")
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    part = next(p for p in clip_parts(manifest) if p["id"] == plan["pilot"])
    assert part["start_frame"] == 0 and part["end_frame"] == 481, "Pilot changed: revise protocol before running"
    assert journal["identity"]["source_sha256"] == manifest["source"]["sha256"]
    assert journal["identity"]["frame_range"] == [0, 481]
    shot = next(s for s in manifest["shots"] if s["id"] == part["shot_id"])
    assert journal["identity"]["geometry_sha256"] == shot["geometry"]["sha256"]
    reserved = [c["label_block"] for c in plan["cases"] if c["split"] == "reserved"]
    for case in plan["cases"]:
        assert 0 <= case["start"] == case["seed_frame"] <= case["display_start"] < case["end"] <= 481
        assert not any(a <= case["seed_frame"] < b for a, b in reserved), "Reserved label used as seed"
    fingerprints = {"manifest": {"path": str(path), "sha256": sha256_file(path)},
                    "journal": {"path": str(journal_path), "sha256": sha256_file(journal_path)},
                    "geometry": shot["geometry"], "video": {"path": manifest["source"]["path"], "sha256": manifest["source"]["sha256"]},
                    "caches": manifest["caches"],
                    "files": {str(p.relative_to(ROOT)): sha256_file(p) for p in
                              [*sorted((ROOT / "src/seima_mocap").glob("*.py")), ROOT / "scripts/stroke_guidance_checkpoint.py",
                               ROOT / "config/ball_detection_defaults.json", ROOT / "config/ball_tracking_defaults.json"]}}
    plan["input_provenance"] = fingerprints
    plan["created_utc"] = datetime.now(timezone.utc).isoformat()
    plan["background_policy"] = "unchanged baseline: 17 median samples from the validated fixed shot, no labels; offline context includes outside displayed windows"
    plan["label_selection"] = "Case A retains the longer baseline segment after seed 17 as a regression control; B/C follow existing diagnostic candidates, not verified contact labels"
    plan["remaining_reserved"] = [[100, 255], [390, 481]]
    if frozen:
        plan = frozen
        plan["preparation_resume"] = {"reason": "raw decoder stderr pipe saturated by nonmonotonic mux DTS warnings",
            "verified_warning_bytes_before": 4363, "verified_warning_bytes_after": 0,
            "fix": "microsecond rawvideo encoder timebase; no detector or physical parameter changes",
            "worker_source_hashes_authoritative": True, "completed_bundles_reused": True}
    atomic_json(output / "protocol.json", plan)
    atomic_json(output / "checkpoint.json", {"status": "preparing_stage1", "manual_validation_required": True,
                                             "later_stages_authorized": False})
    archive = output / "baseline_source.zip"
    if not archive.exists():
        subprocess.run(["git", "archive", "--format=zip", "--output", str(archive), plan["baseline_commit"], "src", "config"], cwd=ROOT, check=True)
        with zipfile.ZipFile(archive) as z:
            z.extractall(output / "baseline_source")
    atomic_json(output / "source_manifest_snapshot.json", manifest)
    # Labels are isolated from the worker input: workers only receive one seed.
    atomic_json(output / "labels_for_scoring_only.json", journal)
    models = load_cache(manifest["caches"]["racketvision"]["path"])
    bodies, _ = load_body_frames(manifest["caches"]["poses"]["path"], None, manifest["source"]["frame_count"],
                                 manifest["source"]["width"], manifest["source"]["height"])
    params = BallPhysicalParams.from_json(ROOT / "config/ball_tracking_defaults.json")
    for case in plan["cases"]:
        if resume and (output / f"{case['id']}_inputs.pkl").exists():
            existing = load_pickle(output / f"{case['id']}_inputs.pkl")
            expected_times = np.array(manifest["source"]["pts_s"][case["start"]:case["end"]])-manifest["source"]["pts_s"][case["start"]]
            assert np.array_equal(existing["times"], expected_times)
            expected_seed = next(a for a in journal["actions"] if a["action"] == "seed" and a["frame"] == case["seed_frame"])
            assert existing["seed"]["id"] == expected_seed["id"] and existing["seed"]["xy"] == expected_seed["xy"]
            print(f"Reusing verified prepared bundle {case['id']}", flush=True)
            continue
        print(f"Preparing only case {case['id']}: source [{case['start']},{case['end']})", flush=True)
        view = {**part, "start_frame": case["start"], "end_frame": case["end"]}
        times, evidence, local, contexts, scene, recovery = prepare_evidence(ROOT, manifest, view, models, bodies)
        recovery_values = [recovery(i) for i in range(len(times))]
        original = next(a for a in journal["actions"] if a["action"] == "seed" and a["frame"] == case["seed_frame"])
        seed = {k: original[k] for k in ("action", "xy", "kind", "id")}
        seed.update(frame=0, episode_start=0, trigger_frame=0)
        bundle = dict(times=times, evidence=evidence, models=local, contexts=contexts, scene=scene, params=params,
                      recovery=recovery_values, seed=seed)
        save_pickle(output / f"{case['id']}_inputs.pkl", bundle)
    # Freeze preparation hashes. All variants must consume these same bundles.
    atomic_json(output / "bundle_hashes.json", {c["id"]: sha256_file(output / f"{c['id']}_inputs.pkl") for c in plan["cases"]})
    return plan


def render_and_score(output):
    sys.path.insert(0, str(ROOT / "src"))
    from dataclasses import replace
    from collections import Counter
    import cv2
    import numpy as np
    from seima_mocap.video_io import atomic_json, write_timed_video, iter_frames, bind_verified_source_metadata
    from seima_mocap.table_geometry import sha256_file
    from seima_mocap.stroke_guidance import StrokeGuide, StrokePriorConfig
    from seima_mocap.interaction_tracking import fuse_ball_candidates
    plan = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "source_manifest_snapshot.json").read_text(encoding="utf-8"))
    source = manifest["source"]
    bind_verified_source_metadata(source["path"], source)
    labels = json.loads((output / "labels_for_scoring_only.json").read_text(encoding="utf-8"))["actions"]
    metrics, comparisons, mapping, comparison_times = [], [], [], []
    output_time = 0.
    colors = {"OBSERVED": (40, 240, 60), "PREDICTED": (0, 160, 255), "LOST": (70, 70, 255), "UNINITIALIZED": (160, 160, 160)}
    for case in plan["cases"]:
        data = load_pickle(output / f"{case['id']}_inputs.pkl")
        base = load_pickle(output / f"{case['id']}_baseline.pkl")
        guided = load_pickle(output / f"{case['id']}_h1_soft_region.pkl")
        traces = json.loads((output / f"{case['id']}_h1_soft_region_details.json").read_text())["candidate_trace"]
        guide = StrokeGuide(data["scene"].table_polygon_xy, StrokePriorConfig(**plan["stroke_parameters"]))
        priors = [guide.step(c) for c in data["contexts"]]
        start = case["display_start"]-case["start"]
        # Only now, after both variants have run, read positions of held-out labels.
        scoring = [a for a in labels if a["action"] == "seed" and case["display_start"] <= a["frame"] < case["end"]
                   and a["frame"] != case["seed_frame"]]
        sample_metrics = []
        for label in scoring:
            i = label["frame"]-case["start"]
            values = {"source_frame": label["frame"], "source_pts_s": source["pts_s"][label["frame"]],
                      "region_contains_label": priors[i].contains(label["xy"]), "prior_weight_at_label": priors[i].weight(label["xy"]),
                      "phases": [r.phase for r in priors[i].regions]}
            pool = data["evidence"][i]+data["recovery"][i]
            positions = [c.observation.pixel_xy for c in pool]+[c.pixel_xy for c in data["models"][i].ball_candidates]
            values["nearest_raw_candidate_px"] = float(min((np.linalg.norm(p-label["xy"]) for p in positions), default=1e9))
            for name, result in (("baseline", base), ("h1", guided)):
                track = result.tracks[i]
                values[name+"_status"] = track.status.value
                values[name+"_error_px"] = None if track.observed_xy is None else float(np.linalg.norm(track.observed_xy-label["xy"]))
            sample_metrics.append(values)
        row = {"case": case, "sparse_labels_only": sample_metrics,
               "baseline_states": dict(Counter(t.status.value for t in base.tracks[start:])),
               "h1_states": dict(Counter(t.status.value for t in guided.tracks[start:])),
               "baseline_requests": base.requests, "h1_requests": guided.requests,
               "phase_hypotheses": dict(Counter(r.phase for p in priors[start:] for r in p.regions)),
               "human_identity_labels": "pending; OBSERVED is not accuracy; no intervention-reduction claim"}
        metrics.append(row)
        for local_i, (source_i, raw) in enumerate(iter_frames(Path(source["path"]), case["display_start"], case["end"])):
            i = local_i+start
            prior = priors[i]
            pool = data["evidence"][i]+data["recovery"][i]
            # Display top visual/context candidates at the displayed result's own reference.
            panels = []
            for name, result in (("BASELINE", base), ("H1 REGION SUAVE", guided)):
                track = result.tracks[i]
                panel = cv2.resize(raw, (960, 540))
                scale = 960/source["width"]
                if name != "BASELINE":
                    for region in prior.regions:
                        cv2.ellipse(panel, tuple(np.round(np.array(region.center)*scale).astype(int)),
                                    tuple(np.round(np.array(region.radii)*scale).astype(int)), 0, 0, 360, (255, 220, 20), 2)
                # The same raw visual candidates on both panels, independent of
                # tracker state. Actual H1 post-fusion scores are saved in trace.
                observations = [c.observation for c in pool]+list(data["models"][i].ball_candidates)
                candidates = [{"xy": o.pixel_xy.tolist()} for o in sorted(observations, key=lambda o: -o.confidence)[:8]]
                for c in candidates:
                    cv2.circle(panel, tuple(np.round(np.array(c["xy"])*scale).astype(int)), 4, (255, 255, 255), 1)
                point = track.observed_xy if track.observed_xy is not None else track.predicted_xy
                if point is not None and track.status.value in ("OBSERVED", "PREDICTED"):
                    cv2.circle(panel, tuple(np.round(np.array(point)*scale).astype(int)), 10, colors[track.status.value], 2)
                if i == 0:
                    cv2.drawMarker(panel, tuple(np.round(np.array(data["seed"]["xy"])*scale).astype(int)), (255, 60, 220), cv2.MARKER_CROSS, 25, 2)
                canvas = cv2.copyMakeBorder(panel, 86, 80, 0, 0, cv2.BORDER_CONSTANT)
                lines = [f"{name} | caso {case['id']} {case['split']} | {track.status.value}",
                         f"Fotograma original {source_i} | PTS original {source['pts_s'][source_i]:.6f} s",
                         "Una semilla inicial; siguientes etiquetas ocultas; solicitudes de ayuda conservadas"]
                for n, text in enumerate(lines):
                    cv2.putText(canvas, text, (10, 23+n*25), cv2.FONT_HERSHEY_SIMPLEX, .50, (245, 245, 245), 1)
                description = " / ".join(r.player+":"+r.phase.replace("_hypothesis", "?") for r in prior.regions)
                cv2.putText(canvas, description[:125] if name != "BASELINE" else "Asociacion y parametros originales", (10, 648), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 220, 20), 1)
                cv2.putText(canvas, "Blanco=candidatos | Verde=observado (no identidad validada) | Naranja=predicho", (10, 673), cv2.FONT_HERSHEY_SIMPLEX, .46, (240, 240, 240), 1)
                panels.append(canvas)
            comparisons.append(np.hstack(panels))
            comparison_times.append(output_time)
            mapping.append({"comparison_frame": len(mapping), "case": case["id"], "source_frame": source_i,
                            "source_pts_s": source["pts_s"][source_i], "comparison_pts_s": output_time})
            output_time += source["pts_s"][source_i+1]-source["pts_s"][source_i]
        print(f"Scored {case['id']}: {len(sample_metrics)} sparse held-out points; human contact/identity check pending", flush=True)
    atomic_json(output / "metrics.json", metrics)
    atomic_json(output / "comparison_frame_map.json", mapping)
    write_timed_video(iter(comparisons), np.array(comparison_times), output / "comparacion_normal.mp4")
    write_timed_video(iter(comparisons), np.array(comparison_times)*4, output / "comparacion_lenta.mp4")
    # Comparison is edited windows: source gaps omitted; slow file explicitly has 4x PTS.
    original_fingerprints = plan["input_provenance"]
    for name in ("manifest", "journal", "geometry", "video"):
        value = original_fingerprints[name]
        assert sha256_file(value["path"]) == value["sha256"], f"Original changed: {name}"
    checkpoint = {"status": "awaiting_manual_validation", "stage": 1, "protocol": str(output / "protocol.json"),
                  "baseline_commit": plan["baseline_commit"], "configurations_tried": ["h1_soft_region"],
                  "development_cases": ["A", "B"], "reserved_scored_once": ["C"],
                  "remaining_reserved": plan["remaining_reserved"], "originals_unchanged": True,
                  "next_experiment": "Only after human review: decide whether ROI misses the ball or candidate identity fails; then one local-contrast or contact-association ablation, not H2 yet",
                  "automatic_continuation_authorized": False, "human_validation": "pending",
                  "videos": ["comparacion_normal.mp4", "comparacion_lenta.mp4"]}
    atomic_json(output / "checkpoint.json", checkpoint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=["baseline", "h1_soft_region"])
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--resume-preparation", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.worker:
        worker(output, args.worker)
        return
    if args.render_only:
        render_and_score(output)
        return
    prepare(output, resume=args.resume_preparation)
    for variant in ("baseline", "h1_soft_region"):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--output", str(output), "--worker", variant], check=True)
    render_and_score(output)
    print("STOP: stage 1 complete. Manual validation required before any further experiment.", flush=True)


if __name__ == "__main__":
    main()
