"""Deterministic replay engine independent of video decoding and UI."""
from __future__ import annotations

from dataclasses import dataclass

from .ball_video_tracking import ProjectedBallTracker, TrackFrame, TrackStatus
from .interaction_tracking import InteractionAwareBallTracker, InteractionMode, fuse_ball_candidates, smooth_event_gaps
from .manual_recovery import LossEpisodes, ManualRecovery, ManualSeed


@dataclass
class ReplayResult:
    tracks: list
    modes: list
    provenance: list
    requests: list
    ended: bool = False


def replay_tracking(times, evidence, models, contexts, scene, params, actions=(), *, stop_on_request=True,
                    recovery_candidates=None):
    if not len(times) == len(evidence) == len(models) == len(contexts):
        raise ValueError("Evidence, contexts and timestamps must have equal coverage")
    physical = ProjectedBallTracker(scene, params, use_physics=True)
    interaction = InteractionAwareBallTracker(ProjectedBallTracker(scene, params, use_physics=True), scene)
    recovery = ManualRecovery(interaction)
    episodes = LossEpisodes(times)
    tracks, modes, provenance, requests = [], [], [], []
    manual_priority = False
    active_seed = None
    segment = 0
    held_last = False
    ended = False
    for index, time in enumerate(times):
        new_action = next((a for a in actions if a["frame"] == index), None)
        cover = next((a for a in actions if a["episode_start"] <= index <= max(a["frame"], a["trigger_frame"])), None)
        held = next((a for a in actions if a["episode_start"] <= index
                     and (index < a["frame"] if a["action"] == "seed" else index <= a["frame"])), None)
        if held and not held_last:
            recovery.reset()
            physical.reset_for_discontinuity()
            segment += 1
        held_last = held is not None
        if new_action and new_action["action"] == "seed":
            active_seed = new_action
            manual_priority = True
            physical.reset_for_discontinuity()
            recovery.begin(ManualSeed(index, time, tuple(new_action["xy"]), new_action["kind"]))
            segment += 1
            episodes = LossEpisodes(times, start_frame=index)
        origin = {"segment_id": segment, "manual_seed_id": "" if active_seed is None else active_seed["id"],
                  "manual_seed_kind": "", "manual_seed_x_px": "", "manual_seed_y_px": "",
                  "recovery_state": recovery.state, "manual_action": "", "protected": False,
                  "assisted_segment": manual_priority}
        request = None
        if held:
            tracked = TrackFrame(time, TrackStatus.LOST, None, None, None, 0., 0)
            mode = InteractionMode.FREE_FLIGHT
            origin.update(manual_action="omitted_gap", protected=True)
            if index == held["frame"]:
                episodes = LossEpisodes(times, start_frame=min(index+1, len(times)-1))
        elif recovery.state == "pending":
            visual = evidence[index]
            if recovery_candidates is not None and time > recovery.seed.timestamp_s:
                visual = list(visual) + list(recovery_candidates(index))
            candidates = fuse_ball_candidates(visual, models[index].ball_candidates, contexts[index],
                                              predicted_xy=recovery.seed.xy)
            tracked = recovery.step(time, candidates)
            mode = InteractionMode.FREE_FLIGHT
            origin.update(recovery_state=recovery.state, protected=True)
            if index == recovery.seed.frame:
                origin.update(manual_seed_kind=recovery.seed.kind, manual_seed_x_px=recovery.seed.xy[0],
                              manual_seed_y_px=recovery.seed.xy[1], manual_action="seed")
            if recovery.state == "failed":
                request = {"episode_start": active_seed["episode_start"], "trigger_frame": index,
                           "inspection_frame": active_seed["frame"], "last_observed_frame": episodes.last_observed_frame,
                           "reason": "manual_recovery_timeout", "kind": active_seed["kind"]}
            elif recovery.state == "confirmed":
                episodes.step(index, tracked)
        elif recovery.state == "failed":
            tracked = TrackFrame(time, TrackStatus.LOST, None, None, None, 0., 0)
            mode = InteractionMode.FREE_FLIGHT
            origin["protected"] = True
        else:
            baseline = physical.step(time, evidence[index])
            visual = evidence[index]
            if manual_priority and recovery_candidates is not None:
                visual = list(visual) + list(recovery_candidates(index))
            result = interaction.step(time, visual, models[index].ball_candidates,
                                      contexts[index], anchor=None if manual_priority else baseline)
            tracked, mode = result.track, result.mode
            request = episodes.step(index, tracked)
        tracks.append(tracked)
        modes.append(mode)
        provenance.append(origin)
        if new_action and new_action["action"] == "end":
            ended = True
            break
        resolved_failure = request is not None and request["reason"] == "manual_recovery_timeout" and any(
            a["id"] != active_seed["id"] and a["episode_start"] <= index <= max(a["frame"], a["trigger_frame"])
            for a in actions)
        if request is not None and (cover is None or (request["reason"] == "manual_recovery_timeout" and not resolved_failure)):
            requests.append(request)
            if stop_on_request:
                break
    if len(tracks) == len(times) and recovery.state == "pending" and not ended:
        requests.append({"episode_start": active_seed["episode_start"], "trigger_frame": len(times)-1,
                         "inspection_frame": active_seed["frame"], "last_observed_frame": episodes.last_observed_frame,
                         "reason": "clip_ended_before_recovery", "kind": active_seed["kind"]})
    return ReplayResult(tracks, modes, provenance, requests, ended)


def smooth_replay(result, scene):
    """Smooth only ordinary spans within a segment, never across manual boundaries."""
    tracks = list(result.tracks)
    start = 0
    while start < len(tracks):
        if result.provenance[start]["protected"]:
            start += 1
            continue
        end = start+1
        while (end < len(tracks) and not result.provenance[end]["protected"]
               and result.provenance[end]["segment_id"] == result.provenance[start]["segment_id"]):
            end += 1
        tracks[start:end], _ = smooth_event_gaps(tracks[start:end], result.modes[start:end], scene)
        start = end
    return ReplayResult(tracks, result.modes, result.provenance, result.requests, result.ended)
