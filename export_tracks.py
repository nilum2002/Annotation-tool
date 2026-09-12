#!/usr/bin/env python3
# coding: utf-8
"""Turn an annotation JSON into CSVs for computing MOTA, IDF1 and HOTA.

    python3 export_tracks.py <annotations.json> [-o outdir]

Writes two files, because one table cannot honestly hold both:

  <name>.tracks.csv   one row per annotated person
  <name>.frames.csv   one row per reviewed scan, people or not

The second one is not optional. A CSV of annotations alone cannot express
"this scan was reviewed and held nobody", and a scorer that cannot tell that
apart from "this scan was never looked at" will count every unreviewed scan
against the tracker. Read frames.csv to know which scans to score.

Both carry the same `#` metadata header, including the matching criterion and
threshold, so a score stays reproducible without a companion script.

Needs no ROS and no bag. Everything comes out of the JSON.
"""

import argparse
import csv
import datetime
import json
import math
import os
import sys

SCHEMA = "laser-person-tracks/1"

TRACK_COLUMNS = [
    "frame",        # 1-based, contiguous over reviewed scans, for MOT tooling
    "scan_index",   # the scan's real index in the bag
    "t_ns",         # scan timestamp, nanoseconds, for aligning tracker output
    "track_id",     # unique within the bag, stable across frames
    "x", "y",       # metres in annotation_frame (the laser)
    "world_x", "world_y",   # metres in odom_frame, blank without odometry
    "consider",     # MOT convention: 1 scores normally, 0 means ignore
]

FRAME_COLUMNS = [
    "frame", "scan_index", "t_ns",
    "robot_x", "robot_y", "robot_yaw",  # pose in odom_frame, blank if absent
    "n_people",     # scored people in this scan
    "n_ignored",    # annotated but flagged ignore
]


def se2_apply(pose, x, y):
    c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
    return c * x - s * y + pose["x"], s * x + c * y + pose["y"]


def compose(outer, inner):
    x, y = se2_apply(outer, inner["x"], inner["y"])
    return {"x": x, "y": y, "yaw": outer["yaw"] + inner["yaw"]}


def header_lines(d, kind):
    m = d.get("match", {})
    return [
        ("schema", d.get("schema", "unknown")),
        ("table", kind),
        ("bag", d.get("bag", "")),
        ("scan_topic", d.get("scan_topic", "")),
        ("annotation_frame", d.get("annotation_frame", "")),
        ("odom_frame", d.get("odom_frame") or ""),
        ("base_frame", d.get("base_frame") or ""),
        ("match_criterion", m.get("criterion", "")),
        ("match_threshold_m", m.get("threshold_m", "")),
        ("generated", datetime.datetime.now().isoformat(timespec="seconds")),
    ]


def write_csv(path, columns, rows, meta):
    with open(path, "w", newline="") as f:
        for k, v in meta:
            f.write("# {}: {}\n".format(k, v))
        w = csv.writer(f)
        w.writerow(columns)
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("json", help="Annotation file written by anno_ros2.py.")
    p.add_argument("-o", "--outdir", default=None,
                   help="Where to write. Default: next to the JSON.")
    args = p.parse_args()

    with open(args.json) as f:
        d = json.load(f)
    if d.get("schema") != SCHEMA:
        print("Warning: schema is {!r}, expected {!r}. Continuing."
              .format(d.get("schema"), SCHEMA), file=sys.stderr)
    if "frames" not in d:
        sys.exit("This file has no 'frames' section, so it predates track ids. "
                 "Open it in anno_ros2.py once to migrate it.")

    l2b = d.get("laser_to_base") or {"x": 0.0, "y": 0.0, "yaw": 0.0}
    base = os.path.basename(args.json)
    for suffix in (".annotations.json", ".json"):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.json))
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir)

    track_rows, frame_rows = [], []
    ids, n_ignored_total = set(), 0
    # Sorted numerically, not as strings, or frame 10 lands before frame 9.
    for frame, si in enumerate(sorted(int(k) for k in d["frames"]), start=1):
        fr = d["frames"][str(si)]
        pose = fr.get("pose")
        laser_pose = compose(pose, l2b) if pose else None
        people = fr.get("people", [])

        for person in people:
            x, y = person["xy"]
            wx, wy = se2_apply(laser_pose, x, y) if laser_pose else ("", "")
            ignore = bool(person.get("ignore", False))
            ids.add(person["id"])
            n_ignored_total += ignore
            track_rows.append([
                frame, si, fr["t_ns"], person["id"],
                round(x, 4), round(y, 4),
                round(wx, 4) if wx != "" else "", round(wy, 4) if wy != "" else "",
                0 if ignore else 1,
            ])

        frame_rows.append([
            frame, si, fr["t_ns"],
            round(pose["x"], 4) if pose else "",
            round(pose["y"], 4) if pose else "",
            round(pose["yaw"], 6) if pose else "",
            sum(1 for q in people if not q.get("ignore")),
            sum(1 for q in people if q.get("ignore")),
        ])

    tracks_path = os.path.join(outdir, base + ".tracks.csv")
    frames_path = os.path.join(outdir, base + ".frames.csv")
    write_csv(tracks_path, TRACK_COLUMNS, track_rows, header_lines(d, "tracks"))
    write_csv(frames_path, FRAME_COLUMNS, frame_rows, header_lines(d, "frames"))

    scored = len(track_rows) - n_ignored_total
    print("{} reviewed scans, {} annotations ({} scored, {} ignored), {} tracks"
          .format(len(frame_rows), len(track_rows), scored, n_ignored_total, len(ids)))
    print("wrote", tracks_path)
    print("wrote", frames_path)
    if not any(r[6] != "" for r in track_rows) and track_rows:
        print("No odometry in this file, so world_x and world_y are blank.")


if __name__ == "__main__":
    main()
