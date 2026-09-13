#!/usr/bin/env python3
# coding: utf-8
"""Annotate people in the laser scans of a ROS 2 bag, reading the .mcap directly.

    python3 anno_ros2.py <bag> [--dry-run]

Shows the laser scan in one window and, if the bag has camera messages, the
matching video frame in a second window. Scans can be viewed in the sensor
frame (robot fixed at the origin, right for a stationary robot) or in the odom
frame (world fixed, robot drives through it, right for a moving robot). Press f
to switch. The odom view needs a nav_msgs/Odometry topic in the bag.

A person is one click. Annotations are always stored in the laser's own frame,
so they mean the same thing in either view, and are written to
<bag>.annotations.json.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

# ---------------------------------------------------------------- settings

# How many scans a batch spans, and which of them are actually shown.
BATCHSIZE = 100
TICKSKIP = 5
# Skip this many batches after each annotated one. 0 annotates the whole bag.
BATCHSKIP = 0

# Radius of the cursor circle, in metres. Also the erase radius.
CIRCRAD = 1.22 / 2

# Drop returns beyond this. None means use the laser's own range_max.
RANGE_CUTOFF = None

# In the odom view, also draw this many nearby scans in grey, to give the
# moving robot some context. 0 draws only the current scan.
CONTEXT_SCANS = 6

# Which way the robot marker points, as an offset on the odometry yaw. This
# only rotates the drawn arrow, never the scan data, so use it when the arrow
# disagrees with how the robot is actually built. 180 turns it around.
ROBOT_YAW_DEG = 180.0

# Horizontal field of view of the supporting camera, drawn as two dotted lines
# from the robot so you can see which laser returns the video can corroborate.
# None hides it. CAMERA_YAW_DEG is measured from the robot marker above, so the
# wedge follows it, and stays 0 for a camera that looks straight ahead. The
# apex sits at the robot origin rather than at the camera itself, because bags
# here carry no static transform for the camera frame.
CAMERA_FOV_DEG = 110.0
CAMERA_YAW_DEG = 0.0

# How a person is drawn. One click, one person.
MARKER, COLOUR, IGNORE_COLOUR = "o", "#50B948", "#B0752A"

SCHEMA = "laser-person-tracks/1"

# How ground truth gets matched to tracker output when scoring. Recorded in
# every file written, so a number a year old is still reproducible.
MATCH_CRITERION, MATCH_THRESHOLD_M = "euclidean_xy", 0.5

# On a new click, continue the nearest track from the previously annotated
# scan if it is within this distance. Compared in the odom frame when the bag
# has odometry, so a moving robot does not break association.
ASSOC_RADIUS_M = 0.7

# --------------------------------------------------------------------------

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import AxesWidget

import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan, Image, CompressedImage
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage

try:
    import cv2
except ImportError:
    cv2 = None


def stamp_ns(header):
    """A message's own capture time. Falls back to nothing if unset."""
    ns = header.stamp.sec * 10 ** 9 + header.stamp.nanosec
    return ns if ns > 0 else None


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))


class SE2(object):
    """A planar pose, and the transform it represents."""

    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x, self.y, self.theta = x, y, theta

    def apply(self, xs, ys):
        c, s = math.cos(self.theta), math.sin(self.theta)
        return c * xs - s * ys + self.x, s * xs + c * ys + self.y

    def invert_point(self, x, y):
        c, s = math.cos(self.theta), math.sin(self.theta)
        dx, dy = x - self.x, y - self.y
        return c * dx + s * dy, -s * dx + c * dy

    def compose(self, other):
        x, y = self.apply(np.array(other.x), np.array(other.y))
        return SE2(float(x), float(y), self.theta + other.theta)


def open_bag(uri):
    last = None
    for storage_id in ("", "mcap", "sqlite3"):
        try:
            r = rosbag2_py.SequentialReader()
            r.open(rosbag2_py.StorageOptions(uri=uri, storage_id=storage_id),
                   rosbag2_py.ConverterOptions("", ""))
            return r
        except Exception as e:
            last = e
    raise RuntimeError("Could not open bag {}: {}".format(uri, last))


class Bag(object):
    """Everything we need from the bag, read in one pass."""

    def __init__(self, path, scan_topic=None, image_topic=None, odom_topic=None,
                 max_images=4000):
        self.path = path
        types = {t.name: t.type for t in open_bag(path).get_all_topics_and_types()}

        def pick(explicit, suffixes, what):
            if explicit:
                if explicit not in types:
                    sys.exit("No topic {} in bag. Topics: {}".format(explicit, sorted(types)))
                return explicit
            found = [t for t, ty in sorted(types.items())
                     if any(ty.endswith(s) for s in suffixes)]
            if found and len(found) > 1:
                print("Several {} topics {}, using {}".format(what, found, found[0]))
            return found[0] if found else None

        self.scan_topic = pick(scan_topic, ["msg/LaserScan"], "laser")
        self.image_topic = pick(image_topic, ["CompressedImage", "msg/Image"], "camera")
        self.odom_topic = pick(odom_topic, ["msg/Odometry"], "odometry")
        if self.scan_topic is None:
            sys.exit("Bag has no LaserScan topic. Topics: {}".format(sorted(types)))

        wanted = [t for t in (self.scan_topic, self.image_topic, self.odom_topic) if t]
        wanted.append("/tf_static")

        reader = open_bag(path)
        reader.set_filter(rosbag2_py.StorageFilter(topics=wanted))

        self.scan_t, self.sweep_ns, ranges, angles = [], [], [], []
        self.unstamped = 0
        self.odom_t, odom_xy, odom_yaw = [], [], []
        self.img_t, self.img_jpg = [], []
        self.proto = None
        self.odom_frame, self.base_frame = None, None
        static = []
        dropped_images = 0

        nread = 0
        while reader.has_next():
            topic, data, ts = reader.read_next()
            nread += 1
            if nread % 2000 == 0:
                print("  ... {} messages, {} scans, {} frames"
                      .format(nread, len(ranges), len(self.img_jpg)))
                sys.stdout.flush()
            if topic == self.scan_topic:
                m = deserialize_message(data, LaserScan)
                self.proto = self.proto or m
                r = np.asarray(m.ranges, dtype=np.float64)
                r[~np.isfinite(r)] = np.nan
                r[(r < m.range_min) | (r > m.range_max)] = np.nan
                # Header stamp, not the bag receive time. The two differ by a
                # full sweep here, because the driver stamps the first ray and
                # publishes once the revolution finishes.
                t = stamp_ns(m.header)
                if t is None:
                    t, self.unstamped = ts, self.unstamped + 1
                self.scan_t.append(t)
                sweep = m.scan_time or (len(r) * m.time_increment)
                self.sweep_ns.append(int(sweep * 1e9))
                ranges.append(r)
                # Per ROS semantics the i-th beam is at angle_min + i*increment.
                # Do not derive it from angle_max, which drifts scan to scan.
                angles.append(m.angle_min
                              + np.arange(len(r)) * m.angle_increment)
            elif topic == self.odom_topic:
                m = deserialize_message(data, Odometry)
                self.odom_frame = self.odom_frame or m.header.frame_id
                self.base_frame = self.base_frame or m.child_frame_id
                p = m.pose.pose.position
                self.odom_t.append(stamp_ns(m.header) or ts)
                odom_xy.append((p.x, p.y))
                odom_yaw.append(yaw_of(m.pose.pose.orientation))
            elif topic == self.image_topic:
                if len(self.img_jpg) >= max_images:
                    dropped_images += 1
                    continue
                jpg, hdr = self._to_jpg(data, types[self.image_topic])
                if jpg is not None:
                    self.img_t.append(hdr or ts)
                    self.img_jpg.append(jpg)
            elif topic == "/tf_static":
                static.extend(deserialize_message(data, TFMessage).transforms)

        if not ranges:
            sys.exit("No messages on {}.".format(self.scan_topic))

        # Kept as lists, not one 2-D array. A spinning lidar samples whatever
        # its motor speed allows, so beam count and angular step both vary from
        # revolution to revolution and no rectangular array fits.
        self.ranges = ranges
        self.angles = angles
        self.beam_counts = (min(len(r) for r in ranges), max(len(r) for r in ranges))
        self.scan_t = np.asarray(self.scan_t, dtype=np.int64)
        self.img_t = np.asarray(self.img_t, dtype=np.int64)
        # A spinning lidar measures across the whole revolution, so the instant
        # that best represents a scan is its middle, not its first ray. This is
        # what the camera frame should be matched against.
        self.scan_mid = self.scan_t + np.asarray(self.sweep_ns, np.int64) // 2
        self.range_max = RANGE_CUTOFF or self.proto.range_max

        self.laser_to_base = self._chain(static, self.proto.header.frame_id, self.base_frame)
        self.poses = self._poses_at_scans(odom_xy, odom_yaw)

        if dropped_images:
            print("Note: kept the first {} images, skipped {} (raise --max-images)"
                  .format(len(self.img_jpg), dropped_images))

    def _to_jpg(self, data, msgtype):
        """(jpeg bytes, capture time) for one image message."""
        if msgtype.endswith("CompressedImage"):
            m = deserialize_message(data, CompressedImage)
            return bytes(m.data), stamp_ns(m.header)
        if cv2 is None:
            return None, None
        m = deserialize_message(data, Image)
        buf = np.frombuffer(m.data, np.uint8)
        if m.encoding in ("rgb8", "bgr8"):
            img = buf.reshape(m.height, m.step)[:, :m.width * 3].reshape(m.height, m.width, 3)
            if m.encoding == "rgb8":
                img = img[:, :, ::-1]
        elif m.encoding in ("mono8", "8UC1"):
            img = buf.reshape(m.height, m.step)[:, :m.width]
        else:
            return None, None
        ok, enc = cv2.imencode(".jpg", img)
        return (bytes(enc) if ok else None), stamp_ns(m.header)

    def _chain(self, static, src, dst):
        """Compose the static transforms leading from `src` up to `dst`."""
        if not src or not dst or src == dst:
            return SE2()
        parent = {tr.child_frame_id: tr for tr in static}
        out, frame = SE2(), src
        for _ in range(len(parent) + 1):
            tr = parent.get(frame)
            if tr is None:
                break
            t = tr.transform.translation
            out = SE2(t.x, t.y, yaw_of(tr.transform.rotation)).compose(out)
            frame = tr.header.frame_id
            if frame == dst:
                return out
        print("Warning: no static transform chain {} -> {}, assuming they coincide."
              .format(src, dst))
        return SE2()

    def _poses_at_scans(self, odom_xy, odom_yaw):
        """Robot pose in the odom frame at each scan's timestamp, or None."""
        if len(odom_xy) < 2:
            return None
        ot = np.asarray(self.odom_t, dtype=np.float64)
        oxy = np.asarray(odom_xy)
        # Unwrap before interpolating so we never average across the +-pi seam.
        oyaw = np.unwrap(np.asarray(odom_yaw))
        st = self.scan_t.astype(np.float64)
        return [SE2(x, y, th) for x, y, th in zip(np.interp(st, ot, oxy[:, 0]),
                                                  np.interp(st, ot, oxy[:, 1]),
                                                  np.interp(st, ot, oyaw))]

    def scan_xy(self, i, base_frame=True):
        """Scan i as x,y. In the base frame by default, else raw laser frame."""
        r = self.ranges[i].copy()
        r[r > self.range_max] = np.nan
        a = self.angles[i]
        x, y = r * np.cos(a), r * np.sin(a)
        return self.laser_to_base.apply(x, y) if base_frame else (x, y)

    def laser_to_odom(self, i):
        """Transform taking a point in the laser frame to the odom frame."""
        if self.poses is None:
            return None
        return self.poses[i].compose(self.laser_to_base)

    def image_for(self, i):
        """(rgb, seconds_off) for the frame nearest scan i, or (None, None)."""
        if not len(self.img_t) or cv2 is None:
            return None, None
        j = int(np.abs(self.img_t - self.scan_mid[i]).argmin())
        img = cv2.imdecode(np.frombuffer(self.img_jpg[j], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None, None
        return img[:, :, ::-1], (self.img_t[j] - self.scan_mid[i]) / 1e9


class MouseCircle(AxesWidget):
    """The cursor circle, drawn by blitting so it can follow the mouse."""

    def __init__(self, ax, radius, **props):
        AxesWidget.__init__(self, ax)
        self.connect_event("motion_notify_event", self.onmove)
        self.connect_event("draw_event", self.storebg)
        props.update(animated=True, radius=radius)
        self.circ = plt.Circle((0, 0), **props)
        self.ax.add_artist(self.circ)
        self.background = None

    def set_radius(self, r):
        self.circ.set_radius(r)

    def storebg(self, e):
        if not self.ignore(e):
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def onmove(self, e):
        if self.ignore(e) or not self.canvas.widgetlock.available(self):
            return
        # Off the axes there is no data coordinate to put the circle at.
        # Matplotlib used to treat a None centre as invisible, but since 3.x it
        # raises inside the patch transform, so hide the circle explicitly.
        if e.xdata is None or e.ydata is None:
            if self.circ.get_visible():
                self.circ.set_visible(False)
                self.update_circle()
            return
        self.circ.set_visible(True)
        self.circ.center = (e.xdata, e.ydata)
        self.update_circle()

    def update_circle(self):
        if self.background is not None:
            self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.circ)
        self.canvas.blit(bbox=None)


class Annotator(object):
    def __init__(self, bag, outpath, dryrun=False, world=False):
        self.bag = bag
        self.outpath = outpath
        self.dryrun = dryrun
        self.world = world and bag.poses is not None
        self.b = self.i = 0

        n = len(bag.scan_t)
        self.batches = []
        step = BATCHSIZE * (BATCHSKIP + 1)
        for start in range(0, max(1, n - 1), step):
            idx = list(range(start, min(start + BATCHSIZE, n), TICKSKIP))
            if idx:
                self.batches.append(idx)

        # Labels and reviewed-set are keyed by scan index, so they survive any
        # change to the batching above.
        # scan index -> list of {"id", "x", "y", "ignore"}. Positions are in
        # the laser frame, ids are unique within the bag.
        self.labels = {}
        self.reviewed = set()
        self.next_id = 1
        self.pending_id = None  # Digits typed, spent on the next click.
        self.load()

        self.fig = plt.figure("laser", figsize=(9, 9))
        self.ax = self.fig.add_subplot(111)
        for ev, fn in (("button_press_event", self.click),
                       ("scroll_event", self.scroll),
                       ("key_press_event", self.key)):
            self.fig.canvas.mpl_connect(ev, fn)
        self.circ = MouseCircle(self.ax, radius=CIRCRAD, linewidth=1, fill=False)

        self.imfig = self.imax = self.imshow = None
        if len(bag.img_t):
            self.imfig = plt.figure("video", figsize=(7, 4.5))
            self.imax = self.imfig.add_subplot(111)
            self.imax.axis("off")
            self.imfig.canvas.mpl_connect("key_press_event", self.key)
            self.imfig.tight_layout()

        plt.pause(0.001)
        self.replot()

    # ---------------------------------------------------------- persistence

    def scan_index(self):
        return self.batches[self.b][self.i]

    def bucket(self, si=None):
        """The list of people marked in a scan, created empty on first look."""
        si = self.scan_index() if si is None else si
        return self.labels.setdefault(si, [])

    def new_id(self):
        i = self.next_id
        self.next_id += 1
        return i

    def odom_xy(self, si, x, y):
        """A laser-frame point in the odom frame, or unchanged without odometry."""
        tf = self.bag.laser_to_odom(si)
        if tf is None:
            return x, y
        ox, oy = tf.apply(np.asarray(x), np.asarray(y))
        return float(ox), float(oy)

    def previous_annotated(self, si):
        earlier = [s for s in self.reviewed if s < si]
        return max(earlier) if earlier else None

    def suggest_id(self, si, x, y):
        """Continue the nearest track from the previously annotated scan.

        Association is done in the odom frame, because a laser-frame position
        moves with the robot even when the person stands still. An id already
        used in this scan is skipped, so one track cannot be claimed twice.
        """
        prev = self.previous_annotated(si)
        if prev is None:
            return self.new_id()
        taken = {p["id"] for p in self.labels.get(si, [])}
        px, py = self.odom_xy(si, x, y)
        best, best_d = None, ASSOC_RADIUS_M
        for q in self.labels.get(prev, []):
            if q["id"] in taken:
                continue
            qx, qy = self.odom_xy(prev, q["x"], q["y"])
            d = math.hypot(px - qx, py - qy)
            if d < best_d:
                best, best_d = q["id"], d
        return best if best is not None else self.new_id()

    def nearest(self, x, y):
        best, best_d = None, CIRCRAD
        for p in self.bucket():
            d = math.hypot(x - p["x"], y - p["y"])
            if d < best_d:
                best, best_d = p, d
        return best

    def save(self):
        if self.dryrun:
            return
        d = os.path.dirname(self.outpath)
        if d and not os.path.isdir(d):
            os.makedirs(d)
        # Only scans we actually opened are written, so an interrupted session
        # does not claim the rest of the bag was reviewed and found empty.
        l2b = self.bag.laser_to_base
        out = {
            "schema": SCHEMA,
            "bag": os.path.abspath(self.bag.path),
            "scan_topic": self.bag.scan_topic,
            "annotation_frame": self.bag.proto.header.frame_id,
            "odom_frame": self.bag.odom_frame,
            "base_frame": self.bag.base_frame,
            "laser_to_base": {"x": round(l2b.x, 6), "y": round(l2b.y, 6),
                              "yaw": round(l2b.theta, 6)},
            "match": {"criterion": MATCH_CRITERION,
                      "threshold_m": MATCH_THRESHOLD_M},
            "note": "one entry per person, xy in metres in the annotation_frame; "
                    "ignore means present but not fairly detectable in the laser",
            # Every reviewed scan appears, empty ones included, so that a scan
            # with no people counts as evidence rather than as a gap.
            "frames": {str(si): {
                "t_ns": int(self.bag.scan_t[si]),
                "pose": (None if self.bag.poses is None else
                         {"x": round(self.bag.poses[si].x, 6),
                          "y": round(self.bag.poses[si].y, 6),
                          "yaw": round(self.bag.poses[si].theta, 6)}),
                "people": [{"id": p["id"],
                            "xy": [round(p["x"], 4), round(p["y"], 4)],
                            "ignore": bool(p["ignore"])}
                           for p in self.labels.get(si, [])],
            } for si in sorted(self.reviewed)},
        }
        tmp = self.outpath + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=1)
        os.replace(tmp, self.outpath)  # So a crash mid-write cannot lose the file.

    def load(self):
        try:
            with open(self.outpath) as f:
                d = json.load(f)
        except (IOError, ValueError):
            return
        if "frames" in d:
            for si, fr in d["frames"].items():
                si = int(si)
                self.reviewed.add(si)
                self.labels[si] = [{"id": int(p["id"]),
                                    "x": float(p["xy"][0]), "y": float(p["xy"][1]),
                                    "ignore": bool(p.get("ignore", False))}
                                   for p in fr.get("people", [])]
        else:
            # Files written before tracks existed: points with no identity.
            # Each one becomes its own id, which is honest but useless for
            # IDSW, so say so rather than implying the tracks are real.
            self.reviewed = set(d.get("reviewed", []))
            for si, pts in d.get("labels", {}).items():
                if isinstance(pts, dict):
                    pts = pts.get("person", [])
                self.labels[int(si)] = [{"id": self.new_id(), "x": float(a),
                                         "y": float(b), "ignore": False}
                                        for a, b in pts]
            if self.labels:
                print("This file predates track ids. Every point was given a "
                      "fresh id, so identities are NOT continuous. Re-link them "
                      "before computing MOTA.")
        self.next_id = max([p["id"] for ps in self.labels.values() for p in ps]
                           + [0]) + 1
        npeople = sum(len(v) for v in self.labels.values())
        print("Resuming: {} scans reviewed, {} annotations, next id {}."
              .format(len(self.reviewed), npeople, self.next_id))

    # -------------------------------------------------------------- drawing

    def to_display(self, i, xs, ys):
        """Laser-frame points to whatever frame we are currently drawing in."""
        if self.world:
            return self.bag.laser_to_odom(i).apply(np.asarray(xs), np.asarray(ys))
        return self.bag.laser_to_base.apply(np.asarray(xs), np.asarray(ys))

    def from_display(self, x, y):
        tf = self.bag.laser_to_odom(self.scan_index()) if self.world else self.bag.laser_to_base
        return tf.invert_point(x, y)

    def replot(self):
        si = self.scan_index()
        self.reviewed.add(si)
        self.bucket()

        self.ax.clear()
        rmax = self.bag.range_max

        if self.world:
            # Neighbouring scans in grey so a moving robot has some context.
            for j in self.batches[self.b][max(0, self.i - CONTEXT_SCANS):
                                          self.i + CONTEXT_SCANS + 1]:
                if j != si:
                    x, y = self.bag.scan_xy(j, base_frame=False)
                    x, y = self.bag.laser_to_odom(j).apply(x, y)
                    self.ax.scatter(x, y, s=4, color="#BBBBBB", alpha=0.35, lw=0)
            px = [p.x for p in self.bag.poses]
            py = [p.y for p in self.bag.poses]
            self.ax.plot(px, py, "-", color="#777777", lw=1, alpha=0.8)

        x, y = self.bag.scan_xy(si, base_frame=False)
        xd, yd = self.to_display(si, x, y)
        self.ax.scatter(xd, yd, s=10, color="#E24A33", alpha=0.7, lw=0)

        # The robot, and where its nose points.
        pose = self.bag.poses[si] if self.world else SE2()
        heading = pose.theta + math.radians(ROBOT_YAW_DEG)
        self.ax.plot([pose.x], [pose.y], marker="s", ms=6, color="k")
        self.ax.arrow(pose.x, pose.y, 0.6 * math.cos(heading), 0.6 * math.sin(heading),
                      head_width=0.18, color="k", length_includes_head=True)

        if CAMERA_FOV_DEG:
            half = math.radians(CAMERA_FOV_DEG) / 2.0
            axis = heading + math.radians(CAMERA_YAW_DEG)
            for edge in (axis - half, axis + half):
                self.ax.plot([pose.x, pose.x + rmax * math.cos(edge)],
                             [pose.y, pose.y + rmax * math.sin(edge)],
                             ls=":", lw=1.0, color="k", alpha=0.55)

        people = self.bucket()
        for p in people:
            dx, dy = self.to_display(si, np.asarray(p["x"]), np.asarray(p["y"]))
            dx, dy = float(dx), float(dy)
            colour = IGNORE_COLOUR if p["ignore"] else COLOUR
            self.ax.scatter([dx], [dy], marker=MARKER, s=90, facecolors="none",
                            edgecolors=colour, linewidths=1.6,
                            linestyle="--" if p["ignore"] else "-")
            self.ax.annotate("{}{}".format(p["id"], "*" if p["ignore"] else ""),
                             (dx, dy), textcoords="offset points",
                             xytext=(9, 6), fontsize=8, color=colour)

        if self.world:
            self.ax.set_xlim(pose.x - rmax, pose.x + rmax)
            self.ax.set_ylim(pose.y - rmax, pose.y + rmax)
        else:
            self.ax.set_xlim(-rmax, rmax)
            self.ax.set_ylim(-rmax, rmax)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(alpha=0.15)

        t = (self.bag.scan_t[si] - self.bag.scan_t[0]) / 1e9
        nign = sum(1 for p in people if p["ignore"])
        counts = "{} tracked".format(len(people) - nign)
        if nign:
            counts += ", {} ignored".format(nign)
        if self.pending_id is not None:
            counts += "   next click = id {}".format(self.pending_id)
        self.fig.suptitle("batch {}/{}  frame {}/{}  scan {}  t={:.1f}s  [{} view]  {}"
                          .format(self.b + 1, len(self.batches), self.i + 1,
                                  len(self.batches[self.b]), si, t,
                                  "odom" if self.world else "sensor", counts))
        self.fig.canvas.draw()
        self.circ.update_circle()
        self.draw_image(si)

    def draw_image(self, si):
        if self.imax is None:
            return
        img, off = self.bag.image_for(si)
        if img is None:
            return
        if self.imshow is None:
            self.imshow = self.imax.imshow(img, interpolation="nearest")
        else:
            self.imshow.set_data(img)
            if self.imshow.get_array().shape != img.shape:
                self.imax.set_xlim(0, img.shape[1])
                self.imax.set_ylim(img.shape[0], 0)
        # A large offset means the nearest frame is not really this moment.
        self.imax.set_title("scan {}   camera {:+.2f}s away".format(si, off),
                            fontsize=9, color="k" if abs(off) < 0.15 else "#C0392B")
        self.imfig.canvas.draw_idle()

    # ---------------------------------------------------------- interaction

    def ignore(self, e):
        tb = getattr(e.canvas, "toolbar", None)
        return tb is not None and getattr(tb, "mode", "") != ""

    def click(self, e):
        if self.ignore(e) or e.xdata is None or e.ydata is None:
            return
        x, y = self.from_display(e.xdata, e.ydata)
        if e.button == 1:
            pid, self.pending_id = self.pending_id, None
            if pid is None:
                pid = self.suggest_id(self.scan_index(), x, y)
            self.bucket().append({"id": pid, "x": x, "y": y, "ignore": False})
            self.next_id = max(self.next_id, pid + 1)
        elif e.button in (2, 3):
            self.clear(x, y)
        self.replot()

    def scroll(self, e):
        if self.ignore(e):
            return
        for _ in range(int(abs(e.step)) or 1):
            self.next_frame() if e.button == "down" else self.prev_frame()
        self.replot()

    def key(self, e):
        if self.ignore(e):
            return
        if e.key in ("right", "d"):
            self.next_frame()
        elif e.key in ("left", "a"):
            self.prev_frame()
        elif e.key in ("down", "s", "pagedown"):
            self.next_batch()
        elif e.key in ("up", "w", "pageup"):
            self.prev_batch()
        elif e.key == "f":
            if self.bag.poses is None:
                print("This bag has no odometry, so there is no odom view.")
            else:
                self.world = not self.world
        elif e.key == "c":
            if e.xdata is not None and e.canvas is self.fig.canvas:
                self.clear(*self.from_display(e.xdata, e.ydata))
        elif e.key == "i":
            if e.xdata is not None and e.canvas is self.fig.canvas:
                p = self.nearest(*self.from_display(e.xdata, e.ydata))
                if p is not None:
                    p["ignore"] = not p["ignore"]
        elif e.key == "n":
            self.pending_id = self.new_id()
        elif e.key == "escape":
            self.pending_id = None
        elif e.key is not None and e.key.isdigit():
            # Digits accumulate, so 1 then 2 means id 12, not id 2.
            self.pending_id = (self.pending_id or 0) * 10 + int(e.key)
        else:
            return
        self.replot()

    def clear(self, x, y):
        self.labels[self.scan_index()] = [
            p for p in self.bucket()
            if math.hypot(x - p["x"], y - p["y"]) > CIRCRAD]

    def next_frame(self):
        self.i = min(len(self.batches[self.b]) - 1, self.i + 1)

    def prev_frame(self):
        self.i = max(0, self.i - 1)

    def next_batch(self):
        self.save()
        self.i = 0
        self.b = (self.b + 1) % len(self.batches)

    def prev_batch(self):
        self.save()
        self.i = 0
        self.b = (self.b - 1) % len(self.batches)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bag", help="Bag directory or .mcap file.")
    p.add_argument("-o", "--out", default=None,
                   help="Annotation file. Default: <bag>.annotations.json next to the bag.")
    p.add_argument("-n", "--dry-run", action="store_true", help="Never write anything.")
    p.add_argument("--world", action="store_true", help="Start in the odom view.")
    p.add_argument("--scan-topic", default=None)
    p.add_argument("--image-topic", default=None)
    p.add_argument("--odom-topic", default=None)
    p.add_argument("--range", type=float, default=None,
                   help="Clip and frame the view at this many metres.")
    p.add_argument("--camera-fov", type=float, default=None,
                   help="Camera field of view in degrees. 0 hides the wedge.")
    p.add_argument("--camera-yaw", type=float, default=None,
                   help="Rotate the wedge off the robot marker, in degrees.")
    p.add_argument("--robot-yaw", type=float, default=None,
                   help="Rotate the robot marker off the odometry yaw, in "
                        "degrees. The wedge follows it. 180 mirrors both.")
    p.add_argument("--tickskip", type=int, default=None,
                   help="Annotate every Nth scan. 1 annotates every scan. "
                        "Lowering it keeps work already done, since "
                        "annotations are keyed by scan index.")
    p.add_argument("--batchsize", type=int, default=None,
                   help="How many scans one batch spans.")
    p.add_argument("--batchskip", type=int, default=None,
                   help="Batches to skip after each annotated one. 0 covers "
                        "the whole bag.")
    p.add_argument("--max-images", type=int, default=4000,
                   help="Cap on frames held in memory.")
    args = p.parse_args()

    print("Reading {} ...".format(args.bag))
    bag = Bag(args.bag, args.scan_topic, args.image_topic, args.odom_topic,
              max_images=args.max_images)
    if args.range:
        bag.range_max = args.range
    if args.camera_fov is not None:
        globals()["CAMERA_FOV_DEG"] = args.camera_fov or None
    if args.camera_yaw is not None:
        globals()["CAMERA_YAW_DEG"] = args.camera_yaw
    if args.robot_yaw is not None:
        globals()["ROBOT_YAW_DEG"] = args.robot_yaw
    for flag, name in (("tickskip", "TICKSKIP"), ("batchsize", "BATCHSIZE"),
                       ("batchskip", "BATCHSKIP")):
        v = getattr(args, flag)
        if v is not None:
            if name != "BATCHSKIP" and v < 1:
                sys.exit("--{} must be at least 1.".format(flag))
            globals()[name] = v
    lo, hi = bag.beam_counts
    beams = "{} beams".format(lo) if lo == hi else "{}-{} beams".format(lo, hi)
    print("{} scans of {}, {} camera frames, odometry: {}".format(
        len(bag.scan_t), beams, len(bag.img_t),
        "yes" if bag.poses is not None else "no"))
    if bag.poses is None:
        print("No odometry topic, so only the sensor view is available.")
    if not len(bag.img_t):
        print("No camera messages, so no video window.")

    out = args.out or os.path.normpath(args.bag).rstrip("/") + ".annotations.json"
    a = Annotator(bag, out, dryrun=args.dry_run, world=args.world)
    print("keys: a/d frame, w/s batch, f sensor/odom view, c erase under cursor,")
    print("      i toggle ignore under cursor, n force a new id, digits set the")
    print("      id for the next click, escape cancels it")
    print("mouse: left click a person, right or middle click to erase, wheel scrubs")
    print("ids continue automatically from the previous annotated scan when a")
    print("click lands within {} m of a track, measured in the odom frame."
          .format(ASSOC_RADIUS_M))
    plt.show()
    a.save()
    if args.dry_run:
        print("Dry run, nothing written. Reviewed {} scans.".format(len(a.reviewed)))
    else:
        print("Saved {} reviewed scans to {}".format(len(a.reviewed), out))


if __name__ == "__main__":
    main()
