#!/usr/bin/env python3
# coding: utf-8
"""Dump a ROS 2 bag into the layout anno1602.py expects.

    <outdir>/<name>.csv        one row per scan: seq,r0,r1,...,rN,
    <outdir>/<name>_dir/<seq>.jpg   camera frame nearest in time to that scan

Scans are numbered 0..N-1 and that index is the `seq` used everywhere.
Each image is filed under the seq of the temporally nearest scan, which is
what lets anno1602's imload() find a frame for a scan via its +-1, +-2 search.

Ranges are rolled so that the middle column is straight ahead, because
anno1602 assumes a zero-centred laser (linspace(-FoV/2, +FoV/2)).
"""

import argparse
import math
import os
import shutil
import sys

import numpy as np

import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan, Image, CompressedImage

import cv2


def open_bag(uri):
    for storage_id in ("", "mcap", "sqlite3"):
        try:
            r = rosbag2_py.SequentialReader()
            r.open(rosbag2_py.StorageOptions(uri=uri, storage_id=storage_id),
                   rosbag2_py.ConverterOptions("", ""))
            return r
        except Exception:
            continue
    raise RuntimeError("Could not open bag at " + uri)


def topic_types(reader):
    return {t.name: t.type for t in reader.get_all_topics_and_types()}


def decode_image(msg, msgtype):
    if msgtype.endswith("CompressedImage"):
        return cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
    # Raw Image. Only the encodings we are likely to meet.
    h, w = msg.height, msg.width
    buf = np.frombuffer(msg.data, np.uint8).reshape(h, msg.step // 1)[:, :w * 3 if msg.encoding in ("rgb8", "bgr8") else w]
    if msg.encoding == "rgb8":
        return cv2.cvtColor(buf.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    if msg.encoding == "bgr8":
        return buf.reshape(h, w, 3)
    if msg.encoding in ("mono8", "8UC1"):
        return cv2.cvtColor(buf.reshape(h, w), cv2.COLOR_GRAY2BGR)
    raise RuntimeError("Unsupported image encoding: " + msg.encoding)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bag", help="Path to the bag directory or .mcap file.")
    p.add_argument("outdir", help="Where to write, i.e. anno1602's basedir.")
    p.add_argument("--name", default=None,
                   help="Basename of the dump. Default: the bag's directory name.")
    p.add_argument("--scan-topic", default="/scan")
    p.add_argument("--image-topic", default=None,
                   help="Default: the first Image or CompressedImage topic found.")
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--no-recentre", action="store_true",
                   help="Keep the raw range order instead of rolling it so that "
                        "the middle column points forward.")
    args = p.parse_args()

    name = args.name or os.path.basename(os.path.normpath(args.bag)).replace(".mcap", "")
    outdir = args.outdir.rstrip("/") + "/"
    imgdir = "{}{}_dir/".format(outdir, name)
    os.makedirs(imgdir, exist_ok=True)

    types = topic_types(open_bag(args.bag))
    if args.scan_topic not in types:
        sys.exit("No topic {} in bag. Topics: {}".format(args.scan_topic, sorted(types)))

    imgtopic = args.image_topic
    if imgtopic is None:
        cands = [t for t, ty in sorted(types.items())
                 if ty.endswith("CompressedImage") or ty.endswith("msg/Image")]
        imgtopic = cands[0] if cands else None
        if imgtopic:
            print("Using image topic {}".format(imgtopic))
    if imgtopic and imgtopic not in types:
        sys.exit("No topic {} in bag.".format(imgtopic))

    # Pass 1: the scans, which define the seq numbering.
    reader = open_bag(args.bag)
    reader.set_filter(rosbag2_py.StorageFilter(topics=[args.scan_topic]))
    stamps, rows, proto = [], [], None
    while reader.has_next():
        _, data, ts = reader.read_next()
        m = deserialize_message(data, LaserScan)
        proto = proto or m
        r = np.asarray(m.ranges, dtype=np.float64)
        r[~np.isfinite(r)] = np.nan
        r[(r < m.range_min) | (r > m.range_max)] = np.nan
        if not args.no_recentre:
            r = np.roll(r, len(r) // 2)
        stamps.append(ts)
        rows.append(r)

    if not rows:
        sys.exit("No messages on {}.".format(args.scan_topic))
    if len({len(r) for r in rows}) != 1:
        sys.exit("Scans have varying lengths, anno1602 needs a fixed-width CSV.")

    csvpath = "{}{}.csv".format(outdir, name)
    with open(csvpath, "w") as f:
        for seq, r in enumerate(rows):
            # Trailing comma on purpose: anno1602 drops the last column.
            f.write("{},{},\n".format(seq, ",".join("nan" if np.isnan(v) else "{:.4f}".format(v)
                                                    for v in r)))
    print("Wrote {} scans of {} beams to {}".format(len(rows), len(rows[0]), csvpath))

    # Pass 2: the images, each filed under its nearest scan's seq.
    nimg = 0
    have = set()
    if imgtopic:
        stamps = np.asarray(stamps, dtype=np.int64)
        reader = open_bag(args.bag)
        reader.set_filter(rosbag2_py.StorageFilter(topics=[imgtopic]))
        while reader.has_next():
            _, data, ts = reader.read_next()
            msgtype = types[imgtopic]
            cls = CompressedImage if msgtype.endswith("CompressedImage") else Image
            img = decode_image(deserialize_message(data, cls), msgtype)
            if img is None:
                continue
            seq = int(np.abs(stamps - ts).argmin())
            cv2.imwrite("{}{}.jpg".format(imgdir, seq), img,
                        [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            have.add(seq)
            nimg += 1
        print("Wrote {} images to {}".format(nimg, imgdir))

        # The camera is usually slower than the laser, so some scans end up with
        # no frame of their own. anno1602 only looks +-2 seqs away, so hardlink
        # the nearest frame into every remaining gap. Hardlinks cost no disk.
        if have:
            src = np.array(sorted(have))
            nlink = 0
            for seq in range(len(rows)):
                if seq in have:
                    continue
                near = int(src[np.abs(src - seq).argmin()])
                try:
                    os.link("{}{}.jpg".format(imgdir, near),
                            "{}{}.jpg".format(imgdir, seq))
                except OSError:
                    shutil.copyfile("{}{}.jpg".format(imgdir, near),
                                    "{}{}.jpg".format(imgdir, seq))
                nlink += 1
            print("Filled {} gaps by linking the nearest frame".format(nlink))
    else:
        print("No image topic, laser only.")

    fov = math.degrees(proto.angle_max - proto.angle_min)
    print("\nSet these in anno1602.py:")
    print('  basedir      = "{}"'.format(outdir))
    print("  laserFoV     = {:.1f}".format(360.0 if args.no_recentre else fov))
    print("  laser_cutoff = {:.1f}".format(proto.range_max))
    print("  and run:  python anno1602.py {}".format(name))


if __name__ == "__main__":
    main()
