"""Minimal ROS1 .bag (v2.0) reader for CED: sensor_msgs/Image and dvs_msgs/EventArray only. Stdlib + numpy.
shortcut: supports chunk compression none/bz2 only; add lz4 if a bag needs it.
"""
import bz2, struct
import numpy as np

EV_DT = np.dtype([("x", "<u2"), ("y", "<u2"), ("sec", "<u4"), ("nsec", "<u4"), ("p", "u1")])  # 13 B, packed


def _fields(h):
    out, i = {}, 0
    while i < len(h):
        (n,) = struct.unpack_from("<I", h, i)
        k, v = h[i + 4:i + 4 + n].split(b"=", 1)
        out[k.decode()] = v
        i += 4 + n
    return out


def _records(buf, i=0):
    while i + 8 <= len(buf):
        (hl,) = struct.unpack_from("<I", buf, i)
        h = _fields(buf[i + 4:i + 4 + hl]); i += 4 + hl
        (dl,) = struct.unpack_from("<I", buf, i)
        yield h, buf[i + 4:i + 4 + dl]; i += 4 + dl


def _string(d, i):
    (n,) = struct.unpack_from("<I", d, i)
    return d[i + 4:i + 4 + n].decode(), i + 4 + n


def _after_header(d):  # std_msgs/Header: seq, stamp (sec, nsec), frame_id
    seq, s, ns = struct.unpack_from("<III", d, 0)
    _, i = _string(d, 12)
    return s + ns * 1e-9, i


def read_bag(path, topics):
    """-> {topic: [raw message bytes, ...]} in file order, plus {topic: msg type}."""
    with open(path, "rb") as f:
        buf = f.read()
    assert buf[:13] == b"#ROSBAG V2.0\n", "not a ROS1 v2.0 bag"
    conns, types, out = {}, {}, {t: [] for t in topics}

    def handle(h, d):
        op = h["op"][0]
        if op == 7:  # connection
            topic = h["topic"].decode()
            conns[struct.unpack("<I", h["conn"])[0]] = topic
            types[topic] = _fields(d).get("type", b"").decode()
        elif op == 2:  # message
            t = conns.get(struct.unpack("<I", h["conn"])[0])
            if t in out:
                out[t].append(d)

    for h, d in _records(buf, 13):
        op = h["op"][0]
        if op == 5:  # chunk
            comp = h["compression"].decode()
            if comp == "bz2":
                d = bz2.decompress(d)
            elif comp != "none":
                raise NotImplementedError(f"chunk compression {comp}")
            for h2, d2 in _records(d):
                handle(h2, d2)
        else:
            handle(h, d)
    return out, types


def parse_image(d):
    """sensor_msgs/Image -> (t_sec, encoding, (H,W) uint8 array)."""
    t, i = _after_header(d)
    hgt, wid = struct.unpack_from("<II", d, i); i += 8
    enc, i = _string(d, i)
    i += 1  # is_bigendian
    (step,) = struct.unpack_from("<I", d, i); i += 4
    (n,) = struct.unpack_from("<I", d, i); i += 4
    assert n == step * hgt and step == wid, (enc, step, wid)  # 8-bit single channel only
    return t, enc, np.frombuffer(d, np.uint8, n, i).reshape(hgt, wid)


def parse_events(d):
    """dvs_msgs/EventArray -> structured array EV_DT."""
    _, i = _after_header(d)
    i += 8  # height, width
    (n,) = struct.unpack_from("<I", d, i); i += 4
    assert len(d) - i == n * EV_DT.itemsize, "unexpected Event layout"
    return np.frombuffer(d, EV_DT, n, i)


if __name__ == "__main__":
    import sys
    msgs, types = read_bag(sys.argv[1], ["/dvs/image_raw", "/dvs/events"])
    print(types)
    t, enc, im = parse_image(msgs["/dvs/image_raw"][0])
    ev = np.concatenate([parse_events(m) for m in msgs["/dvs/events"]])
    print(len(msgs["/dvs/image_raw"]), "images", enc, im.shape, im.min(), im.max(), "| events", len(ev), ev[:2])
