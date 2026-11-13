#!/usr/bin/env python3
"""Generate the serde decode inputs for both sides of the oracle.

The serde group decodes the same JSON texts on both sides: serde_json 1.0.151 in the
Rust driver, src/serde.cyr in the probe. This script is the single list of those
texts. It writes them, hex-encoded so control bytes and invalid UTF-8 survive, to
  scripts/oracle/rust/src/serde_inputs.rs
  scripts/oracle/serde_inputs.cyr
Regenerate both after editing CASES:
  python3 -I scripts/oracle/make_serde_inputs.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# (type, text). Text is a Python str; non-ASCII is written as UTF-8, and a
# bytes value is taken as-is (for invalid UTF-8).
CASES = []


def add(t, *texts):
    for x in texts:
        CASES.append((t, x))


RGBA = '{"r":10,"g":20,"b":30,"a":40}'
# ---- Srgba: u8 fields, deny_unknown_fields ----
add("Srgba",
    RGBA, '{"a":40,"b":30,"g":20,"r":10}', ' \t\n\r{ "r" : 1 , "g":2,"b":3,"a":4 } \n',
    '[1,2,3,4]', '[ 255 , 0 , 128 , 7 ]', '{"\\u0072":1,"g":2,"b":3,"a":4}',
    '{"r":0,"g":0,"b":0,"a":255}', '{"r":255,"g":255,"b":255,"a":255}',
    # u8 range and type
    '{"r":256,"g":0,"b":0,"a":0}', '{"r":-1,"g":0,"b":0,"a":0}', '{"r":-0,"g":0,"b":0,"a":0}',
    '{"r":1.0,"g":0,"b":0,"a":0}', '{"r":1e0,"g":0,"b":0,"a":0}', '{"r":"1","g":0,"b":0,"a":0}',
    '{"r":null,"g":0,"b":0,"a":0}', '{"r":true,"g":0,"b":0,"a":0}', '{"r":[1],"g":0,"b":0,"a":0}',
    '{"r":{},"g":0,"b":0,"a":0}', '{"r":99999999999999999999,"g":0,"b":0,"a":0}',
    '{"r":1e400,"g":0,"b":0,"a":0}', '{"r":01,"g":0,"b":0,"a":0}', '{"r":+1,"g":0,"b":0,"a":0}',
    '{"r":.5,"g":0,"b":0,"a":0}', '{"r":1.,"g":0,"b":0,"a":0}', '{"r":1e,"g":0,"b":0,"a":0}',
    '{"r":-,"g":0,"b":0,"a":0}', '{"r":tru,"g":0,"b":0,"a":0}', '{"r":nul,"g":0,"b":0,"a":0}',
    # fields
    '{"r":1,"g":2,"b":3}', '{"r":1,"g":2,"b":3,"a":4,"x":5}', '{"r":1,"r":2,"g":2,"b":3,"a":4}',
    '{"r":1,"g":2,"b":3,"a":4,"a":5}', '{"x":1,"r":1,"g":2,"b":3,"a":4}', '{}',
    '{"r":1,"r":2', '{"x":1', '{"r":1,"g":2,"b":3,"a":4,}', '{"r":1,"g":2,"b":3,"a":4',
    '{"r":1 "g":2,"b":3,"a":4}', '{"r" 1,"g":2,"b":3,"a":4}', '{r:1,"g":2,"b":3,"a":4}',
    '{,"r":1,"g":2,"b":3,"a":4}', '{"r":1,,"g":2,"b":3,"a":4}',
    # sequence form
    '[1,2,3]', '[1,2,3,4,5]', '[1,2,3,4,]', '[1,2,3,', '[]', '[1 2 3 4]', '[,1,2,3,4]',
    # top level
    '', '   ', 'null', '1', '"r"', 'true', RGBA + 'x', RGBA + ' ', RGBA + ',', RGBA + '{}',
    '[' * 3, '{"r":1,"g":2,"b":3,"a":4}}',
    # strings in keys
    '{"r\\u0000":1,"g":2,"b":3,"a":4}', '{"\\q":1,"g":2,"b":3,"a":4}', '{"\\u00":1}',
    '{"\\ud800":1}', '{"\\udc00":1}', '{"\\ud83d\\ude00":1,"g":2,"b":3,"a":4}',
    '{"r\x01":1}', '{"é":1,"r":1,"g":2,"b":3,"a":4}', b'{"\xff":1}', b'{"\xc0\xaf":1}',
    '{"r":1,"g":2,"b":3,"a":4,"\\"":1}')

FLOATS = ["0", "1", "-1", "0.1", "-0", "-0.0", "1e0", "1E+2", "2.5e-3", "1e-45", "1e-50",
          "3.4028235e38", "1e39", "1e308", "1e400", "-1e400", "1.7976931348623157e308",
          "4.9e-324", "1e-400", "0.30000000000000004", "123456789", "18446744073709551615",
          "18446744073709551616", "-9223372036854775808", "-9223372036854775809",
          "100000000000000000000000000000", "0.000001", "16777217", "33554433"]
# ---- the f32 structs: deny_unknown_fields ----
for t, names in [("LinRgba", "r g b a"), ("Hsl", "h s l"), ("Cmyk", "c m y k"),
                 ("Oklab", "l a b"), ("Oklch", "l c h")]:
    fs = names.split()
    for v in FLOATS:
        add(t, "{" + ",".join('"%s":%s' % (f, v if i == 0 else "0.5") for i, f in enumerate(fs)) + "}")
    add(t, "[" + ",".join(["0.25"] * len(fs)) + "]",
        "{" + ",".join('"%s":0' % f for f in fs) + ',"x":1}',
        "{" + ",".join('"%s":null' % f for f in fs) + "}",
        "{" + ",".join('"%s":"1"' % f for f in fs) + "}",
        "{" + ",".join('"%s":0' % f for f in fs[:-1]) + "}")
# ---- the f64 structs ----
for t, names in [("CieXyz", "x y z"), ("CieLab", "l a b")]:
    fs = names.split()
    for v in FLOATS:
        add(t, "{" + ",".join('"%s":%s' % (f, v if i == 1 else "0.5") for i, f in enumerate(fs)) + "}")
    add(t, "[0.95047,1.0,1.08883]", "{" + ",".join('"%s":0' % f for f in fs) + ',"x":1}',
        "[1,2]", "[1,2,3,4]")
# ---- Affine: no deny_unknown_fields ----
AFF = '{"a":1,"b":0,"c":0,"d":1,"tx":5,"ty":-6}'
add("Affine", AFF, '[1,0,0,1,5,-6]',
    '{"a":1,"b":0,"c":0,"d":1,"tx":5,"ty":-6,"extra":1}',
    '{"zz":[[{"q":[1,2,{"w":null}]}],true,false,"s\\n"],"a":1,"b":0,"c":0,"d":1,"tx":5,"ty":-6}',
    '{"a":1,"b":0,"c":0,"d":1,"tx":5,"ty":-6,"extra":1e400}',
    '{"a":1,"b":0,"c":0,"d":1,"tx":5,"ty":-6,"extra":[1,]}',
    '{"a":1,"b":0,"c":0,"d":1,"tx":5,"ty":-6,"extra":01}',
    '{"a":1,"b":0,"c":0,"d":1,"tx":5,"ty":-6,"extra":{"k" 1}}',
    '{"a":1,"b":0,"c":0,"d":1,"tx":5,"ty":-6,"extra":' + '[' * 300 + ']' * 300 + '}',
    '{"a":1,"b":0,"c":0,"d":1,"tx":5,"ty":-6,"extra":' + '[' * 300 + ']' * 299 + '}',
    '{"a":1,"b":0,"c":0,"d":1,"tx":5}', '{"a":1,"a":1,"b":0,"c":0,"d":1,"tx":5,"ty":-6}',
    '{"a":1e400,"b":0,"c":0,"d":1,"tx":5,"ty":-6}', '{"a":"1","b":0,"c":0,"d":1,"tx":5,"ty":-6}')
# ---- Perspective: [[f64; 3]; 3], no deny ----
add("Perspective", '{"m":[[1,0,0],[0,1,0],[0,0,1]]}', '[[[1,2,3],[4,5,6],[7,8,9]]]',
    '{"m":[[1,0,0],[0,1,0]]}', '{"m":[[1,0,0],[0,1,0],[0,0,1],[0,0,0]]}',
    '{"m":[[1,0],[0,1,0],[0,0,1]]}', '{"m":[[1,0,0,0],[0,1,0],[0,0,1]]}',
    '{"m":[[1,0,0],[0,1,0],[0,0,1]],"x":{"y":[]}}', '{"m":null}', '{"m":[1,2,3]}',
    '{"m":[[1,0,0],[0,1,0],[0,0,"1"]]}', '{}', '{"m":[[1,0,0],[0,1,0],[0,0,1],]}')
# ---- unit enums ----
for t, vs in [("PixelFormat", "Rgba8 Argb8 Rgb8 Yuv420p Nv12 RgbaF32"),
              ("ColorSpace", "Srgb LinearRgb DisplayP3 Bt601 Bt709 Bt2020 CieXyz"),
              ("BlendMode", "Normal Multiply Screen Overlay Darken Lighten ColorDodge ColorBurn "
                            "SoftLight HardLight Difference Exclusion")]:
    v = vs.split()
    for name in v:
        add(t, '"%s"' % name)
    add(t, ' { "%s" : null } ' % v[1], '{"%s":0}' % v[1], '{"%s":null,"x":1}' % v[1],
        '{"%s"}' % v[1], '{"%s":}' % v[1], '"Nope"', '{"Nope":null}', '"%s"' % v[0].lower(),
        '5', '[]', 'null', '{}', '"%s" x' % v[0], '"\\u0%s"' % format(ord(v[0][0]), "03x") + "",
        '"%s' % v[0], '{"%s":nul}' % v[1], '{"%s":[]}' % v[1])
# ---- ToneCurve: externally tagged newtype variants ----
add("ToneCurve", '{"Gamma":2.2}', '{"Gamma":1}', '{"Table":[0,0.5,1]}', '{"Table":[]}',
    ' { "Table" : [ 1 , 2 ] } ', '"Gamma"', '"Table"', '"Nope"', '{"Nope":1}',
    '{"Gamma":"2"}', '{"Gamma":null}', '{"Table":[1,]}', '{"Table":[1,"x"]}', '{"Table":{}}',
    '{"Gamma":1,"Table":[]}', '{"Gamma":1}x', '{"Gamma":1e400}', '{}', '[]', '5',
    '{"Gamma":1', '{"Table":[1,2', '{"Gamma" 1}', '{"Table":null}')


def to_bytes(x):
    return x if isinstance(x, bytes) else x.encode("utf-8")


def main():
    rows = [(t, to_bytes(x).hex()) for t, x in CASES]
    rs = ["// Generated by scripts/oracle/make_serde_inputs.py. Do not edit.",
          "pub const SERDE_INPUTS: &[(&str, &str)] = &["]
    rs += ['    ("%s", "%s"),' % r for r in rows]
    rs += ["];", ""]
    with open(os.path.join(HERE, "rust", "src", "serde_inputs.rs"), "w") as f:
        f.write("\n".join(rs))
    # No branches: one lookup function with an `if` per input would pass the
    # compiler's 1,023-jump-target limit. serde_inputs_init fills two tables.
    cy = ["# Generated by scripts/oracle/make_serde_inputs.py. Do not edit.",
          "# The serde decode inputs, hex-encoded: (type, text) pairs, in order.",
          "var SERDE_N = %d;" % len(rows),
          "var _SERDE_T = 0;",
          "var _SERDE_H = 0;",
          "fn serde_inputs_init() {",
          "    _SERDE_T = fl_alloc(SERDE_N * 8);",
          "    _SERDE_H = fl_alloc(SERDE_N * 8);"]
    for i, (t, h) in enumerate(rows):
        cy.append('    store64(_SERDE_T + %d, "%s"); store64(_SERDE_H + %d, "%s");' % (i * 8, t, i * 8, h))
    cy += ["    return 0;", "}",
           "fn serde_input(i, which) {",
           "    if (_SERDE_T == 0) { serde_inputs_init(); }",
           "    if (which == 0) { return load64(_SERDE_T + i * 8); }",
           "    return load64(_SERDE_H + i * 8);",
           "}", ""]
    with open(os.path.join(HERE, "serde_inputs.cyr"), "w") as f:
        f.write("\n".join(cy))
    print("%d serde inputs" % len(rows))


if __name__ == "__main__":
    main()
