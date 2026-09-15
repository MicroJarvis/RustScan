#!/usr/bin/env python3
"""Translate the checked-in COLMAP Rust arithmetic subset, never eval Rust.

Run from any directory; --check verifies the committed artifact. No dependencies.
Array indices and polynomial term order are preserved; polynomial rounding is
compensated in f32. Elimination evaluation order is unchanged. Unsupported syntax,
missing/duplicate assignments or out-of-bounds operands fail closed.
"""
import argparse
import ast
import hashlib
from pathlib import Path
import re

if not __debug__:
    raise RuntimeError("run without -O: generator validation assertions are required")

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "RustSFM/src/geometry/five_point_generated.rs"
OUTPUT = ROOT / "RustSFM/src/gpu/shaders/five_point_generated.wgsl"


def expression(text, arrays):
    tree = ast.parse(" ".join(text.split()), mode="eval")

    def validate(node):
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            validate(node.left)
            validate(node.right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            validate(node.operand)
        elif isinstance(node, ast.Constant) and type(node.value) in (int, float):
            pass
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            assert node.value.id in arrays, node.value.id
            assert isinstance(node.slice, ast.Constant) and type(node.slice.value) is int
            assert 0 <= node.slice.value < arrays[node.value.id]
        else:
            raise ValueError(f"unsupported expression: {ast.dump(node)}")

    validate(tree.body)
    return " ".join(text.split())


def translate(source, name, arg, size, result, count, powers=False):
    prefix = f"pub(crate) fn {name}({arg}: &[f64; {size}]) -> [f64; {count}] {{"
    start = source.index(prefix) + len(prefix)
    end = source.index("\n}", start)
    body = source[start:end].strip()
    init = f"let mut {result} = [0.0f64; {count}];"
    assert body.startswith(init)
    body = body[len(init):].strip()
    if powers:
        lines = ["@compute @workgroup_size(1,32,1)",
                 "fn elimination(@builtin(global_invocation_id) id: vec3<u32>) {",
                 "  if(id.y>=200u) { return; }",
                 "  var e: array<f32,36>;",
                 "  for(var i=0u;i<36u;i++) { e[i]=bases[id.x*36u+i]; }"]
    else:
        lines = ["@compute @workgroup_size(1)",
                 "fn polynomial(@builtin(workgroup_id) id: vec3<u32>) {",
                 "  if(algebra_output[id.x*352u+350u]!=0.0) { return; }",
                 "  var b: array<f32,39>;",
                 "  for(var i=0u;i<39u;i++) { b[i]=algebra_output[id.x*352u+300u+i]; }"]
    arrays = {arg: size}
    if powers:
        prelude = """let mut e2 = [0.0f64; 36];
    let mut e3 = [0.0f64; 36];
    for i in 0..36 {
        e2[i] = e[i] * e[i];
        e3[i] = e2[i] * e[i];
    }"""
        assert body.startswith(prelude), "Rust power prelude changed"
        body = body[len(prelude):].strip()
        arrays.update(e2=36, e3=36)
        lines += ["  var e2: array<f32, 36>; var e3: array<f32, 36>;",
                  "  for (var i = 0u; i < 36u; i++) { e2[i] = e[i] * e[i]; e3[i] = e2[i] * e[i]; }"]
    assert body.endswith(result)
    body = body[:-len(result)].strip()
    lines.append("  switch id.y {")
    seen = set()
    for statement in body.split(";"):
        if not statement.strip():
            continue
        match = re.fullmatch(r"\s*" + result + r"\[(\d+)\]\s*=\s*(.+)", statement, re.S)
        assert match, statement
        index = int(match[1])
        assert index not in seen and 0 <= index < count
        seen.add(index)
        expr = expression(match[2], arrays)
        if powers:
            lines.append(f"    case {index}u: {{ algebra_output[id.x*352u+{index}u] = {expr}; }}")
        else:
            lines.append(f"    case {index}u: {{ algebra_output[id.x*352u+{339+index}u] = {expr}; }}")
    assert seen == set(range(count)), "incomplete output assignments"
    lines += ["    default: {}", "  }", "}"]
    return "\n".join(lines)


def polynomial_table(source):
    """Lower left-associated signed triple products to a compact WGSL term table.

    Keeps term order/signs; compensates rounded triple products and accumulation.
    Actual GPU regression tests are required: WGSL/Metal may optimize error terms.
    """
    # Validate the complete function body, including its prelude/return and every
    # assignment, before lowering the more restrictive triple-product subset.
    translate(source, "determinant_coeffs", "b", 39, "coeffs", 11)
    body = source[source.index("pub(crate) fn determinant_coeffs"):]
    assignments = re.findall(r"coeffs\[(\d+)\]\s*=\s*(.*?);", body, re.S)
    assert [int(i) for i, _ in assignments] == list(range(11))
    words, offsets = [], [0]

    def terms(node):
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            return terms(node.left) + [(node.right, isinstance(node.op, ast.Sub))]
        return [(node, False)]

    for _, text in assignments:
        expression(text, {"b": 39})
        ordered_terms = terms(ast.parse(" ".join(text.split()), mode="eval").body)
        assert not ordered_terms[0][1]
        for node, subtract in ordered_terms:
            assert isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)
            assert isinstance(node.left, ast.BinOp) and isinstance(node.left.op, ast.Mult)
            factors = [node.left.left, node.left.right, node.right]
            negate_first = isinstance(factors[0], ast.UnaryOp)
            if negate_first:
                assert isinstance(factors[0].op, ast.USub)
                factors[0] = factors[0].operand
            indices = []
            for factor in factors:
                assert isinstance(factor, ast.Subscript) and factor.value.id == "b"
                indices.append(factor.slice.value)
            words.append(indices[0] | indices[1] << 6 | indices[2] << 12
                         | int(subtract) << 18 | int(negate_first) << 19)
        offsets.append(len(words))
    return (f"const polynomial_terms = array<u32, {len(words)}>(\n  "
            + ",\n  ".join(", ".join(f"{v}u" for v in words[i:i+12]) for i in range(0,len(words),12))
            + ");\nconst polynomial_offsets = array<u32, 12>("
            + ", ".join(f"{v}u" for v in offsets) + ");\n"
            + """
@compute @workgroup_size(1)
fn polynomial(@builtin(workgroup_id) id: vec3<u32>) {
  let ob=id.x*352u;
  if(algebra_output[ob+350u]!=0.0) { return; }
  var value=0.0;
  var correction=0.0;
  let start=polynomial_offsets[id.y];
  let end=polynomial_offsets[id.y+1u];
  for(var i=start;i<end;i++) {
    let word=polynomial_terms[i];
    var first=algebra_output[ob+300u+(word&63u)];
    if((word&524288u)!=0u) { first=-first; }
    let second=algebra_output[ob+300u+((word>>6u)&63u)];
    let third=algebra_output[ob+300u+((word>>12u)&63u)];
    if((word&262144u)!=0u) { first=-first; }
    let pair=first*second;
    let pair_error=fma(first,second,-pair);
    let product=pair*third;
    let product_error=fma(pair,third,-product)+pair_error*third;
    let next=value+product;
    var sum_error=0.0;
    // Explicit fma subtraction prevents the measured Metal reassociation of
    // (value-next)+product to zero. This is device-tested, not a WGSL guarantee.
    if(abs(value)>=abs(product)) { sum_error=fma(-1.0,next,value)+product; }
    else { sum_error=fma(-1.0,next,product)+value; }
    correction+=sum_error+product_error;
    value=next;
  }
  algebra_output[ob+339u+id.y]=value+correction;
}
""")


def generate():
    source = SOURCE.read_text()
    header = source[:source.index("#![")]
    return (header + "// Mechanically generated by scripts/generate_five_point_wgsl.py; do not edit.\n"
            + f"// Source SHA256: {hashlib.sha256(source.encode()).hexdigest()}\n"
            + "// e: column-major 9x4; a: column-major 10x20; b: column-major 13x3.\n"
            + "// coeffs[0..11]: descending z^10 through z^0, unnormalized.\n"
            + "// Dispatch: elimination (samples, ceil(200/32), 1), global row id; polynomial (samples, 11, 1).\n\n"
            + translate(source, "build_elimination_matrix", "e", 36, "a", 200, True)
            + "\n\n" + polynomial_table(source) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = generate()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != text:
            raise SystemExit("generated WGSL is stale; run scripts/generate_five_point_wgsl.py")
        print("five-point WGSL reproducibility check passed (200 + 11 expressions)")
    else:
        OUTPUT.write_text(text)
        print(f"generated {OUTPUT.relative_to(ROOT)}")
