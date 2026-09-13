"""Focused generator checks; run python3 -m unittest discover -s scripts."""
import unittest

import generate_five_point_wgsl as generator


class GeneratorTests(unittest.TestCase):
    def test_artifact_is_reproducible(self):
        self.assertEqual(generator.generate(), generator.OUTPUT.read_text())
        self.assertEqual(generator.generate(), generator.generate())

    def test_arithmetic_only_and_bounds(self):
        for text in ["b[39]", "b[-1]", "b[1.0]", "c[0]", "b[0] / b[1]",
                     "abs(b[0])", "b[0] ** 2", "b[0] if True else b[1]"]:
            with self.subTest(text=text), self.assertRaises((AssertionError, ValueError)):
                generator.expression(text, {"b": 39})

    def test_missing_duplicate_and_unknown_statements_fail_closed(self):
        source = generator.SOURCE.read_text()
        for changed in [source.replace("a[190] =", "a[191] =", 1),
                        source.replace("a[190] =", "a[200] =", 1),
                        source.replace("    a\n}", "    unknown();\n    a\n}", 1),
                        source.replace("e3[i] = e2[i] * e[i]", "e3[i] = e[i] * e[i]", 1)]:
            with self.assertRaises((AssertionError, ValueError)):
                generator.translate(changed, "build_elimination_matrix", "e", 36, "a", 200, True)
        with self.assertRaises((AssertionError, ValueError)):
            generator.polynomial_table(source.replace("coeffs[10] =", "coeffs[9] =", 1))
        with self.assertRaises((AssertionError, ValueError)):
            generator.polynomial_table(source.replace("    coeffs\n}", "    unknown();\n    coeffs\n}", 1))

    def test_term_order_sign_and_packing(self):
        # Minimal full function: 11 outputs to exercise the same strict parser.
        source = "pub(crate) fn determinant_coeffs(b: &[f64; 39]) -> [f64; 11] {\n"
        source += "let mut coeffs = [0.0f64; 11];\n"
        source += "\n".join(f"coeffs[{i}] = -b[0] * b[1] * b[2] + b[3] * b[4] * b[5] - b[6] * b[7] * b[8];" for i in range(11))
        source += "\ncoeffs\n}"
        result = generator.polynomial_table(source)
        words = [0 | 1 << 6 | 2 << 12 | 1 << 19,
                 3 | 4 << 6 | 5 << 12,
                 6 | 7 << 6 | 8 << 12 | 1 << 18]
        self.assertIn(", ".join(f"{word}u" for word in words), result)
        self.assertIn("array<u32, 33>", result)
        self.assertIn("0u, 3u, 6u, 9u, 12u, 15u, 18u, 21u, 24u, 27u, 30u, 33u", result)


if __name__ == "__main__":
    unittest.main()
