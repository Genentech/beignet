import beignet.polynomial
import numpy
import numpy.testing
import pytest


class TestInit:
    """
    Test polynomial creation with symbol kwarg.
    """

    c = [1, 2, 3]

    def test_default_symbol(self):
        p = beignet.polynomial.Polynomial(self.c)
        numpy.testing.assert_equal(p.symbol, "x")

    @pytest.mark.parametrize(
        ("bad_input", "exception"),
        (
            ("", ValueError),
            ("3", ValueError),
            (None, TypeError),
            (1, TypeError),
        ),
    )
    def test_symbol_bad_input(self, bad_input, exception):
        with pytest.raises(exception):
            beignet.polynomial.Polynomial(self.c, symbol=bad_input)

    @pytest.mark.parametrize(
        "symbol",
        (
            "x",
            "x_1",
            "A",
            "xyz",
            "β",
        ),
    )
    def test_valid_symbols(self, symbol):
        """
        Values for symbol that should pass input validation.
        """
        p = beignet.polynomial.Polynomial(self.c, symbol=symbol)
        numpy.testing.assert_equal(p.symbol, symbol)

    def test_property(self):
        """
        'symbol' attribute is read only.
        """
        p = beignet.polynomial.Polynomial(self.c, symbol="x")
        with pytest.raises(AttributeError):
            p.symbol = "z"

    def test_change_symbol(self):
        p = beignet.polynomial.Polynomial(self.c, symbol="y")
        # Create new polynomial from p with different symbol
        pt = beignet.polynomial.Polynomial(p.coef, symbol="t")
        numpy.testing.assert_equal(pt.symbol, "t")


class TestUnaryOperators:
    p = beignet.polynomial.Polynomial([1, 2, 3], symbol="z")

    def test_neg(self):
        n = -self.p
        numpy.testing.assert_equal(n.symbol, "z")

    def test_scalarmul(self):
        out = self.p * 10
        numpy.testing.assert_equal(out.symbol, "z")

    def test_rscalarmul(self):
        out = 10 * self.p
        numpy.testing.assert_equal(out.symbol, "z")

    def test_pow(self):
        out = self.p**3
        numpy.testing.assert_equal(out.symbol, "z")


@pytest.mark.parametrize(
    "rhs",
    (
        beignet.polynomial.Polynomial([4, 5, 6], symbol="z"),
        numpy.array([4, 5, 6]),
    ),
)
class TestBinaryOperatorsSameSymbol:
    """
    Ensure symbol is preserved for numeric operations on polynomials with
    the same symbol
    """

    p = beignet.polynomial.Polynomial([1, 2, 3], symbol="z")

    def test_add(self, rhs):
        out = self.p + rhs
        numpy.testing.assert_equal(out.symbol, "z")

    def test_sub(self, rhs):
        out = self.p - rhs
        numpy.testing.assert_equal(out.symbol, "z")

    def test_polymul(self, rhs):
        out = self.p * rhs
        numpy.testing.assert_equal(out.symbol, "z")

    def test_divmod(self, rhs):
        for out in divmod(self.p, rhs):
            numpy.testing.assert_equal(out.symbol, "z")

    def test_radd(self, rhs):
        out = rhs + self.p
        numpy.testing.assert_equal(out.symbol, "z")

    def test_rsub(self, rhs):
        out = rhs - self.p
        numpy.testing.assert_equal(out.symbol, "z")

    def test_rmul(self, rhs):
        out = rhs * self.p
        numpy.testing.assert_equal(out.symbol, "z")

    def test_rdivmod(self, rhs):
        for out in divmod(rhs, self.p):
            numpy.testing.assert_equal(out.symbol, "z")


class TestBinaryOperatorsDifferentSymbol:
    p = beignet.polynomial.Polynomial([1, 2, 3], symbol="x")
    other = beignet.polynomial.Polynomial([4, 5, 6], symbol="y")
    ops = (p.__add__, p.__sub__, p.__mul__, p.__floordiv__, p.__mod__)

    @pytest.mark.parametrize("f", ops)
    def test_binops_fails(self, f):
        numpy.testing.assert_raises(ValueError, f, self.other)


class TestEquality:
    p = beignet.polynomial.Polynomial([1, 2, 3], symbol="x")

    def test_eq(self):
        other = beignet.polynomial.Polynomial([1, 2, 3], symbol="x")
        numpy.testing.assert_(self.p == other)

    def test_neq(self):
        other = beignet.polynomial.Polynomial([1, 2, 3], symbol="y")
        numpy.testing.assert_(not self.p == other)


class TestExtraMethods:
    """
    Test other methods for manipulating/creating polynomial objects.
    """

    p = beignet.polynomial.Polynomial([1, 2, 3, 0], symbol="z")

    def test_copy(self):
        other = self.p.copy()
        numpy.testing.assert_equal(other.symbol, "z")

    def test_trim(self):
        other = self.p.trim()
        numpy.testing.assert_equal(other.symbol, "z")

    def test_truncate(self):
        other = self.p.truncate(2)
        numpy.testing.assert_equal(other.symbol, "z")

    @pytest.mark.parametrize(
        "kwarg",
        (
            {"domain": [-10, 10]},
            {"window": [-10, 10]},
            {"kind": beignet.polynomial.ChebyshevPolynomial},
        ),
    )
    def test_convert(self, kwarg):
        other = self.p.convert(**kwarg)
        numpy.testing.assert_equal(other.symbol, "z")

    def test_integ(self):
        other = self.p.integ()
        numpy.testing.assert_equal(other.symbol, "z")

    def test_deriv(self):
        other = self.p.deriv()
        numpy.testing.assert_equal(other.symbol, "z")


def test_composition():
    p = beignet.polynomial.Polynomial([3, 2, 1], symbol="t")
    q = beignet.polynomial.Polynomial([5, 1, 0, -1], symbol="λ_1")
    r = p(q)
    assert r.symbol == "λ_1"


#
# Class methods that result in new polynomial class instances
#


def test_fit():
    x, y = (range(10),) * 2
    p = beignet.polynomial.Polynomial.fit(x, y, deg=1, symbol="z")
    numpy.testing.assert_equal(p.symbol, "z")


def test_froomroots():
    roots = [-2, 2]
    p = beignet.polynomial.Polynomial.fromroots(roots, symbol="z")
    numpy.testing.assert_equal(p.symbol, "z")


def test_identity():
    p = beignet.polynomial.Polynomial.identity(
        domain=[-1, 1], window=[5, 20], symbol="z"
    )
    numpy.testing.assert_equal(p.symbol, "z")


def test_basis():
    p = beignet.polynomial.Polynomial.basis(3, symbol="z")
    numpy.testing.assert_equal(p.symbol, "z")
