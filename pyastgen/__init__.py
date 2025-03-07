import ast
from collections import ChainMap, defaultdict
from collections.abc import Callable, Iterable, Iterator
from functools import partialmethod
from itertools import count
from typing import Protocol, overload


# FIXME: Remove once mypy has support for partialmethod
# https://github.com/python/mypy/issues/8619
class _BinOp(Protocol):
    def __get__(self, obj: "Value", cls=None) -> Callable[["Value"], "Value"]: ...


class _UnaryOp(Protocol):
    def __get__(self, obj: "Value", cls=None) -> Callable[[], "Value"]: ...


class _AugAssign(Protocol):
    def __get__(
        self, obj: "Builder", cls=None
    ) -> Callable[["Target", "Value"], None]: ...


class Slice:
    __slots__ = ("lower", "upper", "step")

    def __init__(
        self,
        lower: "Value | None" = None,
        upper: "Value | None" = None,
        step: "Value | None" = None,
    ) -> None:
        self.lower = lower
        self.upper = upper
        self.step = step


class Value:
    __slots__ = ("expr",)

    def __init__(self, expr: ast.expr) -> None:
        self.expr = expr

    def attribute(self, name: str) -> "Target":
        return Target(ast.Attribute(self.expr, name))

    def call(self, *args: "Value | Starred", **kwargs: "Value") -> "Value":
        return self.apply(args, kwargs.items())

    def apply(
        self,
        args: Iterable["Value | Starred"],
        kwargs: Iterable[tuple[str | None, "Value"]],
    ) -> "Value":
        return Value(
            ast.Call(
                self.expr,
                [v.expr for v in args],
                [ast.keyword(k, v.expr) for k, v in kwargs],
            )
        )

    def _unaryop(self, op: ast.unaryop) -> "Value":
        return Value(ast.UnaryOp(op, self.expr))

    def _binop(self, op: ast.operator, other: "Value") -> "Value":
        return Value(ast.BinOp(self.expr, op, other.expr))

    def _compare(self, op: ast.cmpop, other: "Value") -> "Value":
        return Value(ast.Compare(self.expr, [op], [other.expr]))

    def _boolop(self, op: ast.boolop, other: "Value") -> "Value":
        return Value(ast.BoolOp(op, [self.expr, other.expr]))

    # arithmetic
    add: _BinOp = partialmethod(_binop, ast.Add())
    sub: _BinOp = partialmethod(_binop, ast.Sub())
    mul: _BinOp = partialmethod(_binop, ast.Mult())
    div: _BinOp = partialmethod(_binop, ast.Div())
    floordiv: _BinOp = partialmethod(_binop, ast.FloorDiv())
    pow: _BinOp = partialmethod(_binop, ast.Pow())
    mod: _BinOp = partialmethod(_binop, ast.Mod())
    matmul: _BinOp = partialmethod(_binop, ast.MatMult())
    neg: _UnaryOp = partialmethod(_unaryop, ast.USub())
    pos: _UnaryOp = partialmethod(_unaryop, ast.UAdd())

    # bitwise
    bit_and: _BinOp = partialmethod(_binop, ast.BitAnd())
    bit_or: _BinOp = partialmethod(_binop, ast.BitOr())
    bit_xor: _BinOp = partialmethod(_binop, ast.BitXor())
    invert: _UnaryOp = partialmethod(_unaryop, ast.Invert())
    lshift: _BinOp = partialmethod(_binop, ast.LShift())
    rshift: _BinOp = partialmethod(_binop, ast.RShift())

    # logical
    and_: _BinOp = partialmethod(_boolop, ast.And())
    or_: _BinOp = partialmethod(_boolop, ast.Or())
    not_: _UnaryOp = partialmethod(_unaryop, ast.Not())

    # comparison
    contains: _BinOp = partialmethod(_compare, ast.In())
    is_: _BinOp = partialmethod(_compare, ast.Is())
    is_not: _BinOp = partialmethod(_compare, ast.IsNot())
    lt: _BinOp = partialmethod(_compare, ast.Lt())
    lte: _BinOp = partialmethod(_compare, ast.LtE())
    gt: _BinOp = partialmethod(_compare, ast.Gt())
    gte: _BinOp = partialmethod(_compare, ast.GtE())
    eq: _BinOp = partialmethod(_compare, ast.Eq())
    ne: _BinOp = partialmethod(_compare, ast.NotEq())

    def subscript(self, key: "Value | Slice") -> "Target":
        if isinstance(key, Slice):
            key_expr: ast.expr | ast.Slice = ast.Slice(
                key.lower.expr if key.lower else None,
                key.upper.expr if key.upper else None,
                key.step.expr if key.step else None,
            )
        else:
            key_expr = key.expr
        return Target(ast.Subscript(self.expr, key_expr))


class Starred:
    __slots__ = ("expr",)

    def __init__(self, value: Value) -> None:
        self.expr = ast.Starred(value.expr)


def constant(value) -> Value:
    return Value(ast.Constant(value))


def list_(*elts: Value | Starred) -> Value:
    return Value(ast.List([el.expr for el in elts]))


def tuple_(*elts: Value | Starred) -> Value:
    return Value(ast.Tuple([el.expr for el in elts]))


def set_(*elts: Value | Starred) -> Value:
    return Value(ast.Set([el.expr for el in elts]))


def dict_(*items: tuple[Value | None, Value]) -> Value:
    if not items:
        return Value(ast.Dict())
    keys, values = zip(*items, strict=True)
    return Value(
        ast.Dict([k.expr if k else None for k in keys], [v.expr for v in values])
    )


def yield_(value: Value | None = None) -> Value:
    return Value(ast.Yield(value.expr if value else None))


def yield_from(value: Value) -> Value:
    return Value(ast.YieldFrom(value.expr))


def await_(value: Value) -> Value:
    return Value(ast.Await(value.expr))


class InvalidComprehension(Exception):
    pass


class Comprehension:
    __slots__ = ("_comprehensions", "_scope")

    def __init__(self, parent_scope: "Scope") -> None:
        self._comprehensions: list[ast.comprehension] = []
        self._scope = parent_scope.new_child()

    # FIXME: Support more complex targets
    def for_(self, target: str, iter_: Value, *, is_async: bool = False) -> "Variable":
        var = self._scope.declare(target)
        self._comprehensions.append(
            ast.comprehension(var.expr, iter_.expr, is_async=is_async)
        )
        return var

    def if_(self, cond: Value) -> None:
        if not self._comprehensions:
            raise InvalidComprehension("Comprehension must start with for")
        self._comprehensions[-1].ifs.append(cond.expr)

    def list(self, value: Value) -> Value:
        return Value(ast.ListComp(value.expr, self._comprehensions[:]))

    def set(self, value: Value) -> Value:
        return Value(ast.SetComp(value.expr, self._comprehensions[:]))

    def generator(self, value: Value) -> Value:
        return Value(ast.GeneratorExp(value.expr, self._comprehensions[:]))

    def dict(self, key: Value, value: Value) -> Value:
        return Value(ast.DictComp(key.expr, value.expr, self._comprehensions[:]))


class Target(Value):
    __slots__ = ()

    def store(self, builder: "Builder", value: Value) -> None:
        builder.assign([self], value)


class Variable(Value):
    __slots__ = "name", "target"

    def __init__(self, name) -> None:
        super().__init__(ast.Name(name, ctx=ast.Load()))
        self.name = name
        self.target = Target(ast.Name(name, ctx=ast.Store()))

    def store(self, builder: "Builder", value: Value) -> None:
        self.target.store(builder, value)


class Parameters:
    __slots__ = (
        "posonlyargs",
        "args",
        "vararg",
        "kwonlyargs",
        "kw_defaults",
        "kwarg",
        "defaults",
    )

    def __init__(
        self,
        posonlyargs: list[str] | None = None,
        args: list[str] | None = None,
        vararg: str | None = None,
        kwonlyargs: list[str] | None = None,
        kw_defaults: dict[str, Value] | None = None,
        kwarg: str | None = None,
        defaults: list[Value] | None = None,
    ) -> None:
        self.posonlyargs = posonlyargs or []
        self.args = args or []
        self.vararg = vararg
        self.kwonlyargs = kwonlyargs or []
        self.kw_defaults = kw_defaults or {}
        self.kwarg = kwarg
        self.defaults = defaults or []

    def _as_arguments(self) -> ast.arguments:
        kw_defaults = []
        for name in self.kwonlyargs:
            val = self.kw_defaults.get(name)
            if val is not None:
                expr = val.expr
            else:
                expr = None
            kw_defaults.append(expr)

        return ast.arguments(
            posonlyargs=[ast.arg(name) for name in self.posonlyargs],
            args=[ast.arg(name) for name in self.args],
            vararg=ast.arg(self.vararg) if self.vararg is not None else None,
            kwonlyargs=[ast.arg(name) for name in self.kwonlyargs],
            kw_defaults=kw_defaults,
            kwarg=ast.arg(self.kwarg) if self.kwarg is not None else None,
            defaults=[val.expr for val in self.defaults],
        )

    def _all(self) -> list[str]:
        all = [
            *self.posonlyargs,
            *self.args,
        ]
        if self.vararg is not None:
            all.append(self.vararg)
        all.extend(self.kwonlyargs)
        if self.kwarg is not None:
            all.append(self.kwarg)
        return all


class NameCollision(Exception):
    pass


class Scope:
    __slots__ = "_mapping", "_name_counter"

    def __init__(self, parent: "Scope | None" = None):
        if parent is None:
            self._mapping = ChainMap[str, Variable]()
        else:
            self._mapping = parent._mapping.new_child()
        self._name_counter = defaultdict[str, Iterator[int]](count)

    def declare(self, name: str, *, exact: bool = False) -> Variable:
        c = next(self._name_counter[name])
        if c != 0:
            if exact:
                raise NameCollision(name)
            name = f"{name}{c}"
        var = self._mapping[name] = Variable(name)
        return var

    def get(self, name: str) -> Variable:
        return self._mapping[name]

    def new_child(self) -> "Scope":
        return Scope(self)


class Builder:
    __slots__ = "scope", "block"

    def __init__(self, block: list[ast.stmt], scope: Scope | None = None) -> None:
        self.scope = scope or Scope()
        self.block = block

    def declare(
        self, name: str, value: Value | None = None, *, exact: bool = False
    ) -> Variable:
        var = self.scope.declare(name, exact=exact)
        if value is not None:
            var.store(self, value)
        return var

    def comprehension(self) -> Comprehension:
        return Comprehension(self.scope)

    def assign(self, targets: list[Target], value: Value) -> None:
        self.block.append(ast.Assign([t.expr for t in targets], value.expr))

    def _aug_assign(self, op: ast.operator, target: Target, value: Value) -> None:
        assert isinstance(target.expr, ast.Name | ast.Attribute | ast.Subscript)
        self.block.append(ast.AugAssign(target.expr, op, value.expr))

    add_assign: _AugAssign = partialmethod(_aug_assign, ast.Add())
    sub_assign: _AugAssign = partialmethod(_aug_assign, ast.Sub())
    mul_assign: _AugAssign = partialmethod(_aug_assign, ast.Mult())
    matmul_assign: _AugAssign = partialmethod(_aug_assign, ast.MatMult())
    div_assign: _AugAssign = partialmethod(_aug_assign, ast.Div())
    floordiv_assign: _AugAssign = partialmethod(_aug_assign, ast.FloorDiv())
    mod_assign: _AugAssign = partialmethod(_aug_assign, ast.Mod())
    pow_assign: _AugAssign = partialmethod(_aug_assign, ast.Pow())
    lshift_assign: _AugAssign = partialmethod(_aug_assign, ast.LShift())
    rshift_assign: _AugAssign = partialmethod(_aug_assign, ast.RShift())
    bit_or_assign: _AugAssign = partialmethod(_aug_assign, ast.BitOr())
    bit_and_assign: _AugAssign = partialmethod(_aug_assign, ast.BitAnd())
    bit_xor_assign: _AugAssign = partialmethod(_aug_assign, ast.BitXor())

    @staticmethod
    def _alias(name: str, alias: str) -> ast.alias:
        if alias == name:
            return ast.alias(name)
        return ast.alias(name, alias)

    def import_(self, *paths: str | tuple[str, str]) -> list[Variable]:
        variables = []
        aliases = []
        for path in paths:
            match path:
                case str():
                    var = self.scope.declare(path)
                    alias = self._alias(path, var.name)
                case (path, name):
                    var = self.scope.declare(name)
                    alias = self._alias(path, var.name)
            aliases.append(alias)
            variables.append(var)
        self.block.append(ast.Import(aliases))
        return variables

    def import1(self, path: str | tuple[str, str]) -> Variable:
        return self.import_(path)[0]

    def relative_import(
        self, path: str, *imports: str | tuple[str, str]
    ) -> list[Variable]:
        variables = []
        aliases = []
        for import_ in imports:
            match import_:
                case str():
                    var = self.scope.declare(import_)
                    alias = self._alias(import_, var.name)
                case (export_name, name):
                    var = self.scope.declare(name)
                    alias = self._alias(export_name, var.name)
            aliases.append(alias)
            variables.append(var)
        level = 0
        for c in path:
            if c != ".":
                break
            level += 1
        self.block.append(ast.ImportFrom(path[level:], aliases, level))
        return variables

    def relative_import1(self, path, import_: str | tuple[str, str]) -> Variable:
        return self.relative_import(path, import_)[0]

    def if_(self, condition: Value) -> tuple["Builder", "Builder"]:
        stmt = ast.If(test=condition.expr, body=[], orelse=[])
        self.block.append(stmt)
        return Builder(stmt.body, self.scope), Builder(stmt.orelse, self.scope)

    def while_(self, condition: Value) -> tuple["Builder", "Builder"]:
        stmt = ast.While(test=condition.expr, body=[], orelse=[])
        self.block.append(stmt)
        return Builder(stmt.body, self.scope), Builder(stmt.orelse, self.scope)

    def for_(
        self, target: Value, iter: Value, *, is_async: bool = False
    ) -> tuple["Builder", "Builder"]:
        constructor = ast.AsyncFor if is_async else ast.For
        stmt = constructor(target=target.expr, iter=iter.expr, body=[], orelse=[])
        self.block.append(stmt)
        return Builder(stmt.body, self.scope), Builder(stmt.orelse, self.scope)

    def pass_(self) -> None:
        self.block.append(ast.Pass())

    def return_(self, value: Value) -> None:
        self.block.append(ast.Return(value.expr))

    def expr(self, value: Value) -> None:
        self.block.append(ast.Expr(value.expr))

    def raise_(self, value: Value) -> None:
        self.block.append(ast.Raise(value.expr))

    def assert_(self, value: Value) -> None:
        self.block.append(ast.Assert(value.expr))

    def delete(self, *targets: Target) -> None:
        self.block.append(ast.Delete([t.expr for t in targets]))

    def break_(self) -> None:
        self.block.append(ast.Break())

    def continue_(self) -> None:
        self.block.append(ast.Continue())

    def global_(self, *vars: Variable) -> None:
        self.block.append(ast.Global([var.name for var in vars]))

    def nonlocal_(self, *vars: Variable) -> None:
        self.block.append(ast.Nonlocal([var.name for var in vars]))

    def yield_(self, value: Value) -> None:
        self.expr(yield_(value))

    def yield_from(self, value: Value) -> None:
        self.expr(yield_from(value))

    def await_(self, value: Value) -> None:
        self.expr(await_(value))

    @overload
    def with1(
        self, contextmanager: Value, *, is_async: bool = ...
    ) -> tuple[None, "Builder"]: ...

    @overload
    def with1(
        self, contextmanager: Value, binding: str, *, is_async: bool = ...
    ) -> tuple[Variable, "Builder"]: ...

    def with1(
        self,
        contextmanager: Value,
        binding: str | None = None,
        *,
        is_async: bool = False,
    ) -> tuple[Variable | None, "Builder"]:
        (var,), builder = self.with_((contextmanager, binding), is_async=is_async)
        return var, builder

    def with_(
        self, *items: tuple[Value, str | None], is_async: bool = False
    ) -> tuple[list[Variable | None], "Builder"]:
        withitems = []
        vars = []
        for value, name in items:
            if name is None:
                name_var = None
            else:
                name_var = self.scope.declare(name)
            vars.append(name_var)
            ast_name = ast.Name(name_var.name) if name_var else None
            withitems.append(ast.withitem(value.expr, ast_name))
        with_ = ast.AsyncWith(withitems) if is_async else ast.With(withitems)
        self.block.append(with_)
        return vars, Builder(with_.body, self.scope)

    def new_function(
        self, var: Variable, parameters: Parameters, *, is_async: bool = False
    ) -> tuple[list[Variable], "Builder"]:
        constructor = ast.AsyncFunctionDef if is_async else ast.FunctionDef
        func = constructor(var.name, parameters._as_arguments())
        self.block.append(func)
        func_scope = self.scope.new_child()
        param_vars = [func_scope.declare(name) for name in parameters._all()]
        return param_vars, Builder(func.body, func_scope)

    def new_class(self, var: Variable, *bases: Value) -> "Builder":
        class_ = ast.ClassDef(var.name, bases=[v.expr for v in bases])
        self.block.append(class_)
        return Builder(class_.body, self.scope.new_child())


def new_module() -> tuple[ast.Module, Builder]:
    m = ast.Module()
    return m, Builder(m.body)


def unparse(node: ast.AST) -> str:
    node = ast.fix_missing_locations(node)
    return ast.unparse(node)
