import ast
import textwrap
import unittest

from pyastgen import (
    Builder,
    InvalidComprehension,
    NameCollision,
    Parameters,
    Scope,
    Slice,
    Starred,
    await_,
    constant,
    dict_,
    list_,
    new_module,
    set_,
    tuple_,
    unparse,
    yield_,
    yield_from,
)


class TestValueOperations(unittest.TestCase):
    def test_arithmetic_operations(self):
        a = constant(5)
        b = constant(3)

        add_node = a.add(b)
        self.assertEqual(unparse(add_node.expr), "5 + 3")

        sub_node = a.sub(b)
        self.assertEqual(unparse(sub_node.expr), "5 - 3")

        mul_node = a.mul(b)
        self.assertEqual(unparse(mul_node.expr), "5 * 3")

        floordiv_node = a.floordiv(b)
        self.assertEqual(unparse(floordiv_node.expr), "5 // 3")

        pow_node = a.pow(b)
        self.assertEqual(unparse(pow_node.expr), "5 ** 3")

        mod_node = a.mod(b)
        self.assertEqual(unparse(mod_node.expr), "5 % 3")

        matmul_node = a.matmul(b)
        self.assertEqual(unparse(matmul_node.expr), "5 @ 3")

    def test_bitwise_operations(self):
        a = constant(5)
        b = constant(3)

        bit_and_node = a.bit_and(b)
        self.assertEqual(unparse(bit_and_node.expr), "5 & 3")

        bit_or_node = a.bit_or(b)
        self.assertEqual(unparse(bit_or_node.expr), "5 | 3")

        bit_xor_node = a.bit_xor(b)
        self.assertEqual(unparse(bit_xor_node.expr), "5 ^ 3")

        invert_node = constant(5).invert()
        self.assertEqual(unparse(invert_node.expr), "~5")

        lshift_node = a.lshift(b)
        self.assertEqual(unparse(lshift_node.expr), "5 << 3")

        rshift_node = a.rshift(b)
        self.assertEqual(unparse(rshift_node.expr), "5 >> 3")

    def test_logical_operations(self):
        a = constant(True)
        b = constant(False)

        and_node = a.and_(b)
        self.assertEqual(unparse(and_node.expr), "True and False")

        or_node = a.or_(b)
        self.assertEqual(unparse(or_node.expr), "True or False")

        not_node = a.not_()
        self.assertEqual(unparse(not_node.expr), "not True")

    def test_comparisons(self):
        a = constant(5)
        b = constant(3)

        eq_node = a.eq(b)
        self.assertEqual(unparse(eq_node.expr), "5 == 3")

        ne_node = a.ne(b)
        self.assertEqual(unparse(ne_node.expr), "5 != 3")

        lt_node = a.lt(b)
        self.assertEqual(unparse(lt_node.expr), "5 < 3")

        lte_node = a.lte(b)
        self.assertEqual(unparse(lte_node.expr), "5 <= 3")

        gt_node = a.gt(b)
        self.assertEqual(unparse(gt_node.expr), "5 > 3")

        gte_node = a.gte(b)
        self.assertEqual(unparse(gte_node.expr), "5 >= 3")

        contains_node = a.contains(b)
        self.assertEqual(unparse(contains_node.expr), "5 in 3")

        is_node = a.is_(b)
        self.assertEqual(unparse(is_node.expr), "5 is 3")

        is_not_node = a.is_not(b)
        self.assertEqual(unparse(is_not_node.expr), "5 is not 3")

    def test_subscripting(self):
        a = constant([1, 2, 3])
        index = constant(1)

        subscript_node = a.subscript(index)
        self.assertEqual(unparse(subscript_node.expr), "[1, 2, 3][1]")

        slice_ = Slice(lower=constant(1), upper=constant(3))
        slice_subscript_node = a.subscript(slice_)
        self.assertEqual(unparse(slice_subscript_node.expr), "[1, 2, 3][1:3]")

    def test_method_calls_and_attributes(self):
        obj = constant({"a": 1})

        method_call_node = obj.call()
        self.assertEqual(unparse(method_call_node.expr), "{'a': 1}()")

        with_kwargs_node = obj.call(a=constant(2))
        self.assertEqual(unparse(with_kwargs_node.expr), "{'a': 1}(a=2)")

        attribute_node = obj.attribute("a")
        self.assertEqual(unparse(attribute_node.expr), "{'a': 1}.a")

        method_chain_node = obj.attribute("a").call()
        self.assertEqual(unparse(method_chain_node.expr), "{'a': 1}.a()")


class TestVariableScope(unittest.TestCase):
    def test_variable_declaration(self):
        # Test variable declaration with name collision
        scope = Scope()

        # Declare variables with the same name
        var1 = scope.declare("x")
        var2 = scope.declare("x")
        var3 = scope.declare("x")

        # Check variable names
        self.assertEqual(var1.name, "x")
        self.assertEqual(var2.name, "x1")
        self.assertEqual(var3.name, "x2")

        # Check that variables are in the scope
        self.assertIn(var1, scope._mapping.values())
        self.assertIn(var2, scope._mapping.values())
        self.assertIn(var3, scope._mapping.values())

    def test_scope_management(self):
        # Test nested scopes and variable shadowing
        outer_scope = Scope()
        inner_scope = outer_scope.new_child()

        # Declare variables in outer and inner scopes
        outer_var = outer_scope.declare("x")
        inner_var = inner_scope.declare("x")

        # Check that inner scope shadows outer scope
        self.assertEqual(outer_var.name, "x")
        self.assertEqual(inner_var.name, "x")

        # Check that outer scope does not contain inner variable
        self.assertNotIn(inner_var, outer_scope._mapping.values())

        # Check that inner scope contains its own variable
        self.assertIn(inner_var, inner_scope._mapping.values())

    def test_exact_declaration(self):
        # Test exact declaration with NameCollision
        scope = Scope()

        # Declare a variable
        scope.declare("x")

        # Try to declare the same variable with exact=True
        with self.assertRaises(NameCollision):
            scope.declare("x", exact=True)

    def test_scope_get(self):
        scope = Scope()

        with self.assertRaises(KeyError):
            scope.get("x")

        var = scope.declare("x")
        var2 = scope.declare("x")

        self.assertNotEqual(var.name, var2.name)
        self.assertEqual(scope.get("x"), var)
        self.assertEqual(scope.get("x1"), var2)


class TestBuilderStatements(unittest.TestCase):
    def test_basic_assignments(self):
        builder = Builder([])
        var = builder.declare("x")
        value = constant(5)
        var.store(builder, value)
        self.assertEqual(len(builder.block), 1)
        self.assertEqual(unparse(builder.block[0]), "x = 5")

    def test_augmented_assignments(self):
        # Test all augmented assignments
        augmented_methods = [
            ("add_assign", "x += 3"),
            ("sub_assign", "x -= 3"),
            ("mul_assign", "x *= 3"),
            ("matmul_assign", "x @= 3"),
            ("div_assign", "x /= 3"),
            ("floordiv_assign", "x //= 3"),
            ("mod_assign", "x %= 3"),
            ("pow_assign", "x **= 3"),
            ("lshift_assign", "x <<= 3"),
            ("rshift_assign", "x >>= 3"),
            ("bit_or_assign", "x |= 3"),
            ("bit_and_assign", "x &= 3"),
            ("bit_xor_assign", "x ^= 3"),
        ]
        for method, expected in augmented_methods:
            builder = Builder([])
            var = builder.declare("x")
            var.value = constant(5)
            getattr(builder, method)(var.target, constant(3))
            self.assertEqual(unparse(builder.block[0]), expected)

    def test_if_statements(self):
        builder = Builder([])
        condition = constant(True)
        body_builder, else_builder = builder.if_(condition)
        var = builder.declare("x")
        var.store(body_builder, constant(1))
        var.store(else_builder, constant(2))
        self.assertEqual(len(builder.block), 1)
        if_node = builder.block[0]
        self.assertEqual(unparse(if_node), "if True:\n    x = 1\nelse:\n    x = 2")

    def test_while_loops(self):
        builder = Builder([])
        condition = constant(True)
        body_builder, else_builder = builder.while_(condition)
        var = body_builder.declare("x")
        var.store(body_builder, constant(1))
        var.store(else_builder, constant(2))
        self.assertEqual(len(builder.block), 1)
        while_node = builder.block[0]
        self.assertEqual(
            unparse(while_node), "while True:\n    x = 1\nelse:\n    x = 2"
        )

    def test_for_loops(self):
        builder = Builder([])
        x = builder.declare("x")
        y = builder.declare("y")
        iterable = constant([1, 2, 3])
        body_builder, else_builder = builder.for_(x.target, iterable)
        y.store(body_builder, constant(1))
        y.store(else_builder, constant(2))
        self.assertEqual(len(builder.block), 1)
        for_node = builder.block[0]
        self.assertEqual(
            unparse(for_node), "for x in [1, 2, 3]:\n    y = 1\nelse:\n    y = 2"
        )

    def test_async_for_loops(self):
        builder = Builder([])
        x = builder.declare("x")
        y = builder.declare("y")
        iterable = constant([1, 2, 3])
        body_builder, else_builder = builder.for_(x.target, iterable, is_async=True)
        y.store(body_builder, constant(1))
        y.store(else_builder, constant(2))
        self.assertEqual(len(builder.block), 1)
        for_node = builder.block[0]
        self.assertEqual(
            unparse(for_node), "async for x in [1, 2, 3]:\n    y = 1\nelse:\n    y = 2"
        )

    def test_function_definitions(self):
        builder = Builder([])
        var = builder.declare("my_function")
        params = Parameters(
            args=["x", "y", "z"],
            vararg="args",
            defaults=[constant(4), constant(5)],
            kwonlyargs=["q"],
            kw_defaults={"q": constant(1)},
            kwarg="kwargs",
        )
        (x, y, z, args, q, kwargs), func_builder = builder.new_function(var, params)
        result_var = func_builder.declare("result")
        result_var.store(func_builder, x.value.add(y.value).add(z.value).add(q.value))
        func_builder.return_(result_var.value)
        self.assertEqual(len(builder.block), 1)
        func_node = builder.block[0]
        self.assertEqual(
            unparse(func_node),
            "def my_function(x, y=4, z=5, *args, q=1, **kwargs):\n"
            "    result = x + y + z + q\n"
            "    return result",
        )

    def test_class_definitions(self):
        builder = Builder([])
        var = builder.declare("MyClass")
        base = builder.declare("object", exact=True)
        class_builder = builder.new_class(var, base.value)
        class_var = class_builder.declare("x")
        class_var.store(class_builder, constant(5))
        self.assertEqual(len(builder.block), 1)
        class_node = builder.block[0]
        self.assertEqual(unparse(class_node), "class MyClass(object):\n    x = 5")

    def test_imports(self):
        builder = Builder([])
        builder.import_("os", ("sys", "system"))
        self.assertEqual(len(builder.block), 1)
        import_node = builder.block[0]
        self.assertEqual(unparse(import_node), "import os, sys as system")

    def test_import1(self):
        module, builder = new_module()
        astvar = builder.import1("ast")
        self.assertEqual(astvar.name, "ast")
        self.assertEqual(unparse(module), "import ast")

    def test_relative_imports(self):
        builder = Builder([])
        builder.relative_import(".relative", "a", ("b", "c"))
        self.assertEqual(len(builder.block), 1)
        import_node = builder.block[0]
        self.assertEqual(unparse(import_node), "from .relative import a, b as c")

    def test_relative_import1(self):
        module, builder = new_module()
        thing = builder.relative_import1(".relative", "thing")
        self.assertEqual(thing.name, "thing")
        self.assertEqual(unparse(module), "from .relative import thing")

    def test_import_name_conflict(self):
        module, builder = new_module()
        thing = builder.relative_import1(".relative", "thing")
        thing1 = builder.relative_import1(".relative2", "thing")
        thing2 = builder.import1("thing")
        self.assertEqual(thing.name, "thing")
        self.assertEqual(thing1.name, "thing1")
        self.assertEqual(thing2.name, "thing2")
        self.assertEqual(
            unparse(module),
            "from .relative import thing\n"
            "from .relative2 import thing as thing1\n"
            "import thing as thing2",
        )

    def test_comprehension_if_without_for(self):
        _module, builder = new_module()

        comp = builder.comprehension()
        with self.assertRaises(InvalidComprehension):
            comp.if_(constant(True))

    def test_comprehensions(self):
        builder = Builder([])
        comp = builder.comprehension()
        x = comp.for_("x", constant([1, 2, 3]))
        list_comp = comp.list(x.value)
        builder.expr(list_comp)
        self.assertEqual(len(builder.block), 1)
        comp_node = builder.block[0]
        self.assertEqual(unparse(comp_node), "[x for x in [1, 2, 3]]")

    def test_comprehension_with_if(self):
        builder = Builder([])
        comp = builder.comprehension()
        x = comp.for_("x", constant([1, 2, 3]))
        comp.if_(x.value.gt(constant(1)))
        list_comp = comp.list(x.value)
        builder.expr(list_comp)
        self.assertEqual(len(builder.block), 1)
        comp_node = builder.block[0]
        self.assertEqual(unparse(comp_node), "[x for x in [1, 2, 3] if x > 1]")

    def test_comprehension_with_multiple_if(self):
        builder = Builder([])
        comp = builder.comprehension()
        x = comp.for_("x", constant([1, 2, 3]))
        comp.if_(x.value.gt(constant(1)))
        comp.if_(x.value.mod(constant(2)).eq(constant(0)))
        list_comp = comp.list(x.value)
        builder.expr(list_comp)
        self.assertEqual(len(builder.block), 1)
        comp_node = builder.block[0]
        self.assertEqual(
            unparse(comp_node), "[x for x in [1, 2, 3] if x > 1 if x % 2 == 0]"
        )

    def test_comprehension_types(self):
        module, builder = new_module()
        comp = builder.comprehension()
        x = comp.for_("x", constant([1, 2, 3]))
        builder.expr(comp.list(x.value))
        builder.expr(comp.generator(x.value))
        builder.expr(comp.set(x.value))
        builder.expr(comp.dict(x.value, x.value))
        self.assertEqual(
            unparse(module),
            "[x for x in [1, 2, 3]]\n"
            "(x for x in [1, 2, 3])\n"
            "{x for x in [1, 2, 3]}\n"
            "{x: x for x in [1, 2, 3]}",
        )

    def test_async_comprehensions(self):
        builder = Builder([])
        comp = builder.comprehension()
        x = comp.for_("x", constant([1, 2, 3]), is_async=True)
        list_comp = comp.list(x.value)
        builder.expr(list_comp)
        self.assertEqual(len(builder.block), 1)
        comp_node = builder.block[0]
        self.assertEqual(unparse(comp_node), "[x async for x in [1, 2, 3]]")

    def test_pass_statement(self):
        builder = Builder([])
        builder.pass_()
        self.assertEqual(len(builder.block), 1)
        pass_node = builder.block[0]
        self.assertEqual(unparse(pass_node), "pass")

    def test_return_statement(self):
        builder = Builder([])
        builder.return_(constant(5))
        self.assertEqual(len(builder.block), 1)
        return_node = builder.block[0]
        self.assertEqual(unparse(return_node), "return 5")

    def test_expr_statement(self):
        builder = Builder([])
        exception_value = builder.declare("Exception", exact=True).value.call(
            constant("error")
        )
        builder.raise_(exception_value)
        self.assertEqual(len(builder.block), 1)
        expr_node = builder.block[0]
        self.assertEqual(unparse(expr_node), "raise Exception('error')")

    def test_assert_statement(self):
        builder = Builder([])
        builder.assert_(constant(True))
        self.assertEqual(len(builder.block), 1)
        assert_node = builder.block[0]
        self.assertEqual(unparse(assert_node), "assert True")

    def test_delete_statement(self):
        builder = Builder([])
        var = builder.declare("x")
        builder.delete(var.target)
        self.assertEqual(len(builder.block), 1)
        delete_node = builder.block[0]
        self.assertEqual(unparse(delete_node), "del x")

    def test_break_statement(self):
        builder = Builder([])
        builder.break_()
        self.assertEqual(len(builder.block), 1)
        break_node = builder.block[0]
        self.assertEqual(unparse(break_node), "break")

    def test_continue_statement(self):
        builder = Builder([])
        builder.continue_()
        self.assertEqual(len(builder.block), 1)
        continue_node = builder.block[0]
        self.assertEqual(unparse(continue_node), "continue")

    def test_global_statement(self):
        builder = Builder([])
        x = builder.declare("x")
        y = builder.declare("y")
        builder.global_(x, y)
        self.assertEqual(len(builder.block), 1)
        global_node = builder.block[0]
        self.assertEqual(unparse(global_node), "global x, y")

    def test_nonlocal_statement(self):
        builder = Builder([])
        x = builder.declare("x")
        y = builder.declare("y")
        builder.nonlocal_(x, y)
        self.assertEqual(len(builder.block), 1)
        nonlocal_node = builder.block[0]
        self.assertEqual(unparse(nonlocal_node), "nonlocal x, y")

    def test_yield_statement(self):
        builder = Builder([])
        builder.yield_(constant(5))
        self.assertEqual(len(builder.block), 1)
        yield_node = builder.block[0]
        self.assertEqual(unparse(yield_node), "yield 5")

    def test_yield_from_statement(self):
        builder = Builder([])
        builder.yield_from(constant([1, 2, 3]))
        self.assertEqual(len(builder.block), 1)
        yield_from_node = builder.block[0]
        self.assertEqual(unparse(yield_from_node), "yield from [1, 2, 3]")


class TestUtilitiesAndExpressions(unittest.TestCase):
    def test_constant(self):
        # Test basic constants
        self.assertEqual(unparse(constant(5).expr), "5")
        self.assertEqual(unparse(constant("hello").expr), "'hello'")

        # Test boolean constants
        self.assertEqual(unparse(constant(True).expr), "True")
        self.assertEqual(unparse(constant(False).expr), "False")

        # Test None
        self.assertEqual(unparse(constant(None).expr), "None")

        # Test complex numbers
        self.assertEqual(unparse(constant(3 + 4j).expr), "(3+4j)")

        # Test bytes
        self.assertEqual(unparse(constant(b"bytes").expr), "b'bytes'")

    def test_list(self):
        # Test empty list
        self.assertEqual(unparse(list_().expr), "[]")

        # Test list with elements
        elements = [constant(1), constant(2), constant(3)]
        list_node = list_(*elements)
        self.assertEqual(unparse(list_node.expr), "[1, 2, 3]")

        # Test list with starred elements
        starred = Starred(list_(*elements))
        starred_list_node = list_(starred)
        self.assertEqual(unparse(starred_list_node.expr), "[*[1, 2, 3]]")

    def test_tuple(self):
        # Test empty tuple
        self.assertEqual(unparse(tuple_().expr), "()")

        # Test tuple with elements
        elements = [constant(1), constant(2), constant(3)]
        tuple_node = tuple_(*elements)
        self.assertEqual(unparse(tuple_node.expr), "(1, 2, 3)")

        # Test tuple with starred elements
        starred = Starred(tuple_(*elements))
        starred_tuple_node = tuple_(starred)
        self.assertEqual(unparse(starred_tuple_node.expr), "(*(1, 2, 3),)")

    def test_set(self):
        # Test empty set
        self.assertEqual(unparse(set_().expr), "{*()}")

        # Test set with elements
        elements = [constant(1), constant(2), constant(3)]
        set_node = set_(*elements)
        self.assertEqual(unparse(set_node.expr), "{1, 2, 3}")

        # Test set with starred elements
        starred = Starred(set_(*elements))
        starred_set_node = set_(starred)
        self.assertEqual(unparse(starred_set_node.expr), "{*{1, 2, 3}}")

    def test_dict(self):
        # Test empty dict
        self.assertEqual(unparse(dict_().expr), "{}")

        # Test dict with key-value pairs
        items = [(constant("a"), constant(1)), (constant("b"), constant(2))]
        dict_node = dict_(*items)
        self.assertEqual(unparse(dict_node.expr), "{'a': 1, 'b': 2}")

        # Test dict with unpacking
        unpacking = (None, constant({"x": 3, "y": 4}))
        dict_node_unpacking = dict_(unpacking)
        self.assertEqual(unparse(dict_node_unpacking.expr), "{**{'x': 3, 'y': 4}}")

    def test_yield(self):
        # Test basic yield
        self.assertEqual(unparse(yield_().expr), "(yield)")
        self.assertEqual(unparse(yield_(constant(None)).expr), "(yield None)")
        self.assertEqual(unparse(yield_(constant(5)).expr), "(yield 5)")

        # Test yield from
        self.assertEqual(
            unparse(yield_from(constant([1, 2, 3])).expr), "(yield from [1, 2, 3])"
        )

    def test_await(self):
        # Test await
        self.assertEqual(unparse(await_(constant(5)).expr), "await 5")
        self.assertEqual(
            unparse(await_(constant(5)).attribute("test").expr), "(await 5).test"
        )
        module, builder = new_module()
        builder.await_(constant(5))
        self.assertEqual(unparse(module), "await 5")


class TestPerformanceAndScalability(unittest.TestCase):
    def test_deeply_nested_structures(self):
        # Create a deeply nested AST
        depth = 100
        module, builder = new_module()
        current = constant(0)
        for i in range(1, depth):
            current = current.add(constant(i))
        builder.expr(current)

        # Check that the AST is correctly generated
        self.assertEqual(len(builder.block), 1)
        self.assertEqual(unparse(module), " + ".join(map(str, range(depth))))

    def test_complex_scenarios(self):
        # Create a complex scenario with nested comprehensions and functions
        module, builder = new_module()
        var = builder.declare("complex_function")
        params = Parameters(args=["x"])
        (x,), func_builder = builder.new_function(var, params)

        comp = func_builder.comprehension()
        y = comp.for_("y", constant([1, 2, 3]))
        list_comp = comp.list(y.value.add(x.value))

        result = func_builder.declare("result")
        result.store(func_builder, list_comp)
        func_builder.return_(result.value)

        # Check that the AST is correctly generated
        self.assertEqual(len(builder.block), 1)
        expected = (
            "def complex_function(x):\n"
            "    result = [y + x for y in [1, 2, 3]]\n"
            "    return result"
        )
        self.assertEqual(unparse(module), expected)

    def test_complex_scoping(self):
        module, builder = new_module()

        global_var = builder.declare("var", constant(111))

        outer = builder.declare("outer")
        (), outer_builder = builder.new_function(outer, Parameters())
        outer_builder.global_(global_var)
        global_var.store(outer_builder, constant(222))
        outer_var = outer_builder.declare("var2", constant(0))

        inner = builder.declare("inner")
        (), inner_builder = outer_builder.new_function(inner, Parameters())
        inner_builder.nonlocal_(outer_var)
        outer_var.store(inner_builder, constant(333))

        outer_builder.expr(inner.value.call())
        outer_builder.assert_(outer_var.value.eq(constant(333)))

        builder.expr(outer.value.call())
        builder.assert_(global_var.value.eq(constant(222)))

        ast.fix_missing_locations(module)
        code = compile(module, "<script>", "exec", dont_inherit=True, optimize=0)
        exec(code, {}, {})


class TestWithBlockCodeGeneration(unittest.TestCase):
    def setUp(self):
        self.module, self.builder = new_module()

    def test_simple_with_block(self):
        # Create the with block
        value = self.builder.declare("context_manager").value
        _, body_builder = self.builder.with_((value, None), is_async=False)

        # Add an empty body
        body_builder.pass_()

        # Generate code
        generated_code = unparse(self.module)

        # Expected code
        expected_code = textwrap.dedent("""\
            with context_manager:
                pass
        """)

        self.assertEqual(generated_code.strip(), expected_code.strip())

    def test_with_variable_assignment(self):
        value = self.builder.declare("context_manager").value

        # Create the with block, assigning to 'ctx'
        var, body_builder = self.builder.with1(value, "ctx", is_async=False)

        # Inside the with block: use 'ctx'
        body_builder.expr(
            body_builder.declare("print", exact=True).value.call(
                var.value.add(constant(" inside"))
            )
        )

        # Generate code
        generated_code = unparse(self.module)

        # Expected code
        expected_code = textwrap.dedent("""\
            with context_manager as ctx:
                print(ctx + ' inside')
        """)

        self.assertEqual(generated_code.strip(), expected_code.strip())

    def test_multiple_context_managers(self):
        cm1 = self.builder.declare("context_manager1").value
        cm2 = self.builder.declare("context_manager2").value

        # Create the with block, assigning both managers
        (var1, var2), body_builder = self.builder.with_((cm1, "ctx1"), (cm2, "ctx2"))

        # Inside the with block: use both ctx1 and ctx2
        body_builder.expr(
            body_builder.declare("print", exact=True).value.call(
                var1.value.add(constant(" and ")).add(var2.value)
            )
        )

        # Generate code
        generated_code = unparse(self.module)

        # Expected code
        expected_code = textwrap.dedent("""\
            with context_manager1 as ctx1, context_manager2 as ctx2:
                print(ctx1 + ' and ' + ctx2)
        """)

        self.assertEqual(generated_code.strip(), expected_code.strip())

    def test_async_with_block(self):
        value = self.builder.declare("async_context_manager").value

        # Create the async with block
        _var, body_builder = self.builder.with_((value, None), is_async=True)

        # Inside the with block: print a message
        body_builder.expr(
            body_builder.declare("print", exact=True).value.call(
                constant("Inside async context")
            )
        )

        # Generate code
        generated_code = unparse(self.module)

        # Expected code
        expected_code = textwrap.dedent("""\
            async with async_context_manager:
                print('Inside async context')
        """)

        self.assertEqual(generated_code.strip(), expected_code.strip())


if __name__ == "__main__":
    unittest.main()
