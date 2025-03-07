# pyastgen

Builder for generating Python source, inspired by [llvmlite](https://github.com/numba/llvmlite).

For example, the following code generates a fibonacci function:

```python
import pyastgen

module, builder = pyastgen.new_module()

fib = builder.declare("fib")
(n,), fib_builder = builder.new_function(fib, pyastgen.Parameters("n"))
then, _else = fib_builder.if_(n.lt(pyastgen.constant(2)))
then.return_(n)
fib_builder.return_(
    fib.call(n.sub(pyastgen.constant(1))).add(
        fib.call(n.sub(pyastgen.constant(2)))
    )
)

print(pyastgen.unparse(module))
```

Output:

```python
def fib(n, /):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)
```
