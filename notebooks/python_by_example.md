---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---


<a id='python-by-example'></a>
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>

+++

# An Introductory Example


<a id='index-0'></a>

+++

## The Task: Plotting a White Noise Process

Suppose we want to simulate and plot the white noise
process $ \epsilon_0, \epsilon_1, \ldots, \epsilon_T $, where each draw $ \epsilon_t $ is independent standard normal.

In other words, we want to generate figures that look something like this:

![https://python-programming.quantecon.org/_static/lecture_specific/python_by_example/test_program_1_updated.png](https://python-programming.quantecon.org/_static/lecture_specific/python_by_example/test_program_1_updated.png)

  
(Here $ t $ is on the horizontal axis and $ \epsilon_t $ is on the
vertical axis.)

We’ll do this in several different ways, each time learning something more
about Python.

We run the following command first, which helps ensure that plots appear in the
notebook if you run it on your own machine.

+++

## Version 1


<a id='ourfirstprog'></a>
Here are a few lines of code that perform the task we set

```{code-cell} ipython3
:hide-output: false

import numpy as np
import matplotlib.pyplot as plt

ϵ_values = np.random.randn(100)
plt.plot(ϵ_values)
plt.show()
```

Let’s break this program down and see how it works.


<a id='import'></a>

+++

### Imports

The first two lines of the program import functionality from external code
libraries.

The first line imports [NumPy](https://python-programming.quantecon.org/numpy.html), a favorite Python package for tasks like

- working with arrays (vectors and matrices)  
- common mathematical functions like `cos` and `sqrt`  
- generating random numbers  
- linear algebra, etc.  


After `import numpy as np` we have access to these attributes via the syntax `np.attribute`.

Here’s two more examples

```{code-cell} ipython3
:hide-output: false

np.sqrt(4)
```

#### Why So Many Imports?

Python programs typically require several import statements.

The reason is that the core language is deliberately kept small, so that it’s easy to learn and maintain.

When you want to do something interesting with Python, you almost always need
to import additional functionality.

+++

### Importing Names Directly

Recall this code that we saw above

```{code-cell} ipython3
:hide-output: false

import numpy as np

np.sqrt(4)
```

Here’s another way to access NumPy’s square root function

```{code-cell} ipython3
:hide-output: false

from numpy import sqrt

sqrt(4)
```

This is also fine.

The advantage is less typing if we use `sqrt` often in our code.

The disadvantage is that, in a long program, these two lines might be
separated by many other lines.

Then it’s harder for readers to know where `sqrt` came from, should they wish to.

+++

### Random Draws

Returning to our program that plots white noise, the remaining three lines
after the import statements are

```{code-cell} ipython3
:hide-output: false

ϵ_values = np.random.randn(100)
plt.plot(ϵ_values)
plt.show()
```

The first line generates 100 (quasi) independent standard normals and stores
them in `ϵ_values`.

The next two lines genererate the plot.

We can and will look at various ways to configure and improve this plot below.

+++

## Alternative Implementations

Let’s try writing some alternative versions of [our first program](#ourfirstprog), which plotted IID draws from the standard normal distribution.

The programs below are less efficient than the original one, and hence
somewhat artificial.

But they do help us illustrate some important Python syntax and semantics in a familiar setting.

+++

### A Version with a For Loop

Here’s a version that illustrates `for` loops and Python lists.


<a id='firstloopprog'></a>

```{code-cell} ipython3
:hide-output: false

ts_length = 100
ϵ_values = []   # empty list

for i in range(ts_length):
    e = np.random.randn()
    ϵ_values.append(e)

plt.plot(ϵ_values)
plt.show()
```



Let’s study some parts of this program in more detail.


+++

### Lists


<a id='index-3'></a>
Consider the statement `ϵ_values = []`, which creates an empty list.

Lists are a *native Python data structure* used to group a collection of objects.

Items in lists are ordered, and duplicates are allowed in lists.

For example, try

```{code-cell} ipython3
:hide-output: false

x = [10, 'foo', False]
type(x)
```

The first element of `x` is an [integer](https://en.wikipedia.org/wiki/Integer_%28computer_science%29), the next is a [string](https://en.wikipedia.org/wiki/String_%28computer_science%29), and the third is a [Boolean value](https://en.wikipedia.org/wiki/Boolean_data_type).

When adding a value to a list, we can use the syntax `list_name.append(some_value)`

```{code-cell} ipython3
:hide-output: false

x
```

```{code-cell} ipython3
:hide-output: false

x.append(2.5)
x
```

Here `append()` is what’s called a *method*, which is a function “attached to” an object—in this case, the list `x`.

We’ll learn all about methods [later on](https://python-programming.quantecon.org/oop_intro.html), but just to give you some idea,

- Python objects such as lists, strings, etc. all have methods that are used to manipulate the data contained in the object.  
- String objects have [string methods](https://docs.python.org/3/library/stdtypes.html#string-methods), list objects have [list methods](https://docs.python.org/3/tutorial/datastructures.html#more-on-lists), etc.  


Another useful list method is `pop()`

```{code-cell} ipython3
:hide-output: false

x
```

```{code-cell} ipython3
:hide-output: false

x.pop()
```

```{code-cell} ipython3
:hide-output: false

x
```

Lists in Python are zero-based (as in C, Java or Go), so the first element is referenced by `x[0]`

```{code-cell} ipython3
:hide-output: false

x[0]   # first element of x
```

```{code-cell} ipython3
:hide-output: false

x[1]   # second element of x
```

### The For Loop


<a id='index-4'></a>
Now let’s consider the `for` loop from [the program above](#firstloopprog), which was

```{code-cell} ipython3
:hide-output: false

for i in range(ts_length):
    e = np.random.randn()
    ϵ_values.append(e)
```

Python executes the two indented lines `ts_length` times before moving on.

These two lines are called a `code block`, since they comprise the “block” of code that we are looping over.

Unlike most other languages, Python knows the extent of the code block *only from indentation*.

In our program, indentation decreases after line `ϵ_values.append(e)`, telling Python that this line marks the lower limit of the code block.

More on indentation below—for now, let’s look at another example of a `for` loop

```{code-cell} ipython3
:hide-output: false

animals = ['dog', 'cat', 'bird']
for animal in animals:
    print("The plural of " + animal + " is " + animal + "s")
```

This example helps to clarify how the `for` loop works: When we execute a
loop of the form

+++ {"hide-output": false}

```python3
for variable_name in sequence:
    <code block>
```

+++

The Python interpreter performs the following:

- For each element of the `sequence`, it “binds” the name `variable_name` to that element and then executes the code block.  


The `sequence` object can in fact be a very general object, as we’ll see
soon enough.

+++

### A Comment on Indentation


<a id='index-5'></a>
In discussing the `for` loop, we explained that the code blocks being looped over are delimited by indentation.

In fact, in Python, **all** code blocks (i.e., those occurring inside loops, if clauses, function definitions, etc.) are delimited by indentation.

Thus, unlike most other languages, whitespace in Python code affects the output of the program.

Once you get used to it, this is a good thing: It

- forces clean, consistent indentation, improving readability  
- removes clutter, such as the brackets or end statements used in other languages  


On the other hand, it takes a bit of care to get right, so please remember:

- The line before the start of a code block always ends in a colon  
  - `for i in range(10):`  
  - `if x > y:`  
  - `while x < 100:`  
  - etc., etc.  
- All lines in a code block **must have the same amount of indentation**.  
- The Python standard is 4 spaces, and that’s what you should use.  

+++

### While Loops


<a id='index-6'></a>
The `for` loop is the most common technique for iteration in Python.

But, for the purpose of illustration, let’s modify [the program above](#firstloopprog) to use a `while` loop instead.


<a id='whileloopprog'></a>

```{code-cell} ipython3
:hide-output: false

ts_length = 100
ϵ_values = []
i = 0
while i < ts_length:
    e = np.random.randn()
    ϵ_values.append(e)
    i = i + 1
plt.plot(ϵ_values)
plt.show()
```

A while loop will keep executing the code block delimited by indentation until the condition (`i < ts_length`) is satisfied.

In this case, the program will keep adding values to the list `ϵ_values` until `i` equals `ts_length`:

```{code-cell} ipython3
:hide-output: false

i == ts_length #the ending condition for the while loop
```

Note that

- the code block for the `while` loop is again delimited only by indentation.  
- the statement  `i = i + 1` can be replaced by `i += 1`.  

+++

## Another Application

Let’s do one more application before we turn to exercises.

In this application, we plot the balance of a bank account over time.

There are no withdraws over the time period, the last date of which is denoted
by $ T $.

The initial balance is $ b_0 $ and the interest rate is $ r $.

The balance updates from period $ t $ to $ t+1 $ according to $ b_{t+1} = (1 + r) b_t $.

In the code below, we generate and plot the sequence $ b_0, b_1, \ldots, b_T $.

Instead of using a Python list to store this sequence, we will use a NumPy
array.

```{code-cell} ipython3
:hide-output: false

r = 0.025         # interest rate
T = 50            # end date
b = np.empty(T+1) # an empty NumPy array, to store all b_t
b[0] = 10         # initial balance

for t in range(T):
    b[t+1] = (1 + r) * b[t]

plt.plot(b, label='bank balance')
plt.legend()
plt.show()
```

The statement `b = np.empty(T+1)` allocates storage in memory for `T+1`
(floating point) numbers.

These numbers are filled in by the `for` loop.

Allocating memory at the start is more efficient than using a Python list and
`append`, since the latter must repeatedly ask for storage space from the
operating system.

Notice that we added a legend to the plot — a feature you will be asked to
use in the exercises.

+++

## Exercise

Simulate and plot the correlated time series

$$
x_{t+1} = \alpha \, x_t + \epsilon_{t+1}
\quad \text{where} \quad
x_0 = 0
\quad \text{and} \quad t = 0,\ldots,T
$$

The sequence of shocks $ \{\epsilon_t\} $ is assumed to be IID and standard normal.

In your solution, restrict your import statements to

```{code-cell} ipython3
:hide-output: false

import numpy as np
import matplotlib.pyplot as plt
```


Set $ T=200 $ and $ \alpha = 0.9 $.

```{code-cell} ipython3
# Put your code here
```

```{code-cell} ipython3
for _ in range(10):
    print("solution below")
```

```{code-cell} ipython3
:hide-output: false

α = 0.9
T = 200
x = np.empty(T+1)
x[0] = 0

for t in range(T):
    x[t+1] = α * x[t] + np.random.randn()

plt.plot(x)
plt.show()
```

## Exercise 

Starting with your solution to exercise 1, plot three simulated time series,
one for each of the cases $ \alpha=0 $, $ \alpha=0.8 $ and $ \alpha=0.98 $.

Use a `for` loop to step through the $ \alpha $ values.

If you can, add a legend, to help distinguish between the three time series.

Hints:

- If you call the `plot()` function multiple times before calling `show()`, all of the lines you produce will end up on the same figure.  
- For the legend, noted that the expression `'foo' + str(42)` evaluates to `'foo42'`.  

```{code-cell} ipython3
for _ in range(12):
    print('solution below')
```

```{code-cell} ipython3
α_values = [0.0, 0.8, 0.98]
T = 200
x = np.empty(T+1)

for α in α_values:
    x[0] = 0
    for t in range(T):
        x[t+1] = α * x[t] + np.random.randn()
    plt.plot(x, label=f'$\\alpha = {α}$')

plt.legend()
plt.show()
```

```{code-cell} ipython3

```
