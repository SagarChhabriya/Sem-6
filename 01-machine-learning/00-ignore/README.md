
[Study Opedia Numpy](https://studyopedia.com/numpy/datatypes-in-numpy/)


## np.arange() vs np.linspace()
Here’s a **Markdown table** summarizing the differences between `numpy.arange()` and `numpy.linspace()` based on the points you provided:

| Feature                  | `numpy.arange()`                                                                 | `numpy.linspace()`                                                      |
|--------------------------|----------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| **Step Definition**      | Allows you to define the `size` of the step.                                      | Allows you to define the `number` of steps.                             |
| **Example**              | `arange(0, 10, 2)`: Steps of 2 from 0 to 10 (exclusive).                        | `linspace(0, 1, 20)`: 20 evenly spaced numbers from 0 to 1 (inclusive).|
| **Best Use Case**        | Best for creating an `array of integers`.                                         | Best for creating an `array of floats`.                                 |

---


## np.eye() vs np.identity()
- eye can fill shifted diagonals, identity cannot


```py
>>> np.identity(3)                                                  
array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.]])
>>> np.eye(3)                                                       
array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.]])
>>> timeit.timeit("import numpy; numpy.identity(3)", number = 10000)
0.05699801445007324
>>> timeit.timeit("import numpy; numpy.eye(3)", number = 10000)     
0.03787708282470703
>>> timeit.timeit("import numpy", number = 10000)                   
0.00960087776184082
>>> timeit.timeit("import numpy; numpy.identity(1000)", number = 10000)
11.379066944122314
>>> timeit.timeit("import numpy; numpy.eye(1000)", number = 10000)     
11.247124910354614
```
What, then, is the advantage of using identity over eye?

identity just calls eye so there is no difference in how the arrays are constructed. Here's the code for identity:

```py
def identity(n, dtype=None):
    from numpy import eye
    return eye(n, dtype=dtype)
```
As you say, the main difference is that with eye the diagonal can may be offset, whereas identity only fills the main diagonal.

Since the identity matrix is such a common construct in mathematics, it seems the main advantage of using identity is for its name alone.
