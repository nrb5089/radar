

==============================
Setting up Python environment
==============================

Python should be installed on your machine under the C:\Python

Recommended Development Environment Setup
------------------------------------------

This is my environment I use, but feel free to use whatever makes sense.

Spyder, Pycharm, or some other IDE
------------------------------------------
Contact IT for installation.


No frills CMD and Notepad++
------------------------------------------
Open up the ``Command Prompt`` using any method you like (I press the ``Windows`` key and type ``cmd`` and hit ``enter``).  

Create a batch file by typing 

.. code-block:: bat

    notepad mysetup.bat


Place the following text in the batch file (Note that you may need to adjust the paths) 

.. code-block:: bat

    @ECHO OFF
    DOSKEY py="<full path>/Python/python.exe"
    DOSKEY npp="<full path>/Notepad++/notepad++.exe"


At the time of this writing, Python should be in ``C:`` and Notepad++ should be in either ``C:/Program Files`` or ``C:/Program Files x86`` on the associated DEV machines.  Save and/or save and close the batch file.  Type ``mysetup.bat``, this should initialize the cmd prompt to now use your short hand expressions ``py`` and ``npp`` for the respective programs.  Create a new ``.py`` file by typing ``npp hello_world.py``.  If  **Notepad++** opens up with this new file, this portion works correctly.  Now add ``print('hello world!')`` to your new ``.py`` file.  Save and/or save and close the new ``hello_world.py`` file.  In the CMD Prompt, type ``py hello_world.py``.  If you received a message in the next line ``hello world!``, then everything is working.  This is the method by which you can execute ``.py`` files

**Note** : You will have to refresh ``mysetup.bat`` for each new CMD window openned.


Why Python?
------------------------------------------

Python is a relatively easy programming language that we use for signal processing demonstration and prototyping of algorithms.  The key mathematics come from operations in linear algebra and concepts in statistics/probability.  We use the analogy of 1D and 2D arrays to correspond to vectors and matrices, respectively.  Ideally, we would do most of our low-level signal processing at the hardware level either via ASICs or FPGAs.   The following tutorial provides some basic arithmetic and implementation syntax, along with the mathematical equivalent statements.  

Using Numpy Arrays
=====================

NumPy is a Python library used for working with arrays. It also has functions for working in the domain of linear algebra, fourier transform, and matrices.

NumPy stands for 'Numerical Python' and comes with Python distributions such as Anaconda.


Importing and using NumPy
------------------------------------------

Once NumPy is installed, you need to import it into your Python environment. You can do so with the following line of code:


``import numpy as np``


We use ``as np`` so that we can refer to numpy with the shortened 'np' instead of typing out 'numpy' each time we want to use it.

Creating a NumPy array
------------------------------------------

Vectors and Matrices are critical to understanding signal processing algorithms, we denote vectors as bold lower case $\textbf{v}$ and matrices as bold upper case $\textbf{M}$.  Tensors (informally, matrices with greater than 2 dimensions), are also represented by multidimensional arrays.  The starting example are all real elements, but extend to complex numbers as well.  Python/Numpy uses $1j$ or $1i$ to denote imaginary numbers, i.e., $1 + 2j$.  

A numpy array is a grid of values, all of the same type, and is indexed by a tuple of nonnegative integers. 

You can create a numpy array from a Python list or tuple using the `array` function. Here's an example

.. code-block:: python

    import numpy as np

    ## Creating a 1D array (Vector)
    v = np.array([1, 2, 3])
    print(v)

    ## Creating a 2D array (Matrix)
    M = np.array([[1, 2, 3], [4, 5, 6]])
    print(M)

    ## Creating a 3D array (Tensor)
    mytensor = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    print(mytensor)

Array Indexing
------------------
The $n$th element of a vector may be denoted as $v[n]$ or $v_n$.  We will denote this in unbolded since it is scalar, i.e., $\textbf{v} = [v_0,\dots,v_n,\dots]$.

You can access the array elements as

.. code-block:: python

    import numpy as np

    v = np.array([1, 2, 3, 4, 5])

    print(v[1])  # Output: 2
    print(v[2] + v[3])  # Output: 7

Likewise for matrices $M[m,n]$ or $M_{m,n}$ denotes the element in row $m$ and column $n$.
For 2D arrays, you need to use comma-separated indices

.. code-block:: python

    import numpy as np

    M = np.array([[1,2,3,4,5], [6,7,8,9,10]])

    # Accessing the element at 1st row and 2nd column
    print(M[1, 2])  # Output: 8


Array Slicing
------------------------------------------
Slices of vectors or matrices are denoted $\textbf{v}[m:n]$ or $\textbf{v}_{m:n}$, and for matrices, $\textbf{M}[m:n,p:q]$ or $\textbf{M}_{m:n,p:q}$.
NumPy arrays can be sliced, You can slice a NumPy array like this

.. code-block:: python

    import numpy as np

    v = np.array([1, 2, 3, 4, 5, 6, 7])

    print(v[1:5])  # Output: array([2, 3, 4, 5])


For 2D arrays, it works similarly:

.. code-block:: python 

    import numpy as np

    M = np.array([[1,2,3,4,5], [6,7,8,9,10]])

    # Accessing the first 2 elements of the first 2 rows
    print(M[0:2, 0:2])  # Output: array([[1, 2], [6, 7]])


Basic Array Operations
------------------------------------------

You can perform element-wise operations on arrays like addition, subtraction, etc.

.. code-block:: python

    import numpy as np

    arr1 = np.array([1, 2, 3])
    
    arr2 = np.array([4, 5, 6])

    ## Addition
    print(arr1 + arr2)  # Output: array([5, 7, 9])

    ## Multiplication
    print(arr1 * arr2)  # Output: array([ 4, 10, 18])

    ## Subtraction
    print(arr1 - arr2)  # Output: array([-3, -3, -3])

    ## Division
    print(arr1 / arr2)  # Output: array([0.25, 0.4 , 0.5 ])


Mathematical Functions
------------------------------------------

NumPy provides standard mathematical functions like sin, cos, exp, etc. These functions operate element-wise on an array, producing an array as output.

.. code-block:: python

    import numpy as np

    arr = np.array([0, 30, 45, 60, 90])

    ## Convert to radians by multiplying by pi/180
    arr_radians = arr * np.pi / 180

    print(np.sin(arr_radians))


Statistical Functions
------------------------------------------

NumPy provides functions to calculate statistical metrics like mean, median, standard deviation, etc.

.. code-block:: python

    import numpy as np

    arr = np.array([1,2,3,4,5])

    # Mean
    print(np.mean(arr))  # Output: 3.0

    # Median
    print(np.median(arr))  # Output: 3.0

    # Standard Deviation
    print(np.std(arr))  # Output: 1.4142135623730951


Remember, this is just a basic tutorial and NumPy offers many more features and functions. For a comprehensive understanding, you should refer to the official documentation, https://numpy.org/doc/.

Linear Algebra Operations
===========================
Here are a few linear algebra operations related to using NumPy arrays in the context of linear algebra.

Matrix-Vector Multiplication
-----------------------------------
3x3 matrix, $\textbf{A}$, and a 3x1 vector, $\textbf{v}$. Perform matrix-vector multiplication.

.. code-block:: python 

    import numpy as np

    # Define a 3x3 matrix
    A = np.array([[1, 2, 3], 
                  [4, 5, 6], 
                  [7, 8, 9]])

    # Define a 3x1 vector
    v = np.array([2, 1, 3])


    # Multiply the matrix and vector
    result = A @ v

    print(result)


Matrix-Matrix Multiplication
------------------------------------------

3x3 matrices, $\textbf{A}$, $\textbf{B}$, compute the element-wise (Hadamard) product $\textbf{A}\circ\textbf{B}$ and the more common matrix-matrix multiplciation $\textbf{A}\textbf{B}$.  When we talk about matrix-matrix multplication, we always mean the latter here, but will be denoted with $\circ$ otherwise.

.. code-block:: python 

    import numpy as np

    # Define two 3x3 matrices
    A = np.array([[1, 2, 3], 
                  [4, 5, 6], 
                  [7, 8, 9]])

    B = np.array([[10, 11, 12], 
                  [13, 14, 15], 
                  [16, 17, 18]])

    # Perform element-wise (Hadamard) product 
    result_dot = A * B

    print("Result using dot function:\n", result_dot)

    # Perform matrix multiplication using the @ operator
    result_operator = A @ B

    print("Result using @ operator:\n", result_operator)


Complex Numbers
------------------------------------------

Python has built-in support for complex numbers, which are written with a "j" as the imaginary part. Here's a quick introduction:

.. code-block:: python 

    # Creating complex numbers
    x = 3 + 4j
    y = 2 - 3j

    # Real and Imaginary parts
    print(x.real)  # Outputs: 3.0
    print(x.imag)  # Outputs: 4.0

    # Conjugate
    print(x.conjugate())  # Outputs: (3-4j)

    # Magnitude
    magnitude = abs(x)
    print(magnitude)  # Outputs: 5.0

    #Phase (wrapped)
    phi = np.angle(x)
    print(theta) 

    # Addition
    z = x + y
    print(z)  # Outputs: (5+1j)

    # Subtraction
    z = x - y
    print(z)  # Outputs: (1+7j)

    # Multiplication
    z = x * y
    print(z)  # Outputs: (18+1j)

    # Division
    z = x / y
    print(z)  # Outputs: (-0.15384615384615385+1.2307692307692308j)

Transpose and Hermitian
------------------------------------------

Compute the transpose, $\textbf{A}^T$, and Hermitian (conjugate transpose), $\textbf{C}^H$ of a matrix.  

.. code-block:: python 

    import numpy as np

    # Define a Real 3x2 matrix
    A = np.array([[1, 2], 
                  [3, 4], 
                  [5, 6]])

    #Complex Portion
    B = 1j * np.array([[7,8],
                        [9,10],
                        [11,12]])
                        
    C = A + B

    # Compute the transpose
    A_T = np.transpose(A) #alternatively A_T = A.T

    #Hermitian 
    C_H = np.conj(np.transpose(C))
    print(A_T)
    print(C_H)


Inverse
------------------------------------------
Compute the inverse of a matrix, $\textbf{A}^{-1}$.

.. code-block:: python 

    import numpy as np

    # Define a 3x3 matrix
    A = np.array([[1, 2, 1], 
                  [3, 2, 1], 
                  [1, 1, 2]])

    # Compute the inverse
    A_inv = np.linalg.inv(A)

    print(A_inv)


Determinant
------------------------------------------
Compute the determinant of a matrix, $\textrm{det}(\textbf{A})$.

.. code-block:: python

    import numpy as np

    # Define a 3x3 matrix
    A = np.array([[1, 2, 3], 
                  [4, 5, 6], 
                  [7, 8, 9]])

    # Compute the determinant
    det_A = np.linalg.det(A)

    print(det_A)


Solve the System of Linear Equations 
------------------------------------------

$3x + y = 9$ and $x + 2y = 8$.

.. code-block:: python 

    import numpy as np

    # Define the system's matrix
    A = np.array([[3, 1],
                  [1, 2]])

    # Define the constant vector
    b = np.array([9, 8])

    # Solve for [x, y]
    x = np.linalg.solve(A, b)

    print(x)


Eigenvalue Decomposition
------------------------------------------

Find the eigenvalues and eigenvectors of a matrix, $A = \textbf{V}\textbf{D}\textbf{V}^{-1}$.

.. code-block:: python 

    import numpy as np

    # Define a 2x2 matrix
    A = np.array([[4, 1], 
                  [2, 3]])

    # Compute the eigenvalues and eigenvectors
    D, V = np.linalg.eig(A)

    print("Eigenvalues:", D)
    print("Eigenvectors:", V)


Numpy includes just about any linear algebraic operation you would require, definitely check out the documentation [4].  Additionally, more detail on matrix algebra and computations involving them can be found in [1,2].  The original Numpy paper is [3].

Further reading:
----------------

[1] Golub, Gene H., and Charles F. Van Loan. Matrix computations. JHU press, 2013.

[2] Strang, Gilbert. Linear algebra and its applications. 2012.

[3] Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2. (Publisher link).

[4] https://numpy.org/doc/stable/user/index.html#user


Project 
==========

Here are 3 problems related to using the linear algebra capabilities in NumPy, along with their solutions.

Problem 1: Matrix Operations
-----------------------------

Given two matrices ``A`` and ``B``:

``A = np.array([[1, 2], [3, 4], [5, 6]])``

``B = np.array([[2, 5, 11], [7, 10,3]])`` 

Write a Python script to perform the following operations using ``@``, ``.T``, and ``*``:

1. Matrix Multiplication of A and B
2. Element-wise Multiplication of A's transpose and B

**Output**
Matrix multiplication of A and B:

.. code-block:: none
 
    [[16 25 17]
    [34 55 45]
    [52 85 73]]
 
Element-wise multiplication of A's transpose and B:
 
.. code-block:: none

    [[ 2 15 55]
    [14 40 18]]
 
 
Problem 2: Determinant and Inverse
------------------------------------

Given a matrix `C = np.array([[4, 7, 9, 12], [2, 6, 1, 0.5], [1, 10, 1, 4], [5, 4, 6, 1]])`, calculate:

1. The determinant of C
2. The inverse of C

**Output**
Determinant of C: ``-239.5000000000001``

Inverse of C:
.. code-block:: none

     [[ 0.434238    2.35908142 -1.39457203 -0.81210856]
     [-0.11064718 -0.37995825  0.33611691  0.17327766]
     [-0.32985386 -1.84968685  1.02087683  0.79958246]
     [ 0.25052192  0.82254697 -0.49686848 -0.43006263]]

 
Problem 3: Eigenvalues and Eigenvectors
------------------------------------------

For the same matrix ``C``, compute

1. The eigenvalues of ``C``
2. The eigenvectors of ``C``
3. Build a diagonal matrix of the vector of eigenvalues ``np.diag()``
4. Reconstruct C using the diagonal matrix and matrix of eigenvectors. The result will be complex.

**Output**
Eigenvalues of ``C``
.. code-block:: none

     [16.06533523+0.j         -5.9476733 +0.j          0.94116903+1.27306956j
      0.94116903-1.27306956j]
 
Eigenvectors of ``C``
.. code-block:: none

     [[ 0.80738772+0.j          0.50074368+0.j          0.76290198+0.j
       0.76290198-0.j        ]
     [ 0.21412914+0.j         -0.09361387+0.j         -0.21182191-0.08063397j
      -0.21182191+0.08063397j]
     [ 0.3153099 +0.j          0.47556472+0.j         -0.50580583+0.11882169j
      -0.50580583-0.11882169j]
     [ 0.45039254+0.j         -0.71716833+0.j          0.30845146+0.03885582j
       0.30845146-0.03885582j]]

Reconstructed ``C``

.. code-block:: none

     [[ 4. -1.25389341e-16j  7. +6.32484664e-15j  9. -6.24830406e-15j
      12. -3.00509803e-17j]
     [ 2. +1.42874811e-17j  6. +1.86223409e-15j  1. -1.84380385e-15j
       0.5+4.81857318e-18j]
     [ 1. +3.03449671e-17j 10. +2.96493222e-15j  1. -2.72124588e-15j
       4. +2.15705682e-17j]
     [ 5. -3.99186102e-17j  4. +3.76743004e-15j  6. -3.91085454e-15j
       1. -2.77193111e-17j]]

Clean Reconstructed ``C``

.. code-block:: none

     [[ 4.   7.   9.  12. ]
     [ 2.   6.   1.   0.5]
     [ 1.  10.   1.   4. ]
     [ 5.   4.   6.   1. ]]


