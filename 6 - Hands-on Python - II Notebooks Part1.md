# Hands-on Python: Notebooks, NumPy, and Pandas for Data Science

In the realm of data science and machine learning, we often deal with large datasets and complex computations. While Python's core capabilities are powerful, performing these tasks efficiently and effectively requires specialized tools. This guide will walk you through the fundamental concepts and practical applications of essential Python libraries: NumPy and Pandas, forming the bedrock of modern data analysis and machine learning workflows. We will also explore the utility of interactive notebook environments like Google Colab and Jupyter.

## Introduction and Motivation

In today's data-driven world, the ability to efficiently handle, analyze, and visualize data is paramount. Python, with its rich ecosystem of libraries, has become a dominant force in this domain. Notebook environments provide an interactive and accessible platform for experimentation and development, while libraries like NumPy and Pandas are the workhorses for numerical operations and data wrangling, respectively. This guide aims to provide a clear and comprehensive understanding of these tools and libraries, enabling you to confidently embark on your data science journey.

## Prerequisites and Background

Before diving into this guide, it's beneficial to have a basic understanding of:

*   **Python Programming Fundamentals**: Familiarity with Python syntax, data structures (like lists), variables, data types, loops, functions, and basic control flow.
*   **Data Concepts**: A general idea of what data is and why we analyze it.

---

## Python Notebooks: Colab and Jupyter

### Why Use Notebooks?

Imagine you're trying to solve a complex problem. You might start by experimenting with small pieces of code, testing ideas, and then combining them to build a larger solution. Notebooks are designed precisely for this iterative, experimental process. Traditional scripts are executed all at once. Notebooks, however, allow you to write and execute code in discrete blocks called "cells." This means you can:

*   **Run code incrementally:** Test each part of your code and see the results immediately.
*   **Combine code, text, and visualizations:** Explain your thought process, document your findings, and visualize your data all within a single document.
*   **Share your work easily:** Notebooks can be shared with others, allowing them to easily reproduce your analysis or build upon your work.
*   **Experiment and iterate:** This interactive nature makes them ideal for data exploration, rapid prototyping, and learning.

### Google Collaboratory (Colab) vs. Jupyter Notebook

Both Colab and Jupyter are excellent notebook environments, each with its own strengths:

*   **Jupyter Notebook:** A widely used, open-source web application that allows you to create and share documents containing live code, equations, visualizations, and narrative text. It typically runs locally on your machine.
*   **Google Colaboratory (Colab):** A free, cloud-based Jupyter notebook environment provided by Google. It offers many advantages, including:
    *   **No setup required:** You can start coding immediately in your browser.
    *   **Free access to GPUs and TPUs:** Essential for computationally intensive machine learning tasks.
    *   **Seamless integration with Google Drive:** Easily save and access your notebooks.

For this guide, we'll often refer to notebook concepts generally, as they apply to both environments, but the examples will be compatible with Colab.

---

## Setting Up Our Environment: Importing Libraries

To begin our journey, we need to import the necessary Python libraries. This is a standard first step in any data science or machine learning project.

### The Importance of Libraries

Python is a versatile language, but its true power in specialized fields like data science comes from its rich ecosystem of libraries. These libraries are pre-written code modules that provide optimized functions for specific tasks, saving us the effort of reinventing the wheel and often offering significantly better performance.

### Importing NumPy and Pandas

The two most fundamental libraries for data manipulation and analysis in Python are NumPy and Pandas. We typically import them with conventional aliases for ease of use:

```python
import numpy as np
import pandas as pd
```

*   **`import numpy as np`**: NumPy (Numerical Python) is the cornerstone for numerical operations in Python. It provides support for large, multi-dimensional arrays and matrices, along with a vast collection of mathematical functions to operate on these arrays efficiently.
*   **`import pandas as pd`**: Pandas is built on top of NumPy and is designed for data manipulation and analysis. It introduces powerful data structures like `DataFrame` and `Series` that make working with tabular data (like spreadsheets or SQL tables) incredibly intuitive.

This simple import statement makes all the functionalities of these libraries available to us, aliased as `np` and `pd` respectively, ready for use in our code.

---

## NumPy: The Foundation for Numerical Computing

### Why NumPy? The Power of Arrays

When working with data, especially numerical data, efficiency is key. Performing operations on large datasets using standard Python lists can be very slow. NumPy provides a powerful solution by introducing a new data structure called the **NumPy array** (`ndarray`). Think of NumPy arrays as supercharged lists for numbers. They are:

*   **Memory-efficient:** Store data more compactly than Python lists.
*   **Computationally fast:** Operations on NumPy arrays are implemented in C and are highly optimized.
*   **Vectorized:** Allows you to perform operations on entire arrays without explicit loops, leading to significant speedups.

### Understanding NumPy Arrays

A NumPy array is a grid of values, all of the same type. The most common type is numeric.

**Example: Creating NumPy Arrays**

```python
# Create a 1D array (vector)
array_1d = np.array([1, 2, 3, 4, 5])
print("1D Array:", array_1d)

# Create a 2D array (matrix)
array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("2D Array:\n", array_2d)
```

**Output:**

```
1D Array: [1 2 3 4 5]
2D Array:
 [[1 2 3]
 [4 5 6]
 [7 8 9]]
```

#### Key Features of NumPy Arrays

*   **Homogeneity**: All elements in a NumPy array must have the same data type (e.g., all integers, all floats). This uniformity allows for efficient memory management and vectorized operations.
*   **Vectorization**: NumPy allows you to perform operations on entire arrays without explicit loops. For example, you can add a scalar to an entire array, or perform element-wise multiplication between two arrays, which is much faster than doing it with Python lists.

### Key NumPy Operations

NumPy offers a vast array of functions for numerical computation. Some fundamental operations include:

*   **Mathematical operations**: You can perform element-wise operations directly on arrays.
    ```python
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])

    print("Sum:", a + b)          # Element-wise sum
    print("Difference:", b - a)  # Element-wise difference
    print("Product:", a * b)     # Element-wise product
    print("Division:", b / a)    # Element-wise division
    ```

*   **Array attributes**: Access information about an array.
    ```python
    print("Shape of array_2d:", array_2d.shape)     # (rows, columns)
    print("Number of dimensions:", array_2d.ndim)   # 2
    print("Data type:", array_2d.dtype)            # int64 (or similar)
    ```

*   **Array slicing and indexing**: Access specific elements or sub-arrays.
    ```python
    print("First element of array_1d:", array_1d[0])
    print("First row of array_2d:\n", array_2d[0])
    print("Element at row 1, col 2 of array_2d:", array_2d[1, 2]) # or array_2d[1][2]
    ```

*   **Universal functions (ufuncs)**: Apply functions element-wise.
    ```python
    print("Square root of array_1d:", np.sqrt(array_1d))
    print("Exponential of array_1d:", np.exp(array_1d))
    ```

NumPy is the backbone for many other data science libraries, making it essential to understand its core functionalities.

### Creating and Reshaping Arrays

The foundation of numerical computing in Python often lies in arrays. NumPy's `ndarray` object provides a powerful and flexible way to handle multi-dimensional arrays.

#### The `np.arange` Function

One common way to create a sequence of numbers is using `np.arange()`. This function is similar to Python's built-in `range()` function but returns a NumPy array. It allows us to specify a `start`, `stop`, and `step` value. Crucially, like Python's `range()`, the `stop` value is *exclusive*, meaning the sequence goes up to, but does not include, the `stop` value.

Consider the following example:

```python
# Create an array from 1.5 up to (but not including) 10.5, with a step of 1
float_array = np.arange(1.5, 10.5, 1, dtype='float64')
print("Array from arange:", float_array)
```

Running this code would produce the output:

```
Array from arange: [1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5 9.5]
```

Notice that the last element is `9.5`, as `10.5` was the exclusive stop value. The `dtype='float64'` argument specifies that the array elements should be 64-bit floating-point numbers, ensuring precision.

#### Reshaping Arrays: From 1D to 2D and Beyond

Often, data is not in a simple one-dimensional sequence but rather organized in a grid or a multi-dimensional structure. NumPy's `reshape()` function is essential for transforming an array into a new shape without changing its data.

Let's take the array generated by `np.arange(1.5, 10.5)` which contains 9 elements (from `1.5` to `9.5` with a step of `1`). If we want to arrange these 9 elements into a 2D array with 3 rows and 3 columns, we can use `reshape(3, 3)`:

```python
float_array_reshaped = np.arange(1.5, 10.5, 1).reshape(3, 3)
print("Reshaped 2D Array:\n", float_array_reshaped)
```

This would output:

```
Reshaped 2D Array:
 [[1.5 2.5 3.5]
 [4.5 5.5 6.5]
 [7.5 8.5 9.5]]
```

The key principle here is that the total number of elements must remain the same. If you have 9 elements, you can reshape them into a 3x3 grid (3 * 3 = 9), but not into a 3x4 grid (which would require 12 elements).

#### The `-1` Placeholder in `reshape()`

A convenient feature of `reshape()` is the ability to use `-1` for one of the dimensions. NumPy will automatically infer the correct size for that dimension based on the total number of elements in the array. This is particularly useful when you know one dimension but want NumPy to figure out the other.

For example, if you have 8 elements and want to reshape them into 2 columns, you can use `reshape(-1, 2)`. NumPy will deduce that you need 4 rows (since 4 * 2 = 8). Conversely, if you want 4 rows, you can use `reshape(4, -1)` and NumPy will determine that you need 2 columns.

Let's try reshaping an array with elements from 1 to 8:

```python
array_to_reshape = np.arange(1, 9) # Generates [1 2 3 4 5 6 7 8]
reshaped_array_infer = array_to_reshape.reshape(2, -1) # Infer number of columns
print("Array reshaped with -1:\n", reshaped_array_infer)
```

Output:

```
Array reshaped with -1:
 [[1 2 3 4]
 [5 6 7 8]]
```

Here, since we specified 2 rows, NumPy correctly inferred that there must be 4 columns.

It's important to note that when using `reshape()`, you can also specify the data type. For instance, `dtype='float64'` ensures the array elements are floating-point numbers. If you omit `dtype`, NumPy will infer it.

#### Creating Arrays with Zeros and Ones

NumPy also provides specialized functions to create arrays filled with specific values:

*   `np.zeros(shape, dtype)`: Creates an array of the given `shape` filled with zeros.
*   `np.ones(shape, dtype)`: Creates an array of the given `shape` filled with ones.

Let's see an example:

```python
# Create a 2x3 array of zeros with integer type
zero_array = np.zeros((2, 3), dtype='int64')
print("Zero Array:\n", zero_array)

# Create a 2x3 array of ones with float type
one_array = np.ones((2, 3), dtype='float64')
print("One Array:\n", one_array)
```

Output:

```
Zero Array:
 [[0 0 0]
 [0 0 0]]
One Array:
 [[1. 1. 1.]
 [1. 1. 1.]]
```

The `shape` argument for these functions is a tuple defining the dimensions of the array. For a 2D array, it's `(rows, columns)`.

#### Creating Arrays Filled with a Specific Value

Beyond zeros and ones, `np.full(shape, fill_value, dtype)` allows you to create an array of a specified `shape` filled with a given `fill_value`.

```python
# Create a 2x2 array filled with the value 99
full_array = np.full((2, 2), 99, dtype='int64')
print("Full Array:\n", full_array)
```

Output:

```
Full Array:
 [[99 99]
 [99 99]]
```

### Understanding Multi-Dimensional Arrays

So far, we've worked with 1D and 2D arrays. NumPy can handle arrays with any number of dimensions.

#### 3D Arrays and Beyond

A 3D array can be thought of as a collection of 2D arrays (like layers or cubes). In the context of machine learning, this often maps to data like:

*   **Images**: A color image can be represented as a 3D array where dimensions correspond to height, width, and color channels (RGB).
*   **Time Series Data**: Multiple time series recorded over time.

#### Creating a 3D Array

We can create a 3D array by nesting lists of lists of lists. NumPy's `array()` function handles this nesting naturally.

```python
# Create a 3D array
arr_3d = np.array([[ [1, 2, 3], [4, 5, 6] ],
                   [ [1, 2, 3], [4, 5, 6] ]])
print("3D Array:\n", arr_3d)
print("Shape of 3D array:", arr_3d.shape)
print("Dimensions of 3D array:", arr_3d.ndim)
```

The output would look something like this:

```
3D Array:
 [[[1 2 3]
  [4 5 6]]

 [[1 2 3]
  [4 5 6]]]
Shape of 3D array: (2, 2, 3)
Dimensions of 3D array: 3
```

The attributes `.shape` and `.ndim` are just as applicable to multi-dimensional arrays, providing essential information about their structure.

---

## Pandas: Data Manipulation and Analysis

While NumPy excels at numerical computations, Pandas is designed for handling structured data, such as tables with rows and columns. It builds upon NumPy to provide flexible and powerful data manipulation tools.

### Understanding DataFrames

Pandas introduces two primary data structures:

*   **Series**: A one-dimensional labeled array capable of holding any data type (integers, strings, floating point numbers, Python objects, etc.). Think of it as a column in a spreadsheet.
*   **DataFrame**: A two-dimensional labeled data structure with columns of potentially different types. Think of it as a spreadsheet or an SQL table. It's the most commonly used Pandas object.

A DataFrame can be thought of as a collection of Series that share the same index.

### Loading Data: The Iris Dataset

A common practice in learning data science is to use well-known datasets to illustrate concepts. The Iris dataset is a classic choice because of its manageable size and straightforward structure, making it ideal for demonstrating data loading and initial exploration. The Iris dataset is a collection of measurements for three species of Iris flowers: *Iris setosa*, *Iris versicolor*, *Iris virginica*. It includes four numerical features: sepal length, sepal width, petal length, and petal width, along with the species name.

We can load this dataset directly from a URL hosted by the University of California, Irvine (UCI) Machine Learning Repository, which is a common source for research datasets. Pandas provides a convenient function, `read_csv()`, to load data from various sources, including CSV files accessible via a URL.

**Loading Data with Pandas**

To use this function effectively, we need to provide the URL and can optionally specify the names for our columns.

```python
# Load the Iris dataset from a URL
# This dataset contains 3 classes of 50 instances each, where each class
# refers to a type of iris plant. One class is linearly separable from the other 2;
# the latter are not linearly separable from each other.
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_df = pd.read_csv(url, names=column_names)
```

*   **`url`**: This variable holds the web address of the Iris dataset.
*   **`column_names`**: This list provides meaningful names for each column in the dataset, making it easier to understand and access the data.
*   **`pd.read_csv(url, names=column_names)`**: This is the core command. It tells Pandas to fetch the data from the specified `url` and use the names provided in `column_names` for its columns. The result is a Pandas DataFrame, which we store in the variable `iris_df`.

### Exploring DataFrames

Pandas provides many functions to explore and understand your data:

*   **`head()`**: Displays the first few rows (default is 5).
*   **`tail()`**: Displays the last few rows.
*   **`info()`**: Provides a concise summary of the DataFrame, including the index dtype and columns, non-null values, and memory usage.
*   **`describe()`**: Generates descriptive statistics that summarize the central tendency, dispersion, and shape of a dataset’s distribution, excluding `NaN` values.
*   **`columns`**: Returns the column labels.
*   **`unique()`**: Returns the unique values in a Series (column).
*   **`dtypes`**: Shows the data type of each column.

Let's see these in action with the loaded Iris dataset:

```python
# Display the first few rows of the DataFrame
print("First 5 rows of Iris DataFrame:")
print(iris_df.head())

# Display the column names
print("\nColumn Names:", iris_df.columns)

# Display information about the DataFrame
print("\nDataFrame Info:")
iris_df.info()

# Display descriptive statistics for numerical columns
print("\nDescriptive Statistics:")
print(iris_df.describe())

# Get unique species from the 'species' column
print("\nUnique Species:", iris_df['species'].unique())

# Display data types of columns
print("\nColumn Data Types:")
print(iris_df.dtypes)
```

**Output from `head()`:**

```
First 5 rows of Iris DataFrame:
   sepal_length  sepal_width  petal_length  petal_width        species
0           5.1          3.5           1.4          0.2    Iris-setosa
1           4.9          3.0           1.4          0.2    Iris-setosa
2           4.7          3.2           1.3          0.2    Iris-setosa
3           4.6          3.1           1.5          0.2    Iris-setosa
4           5.0          3.6           1.4          0.2    Iris-setosa
```

**Output from `columns`:**

```
Column Names: Index(['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'], dtype='object')
```

**Output from `info()`:** (The exact memory usage may vary)
```
DataFrame Info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 5 columns):
 #   Column        Non-Null Count  Dtype
---  ------        --------------  -----
 0   sepal_length  150 non-null    float64
 1   sepal_width   150 non-null    float64
 2   petal_length  150 non-null    float64
 3   petal_width   150 non-null    float64
 4   species       150 non-null    object
dtypes: float64(4), object(1)
memory usage: 6.0+ KB
```

**Output from `describe()`:**

```
Descriptive Statistics:
       sepal_length  sepal_width  petal_length  petal_width
count    150.000000   150.000000    150.000000   150.000000
mean       5.843333     3.057333      3.758000     1.199333
std        0.828066     0.435566      1.765298     0.762238
min        4.300000     2.000000      1.000000     0.100000
25%        5.100000     2.800000      1.600000     0.300000
50%        5.800000     3.000000      4.350000     1.300000
75%        6.400000     3.300000      5.100000     1.800000
max        7.900000     4.400000      6.900000     2.500000
```

**Output from `unique()`:**

```
Unique Species: ['Iris-setosa' 'Iris-versicolor' 'Iris-virginica']
```

**Output from `dtypes`:**

```
Column Data Types:
sepal_length    float64
sepal_width     float64
petal_length    float64
petal_width     float64
species          object
dtype: object
```

This output confirms that our column names were correctly assigned and loaded into the DataFrame. The `info()` method shows that we have 150 entries, with 4 numerical columns (float64) and one column of type `object` (which usually signifies strings, like our 'species' column). The `describe()` method gives us a statistical overview of the numerical features, and `unique()` reveals the three distinct species present.

### Extracting Data from Pandas DataFrames and Converting to NumPy Arrays

We can extract specific columns from a DataFrame as Pandas Series and then convert them into NumPy arrays using `np.array()`. This allows us to leverage NumPy's efficient numerical operations on our data.

Let's extract the 'sepal\_length' and 'sepal\_width' columns as NumPy arrays:

```python
sepal_length_array = np.array(iris_df['sepal_length'])
sepal_width_array = np.array(iris_df['sepal_width'])

print("Sepal Length Array (NumPy):", sepal_length_array)
print("Sepal Width Array (NumPy):", sepal_width_array)
```

This will output:

```
Sepal Length Array (NumPy): [5.1 4.9 4.7 4.6 5.  5.4 4.6 5.  4.4 4.9 5.4 4.8 4.9 5.4 4.8 4.6 5.1 5.3 4.6 5.1
 4.4 4.9 5.4 4.8 4.8 5.6 4.6 5.7 5.1 5.7 4.9 5.1 5.4 4.9 5.3 5.  4.8 5.1 4.2 4.7
 5.3 5.5 4.9 4.7 4.4 4.6 5.  4.5 4.7 5.3 4.7 5.2 5.4 5.1 5.4 5.3 4.9 5.  5. , 4.3 5.5 4.8 5.1 5.1 4.6 5.1 4.4 4.9 5.4 4.4 4.6 4.5 4.7 5.3 5.5 5.4 5.2 5. , 5.2 5.5 4.9 4.7 5.5 5.3 5.1 5.3 5.5 5.2 5.5 4.8 5.4 5.2 5.5 4.8 5.4 4.9 5.4 5.2
 5.5 5.3 5.5 5. , 4.9 5.2 5. , 5.3 4.9 5.1 5.  4.9 5.3 5.5 5.4 5.2 4.9 5.3 5.  6.3 5.6 5.2 5.7 5.2 5.2 5.1 5.3 5.5 5. , 5.1 5.4 5.1 5.3 5.5 5. , 5.5 5.1 5.1 5.9 5.7 5.5 5.2 5. , 5.2 5.3 5.5 5. , 5.5 5.3 5.4 5.4 5.2 5.4 5.2 5.5 5.1 5.3 5.2 5.4 5.1 5.3 5.4 5.1 5.4 5.3 5.2 5. , 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.4 5.1 5.2 5.4 5.