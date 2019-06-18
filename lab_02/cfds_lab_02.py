
# coding: utf-8

# In[1]:


#-*- coding: utf-8 -*-


# <img align="right" style="max-width: 200px; height: auto" src="images/cfds_logo.png">
# 
# ###  Lab 02 - "Introduction to Python Programming"
# 
# Chartered Financial Data Scientist (CFDS), Spring Term 2019

# The lab environment of the **"Chartered Financial Data Scientist (CFDS)"** course is powered by Jupyter Notebooks (https://jupyter.org), which allow one to perform a great deal of data analysis and statistical validation. In this second lab we want to have a look at the basic data types, containers, decision structures, loops and functions of the Python programming language.

# The second lab builds in parts on the great Python tutorial of the Stanford University CS 231n lecture series developed by Andrej Karpathy, Justin Johnson and Fei-Fei Li. The original tutorial is available under the following url: http://cs231n.github.io/python-numpy-tutorial/. 
# 
# In case you experiencing any difficulties with the lab content or have any questions pls. don't hesitate to Marco Schreyer (marco.schreyer@unisg.ch).

# ### Lab Objectives:

# After today's lab you should be able to:
#     
# > 1. Understand the basic **data types** of the Python e.g. integer, boolean, string.
# > 2. Understand the basic **data containers** of Python e.g. lists and dictionaries.
# > 3. Know Python's **decision structures** to guide the worklow of a program.
# > 4. Understand how to **loop** over data containers in order to access and manipulate individual values.
# > 5. Implement small **functions** that allow for the execution of several Python statements.

# ### 1. Python Versions

# There are currently two different supported versions of Python, 2.x and 3.x. 
# 
# Somewhat confusingly, Python 3.x introduced many backwards-incompatible changes to the language, so code written for 2.x may not work under 3.x and vice versa. For this class all code will use Python 3.x (were x referes to an arbitrary version number).

# You my want to check your Python version at the command line by running python **--version**.

# ### 2. Basic Python Data Types

# There are four basic data types in the Python programming language:
# 
# > * **Integer's** - represent positive or negative whole numbers with no decimal point.
# > * **Float's** - represent positive or negative real numbers and are written with a decimal point.
# > * **String's** - represent sequences of unicode characters.
# > * **Boolean's** - represent constant objects that are either 'False' and 'True'. 

# #### 2.1 Numerical Data Type "integer"

# Numbers in Python are often called just **integers** or **'ints'**, and are positive or negative whole numbers with no decimal point. In Python 3, there is effectively no limit to how long an integer value can be. Of course, it is constrained by the amount of memory your system has, as are all things, but beyond that an integer can be as long as you need it to be:

# In[2]:


x = 3
print(x)


# Print the variable type:

# In[4]:


type(x)


# Basic mathematical operations:

# In[5]:


print(x + 1)   # addition
print(x - 1)   # subtraction
print(x * 2)   # multiplication
print(x ** 2)  # exponentiation 


# Basic mathematical operations (shortcuts):

# In[6]:


x += 1
x = x + 1
print(x)  # prints "4"
x *= 2
x = x * 2
print(x)  # prints "8"


# #### 2.2 Numerical Data Type "float"

# The **float** type in Python represent real numbers and are written with a decimal point dividing the integer and fractional parts. As a result float values are specified with a decimal point:

# In[7]:


y = 3.0
print(y)


# Print the variable type:

# In[8]:


type(y)


# Basic mathematical operations:

# In[9]:


print(y + 1)   # addition
print(y - 1)   # subtraction
print(y * 2)   # multiplication
print(y ** 2)  # exponentation


# Optionally, the character e or E followed by a positive or negative integer may be appended to specify scientific notation:

# In[10]:


z = 1e-7      # equals 0.0000001
print(z)
print(z + 1)
print(z * 2)
print(z ** 2)


# #### 2.3 Non-Numerical Data Type "string"

# Strings are sequences of character data. String literals are defined and delimited using either single or double quotes. All the characters between the opening delimiter and matching closing delimiter are part of the string:

# In[11]:


hello = 'hello'   # string literals can use single quotes
world = "world, my name is peterli"   # or double quotes; it does not matter.
print(hello)


# Print the variable type:

# In[12]:


type(hello)


# Print the length of each string in terms of number of characters:

# In[13]:


print(len(hello))
print(len(world))


# Concatenate two strings e.g. to form a sentence:

# In[14]:


hw = hello + '_' + world + ' 12' # string concatenation
print(hw)  # prints "hello world, my name is hubert 12"


# Concatenate two strings in C/C# notation (also allowed in Python):

# In[15]:


hw2 = ' %s %s %d' % (hello, world, 12)  # sprintf style string formatting
print(hw2)  # prints "hello world, my name is hubert 12"


# Concatenate two strings in Python3 notation:

# In[16]:


hw3 = '{} {} 12'.format(hello, world)
print(hw3) # prints "hello world, my name is hubert 12"


# String objects have a bunch of useful methods; for example:

# In[17]:


s = "hello"                     # init string variable
print(s.capitalize())           # capitalize a string; prints "Hello"
print(s.upper())                # convert a string to uppercase; prints "HELLO"
print(s.rjust(7))               # right-justify a string, padding with spaces; prints "  hello"
print(s.center(7))              # center a string, padding with spaces; prints " hello "
print(s.replace('l', '(ell)'))  # replace all instances of one substring with another;
                                # prints "he(ell)(ell)o"
print('  world '.strip())       # strip leading and trailing whitespace; prints "world"


# #### 2.4 Non-Numerical Data Type "boolean"

# Python 3 provides a Boolean data type. Objects of type boolean type may have one of two values, "True" or "False":

# In[18]:


a = True
b = False
print(a)
print(b)


# Print the variable type:

# In[19]:


type(a)


# Booleans are often used in Python to test conditions or constraints. For example a string in Python can be tested for truth value. The return type will be then a Boolean value (True or False). Let’s have a look at a few examples:

# In[20]:


s1 = 'Scary Halloween'
result_upper = s1.isupper()        # test if string contains only upper case characters
print(result_upper)


# Let's have a look at a couple more examples:

# In[21]:


s2 = 'SCARY HALLOWEEN'
result_upper = s2.isupper()        # test if string contains only upper case characters
print(result_upper)


# In[22]:


n1 = 10
result_greather_than = n1 > 100                 # test if 10 > 100
print(result_greather_than)


# In[23]:


n2 = 99
result_in_between = n2 > 10 and n2 < 100        # test if 99 > 10 and 99 < 100
print(result_in_between)


# We can even logically combine the tested conditions above:

# In[24]:


print(a and b)               # Logical AND; prints "False"
print(a or b)                # Logical OR; prints "True"
print(a and result_upper)    # Logical AND; prints "True"
print(not a)                 # Logical NOT; prints "False"
print(a != b)                # Logical XOR; prints "True"


# As you will see in upcoming labs, expressions in Python are often evaluated in Boolean context, meaning they are interpreted to represent truth or falsehood. A value that is true in Boolean context is sometimes said to be “truthy,” and one that is false in Boolean context is said to be “falsy”.

# ### 3. Basic Python Data Containers

# There are four collection data types in the Python programming language:
# 
# > * **List** - is a collection which is ordered and changeable. Allows duplicate members.
# > * **Tuple** - is a collection which is ordered and unchangeable. Allows duplicate members.
# > * **Set** - is a collection which is unordered and unindexed. No duplicate members.
# > * **Dictionary** - is a collection which is unordered, changeable and indexed. No duplicate members.
# 
# When choosing a collection type, it is useful to understand the properties of that type. Choosing the right type for a particular data set could mean retention of meaning, and, it could mean an increase in efficiency or security. 
# 
# During this lab we will have a closer look into **lists** and **dictionaries**.

# #### 3.1. Data Container "List"

# A list is a collection of basic Python data types "elemets" which is ordered and changeable. In Python lists are written with square brackets (equivalent to an array). Python lists allow duplicate elements, are resizeable and can contain elements of different data types. Lists can be used like this:

# In[66]:


awsome_list = [3, 1, 2]    # create a list
print(awsome_list)         # print list


# Print the variable type:

# In[26]:


type(awsome_list)


# Determine indvidual elements of a list:

# In[67]:


print(awsome_list[2])           # print third element of list created
print(awsome_list[-1])          # print the last list element


# Determine number of list elements:

# In[28]:


print(len(awsome_list))         # print the number of elements contained in the list


# Replace a list element by assigning a new value at a specific index:

# In[29]:


awsome_list[2] = 'happy'        # lists can contain elements of different types
print(awsome_list)


# Append element to the end of a list:

# In[30]:


awsome_list.append('coding')    # add a new element to the end of the list
print(awsome_list)


# Remove last element of a list:

# In[31]:


this_last = awsome_list.pop()   # remove and return the last element of the list
print(this_last)                # prints the element removed 
print(awsome_list)              # prints the remaining elements


# Create a list of numbers:

# In[68]:


list_of_numbers = list(range(len(awsome_list)))     # range is a built-in function that creates a list of integers
print(list_of_numbers)               # prints "[0, 1, 2, 3, 4]"


# Slice list using distinct indexing techniques:

# In[33]:


print(list_of_numbers[2:4])    # get a slice from index 2 to 4 (exclusive); prints "[2, 3]"
print(list_of_numbers[2:])     # get a slice from index 2 to the end; prints "[2, 3, 4]"
print(list_of_numbers[:2])     # get a slice from the start to index 2 (exclusive); prints "[0, 1]"
print(list_of_numbers[:])      # get a slice of the whole list; prints ["0, 1, 2, 3, 4]"
print(list_of_numbers[:-1])    # slice indices can be negative; prints ["0, 1, 2, 3]"


# Replace range of list elements:

# In[34]:


list_of_numbers[2:4] = [8, 9]  # assign a new sublist to a slice
print(list_of_numbers)         # prints "[0, 1, 8, 9, 4]"


# #### 3.2. Data Container "Dictionary"

# In Python dictionaries are used to stores (key, value) pairs. A dictionary is a collection of basic Python data types "elemets" which is unordered, changeable and indexed. In Python dictionaries are written with curly brackets. Dictionaries can be used like this:

# In[35]:


this_dictionary = {'cat': 'spooky', 'night': 'scary'}  # create a new dictionary
print(this_dictionary)


# Retrieve and print the value corresponding to the key "cat":

# In[36]:


print(this_dictionary['cat'])          # get an entry from a dictionary; prints "spooky"


# Add a new dictionary element:

# In[37]:


this_dictionary['pumpkin'] = 'ugly'    # set an entry in a dictionary
print(this_dictionary)


# Retrieve and print the value corresponding to the added entry with key "pumpkin":

# In[38]:


print(this_dictionary['pumpkin'])          # get an entry from a dictionary; prints "ugly"


# Retrieve and print all dictionary keys:

# In[39]:


keys = this_dictionary.keys()          # obtain all dictionary keys
print(keys)


# Retrieve and print all dictionary values:

# In[40]:


values = this_dictionary.values()      # obtain all dictionary values
print(values)


# Try to retrieve an dictionary value that is not contained in the dictionary (this will result in an error): 

# In[41]:


#print(this_dictionary['ghost'])                  # KeyError: 'ghost' not a key of the dictionary


# However, we can "catch" such erros using:  

# In[42]:


print(this_dictionary.get('ghost', 'N/A'))       # get an element with a default; prints "N/A"
print(this_dictionary.get('pumpkin', 'N/A'))     # get an element with a default; prints "ugly")


# Remove an element from a dictionary:

# In[43]:


del this_dictionary['pumpkin']                   # remove an element from a dictionary
print(this_dictionary)


# Try to retrieve the removed dictionary element:

# In[44]:


print(this_dictionary.get('pumpkin', 'N/A'))    # "pumpkin" is no longer a key; prints "N/A"


# ### 4. Basic Programming Structures

# As part of this lab we want to have closer look at three basic programming structures of Python:
# 
# > * **For-Loops** - used to iterate over a sequence of program satements.
# > * **Decision Structures** - used to anticipate conditions occurring while execution of a program.
# > * **Functions** - used to define a block of code which only runs when it is called.

# #### 4.1 Python Loop Structures

# To keep a program doing some useful work we need some kind of repetition, looping back over the same block of code again and again. Below we will describe the different kinds of loops available in Python.

# #### 4.1.1. The "For"-Loop and Lists

# The for loop that is used to iterate over elements of a sequence, it is often used when you have a piece of code which you want to repeat a specifc "n" number of time. The nature of a for-loop in Python is very simple  **"for all elements in a list, do this"**
# 
# Let's say that you have a list, you can then loop over the list elements using the `for` keyword like this:

# In[45]:


# list initialization
halloween_elements = ['cat', 'night', 'pumpkin', 100, 244]

# loop initialization and run
for anyname in halloween_elements: 
    print(anyname)


# **Note:** Python relies on the concept of indentation, using whitespace (or tabs), to define scope in the code. Other programming languages such as Java or C# often use brackets or curly-brackets for this purpose.

# Let's have look at another example of a for-loop:

# In[69]:


# init a list of numbers
numbers = [1, 10, 20, 30, 40, 50]

# init the result
result = 0

# loop initialization and run
for number in numbers:    
    
    result = result + number
    # result += number
    
# print the result
print(result)


# In order to loop over a list of numbers we can use Python's `range` function. The `range(lower_bound, upper_bound, step_size)` function generates a sequence of numbers, starting from the `lower_bound` to the `upper_bound`. The `lower_bound` and `step_size` parameters are optional. By default the lower bound is set to zero, the incremental step is set to one.

# In[47]:


# loop over range elements
for i in range(1, 10):
    
    # print current value of i
    print(i)


# To break out from a loop, you can use the keyword `break`. Let's have a look at the following example: 

# In[70]:


# loop over range elements
for i in range(1, 10000000000):
    
    # case: current value of i equals 3?
    if i == 3:
        
        # break: stop the loop
        break
        
    # print current value of i
    print(i)


# In contrast the `continue` keyword is used to tell Python to skip the rest of the statements in the current loop block and to continue to the next iteration of the loop.

# In[49]:


# loop over range elements
for i in range(1, 10):
    
    # case: current value of i equals 3?
    if i == 3:
        
        # continue: jump to next loop iteration 
        continue
    
    # print current value if i
    print(i)


# If you want access to the index of each element within the body of a loop, use the built-in enumerate function:

# In[50]:


halloween_elements = ['cat', 'night', 'pumpkin']
for idx, element in enumerate(halloween_elements):
    print('#%d: %s' % (idx + 1, element))


# When programming, frequently we want to transform one type of data into another. As a simple example, consider the following code that computes square numbers:

# In[51]:


nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print(squares)


# You can make this code simpler using a list comprehension:

# In[52]:


nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print(squares)


# List comprehensions can also contain conditions:

# In[53]:


nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print(even_squares)


# #### 4.1.2. The "For"-Loop and Dictionaries

# Similarly, it is easy to iterate over the keys in a dictionary:

# In[54]:


d = {'pumpkin': 0, 'cat': 4, 'spider': 8}
for animal in d:
    legs = d[animal]
    print('A %s has %d legs' % (animal, legs))


# If you want access to keys and their corresponding values, use the items method:

# In[55]:


d = {'pumpkin': 2, 'cat': 4, 'spider': 8}
for animal, legs in d.items():
    print('A %s has %d legs' % (animal, legs))


# ### Exercises:

# We recommend you to try the following exercises as part of the lab:
# 
# **1. Write a Python loop that multiplies all elements of a list with 10.**
# 
# > Write a Python loop that multiplies all elements of a list with 10. The input list is given by `range(0, 10)` and its output should result in: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90].

# In[71]:


my_list = range(1,10)
solution = []

for number in my_list:
    
    solution.append(number*10)
    
print(solution)


# **2. Write a Python loop that prints the numbers 0 to 10 backwards.**
# 
# > Write a Python loop that prints the numbers 0 to 10 backwards. The output of the loop should result in: 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0.

# In[72]:


my_list = range(10, -1, -1)
solution = []

for number in my_list:
    
    solution.append(number*1)
    
print(solution)


# In[74]:


this_list =[]
for i in range(0,11):
    this_list.append(i)
    
for i in this_list:
    if(i!=0):
        print(this_list[-i])


# #### 4.2 Python Decision Structures

# Decision structures evaluate multiple expressions which produce **True** or **False** as outcome. When solving a data science task you often need to determine which action to take and which statements to execute if an outcome is **True** or **False** otherwise. 
# 
# Let's briefly recap Python's use of logical or mathematical conditions:

# In[56]:


# init sample variables
a = 4
b = 7


# In[57]:


print(a == b)  # equals
print(a != b)  # not equals
print(a < b)   # less than
print(a <= b)  # less than or equal to 
print(a > b)   # greater than
print(a >= b)  # greater than or equal to


# The mathematical conditions outlined above can be used in several ways, most commonly in if-statements. An if-statement is written by using the `if` keyword. Let's have a look at an example:

# In[58]:


# init sample variables
a = 4
b = 7

# test condition
if b > a:
    print("b is greater than a")


# In the example above we used two variables, `a` and `b`, which are used as part of the if-statement to test whether `b` is greater than `a`. As a is 4, and b is 7, we know that 7 is greater than 4, and so we print that "b is greater than a".

# We can easily enhance the if-statement above by additional condidtions by the `elif` keyword. The `elif` keyword is pythons way of saying "if the previous conditions were not true, then try this condition":

# In[59]:


a = 4
b = 4

# test condition 1
if b > a:
  print("b is greater than a")

# test condition 2
elif a == b:
  print("a and b are equal")

elif a != b:
    print("test check and so on... ")


# Finally, we can use the `else` keyword to catch any case which isn't caught by the preceding conditions:

# In[60]:


a = 8
b = 4

# test condition 1
if b > a:
  print("b is greater than a")

# test condition 2
elif a == b:
  print("a and b are equal")

# all other cases
else:
  print("a is greater than b")


# In the example above the value assigned to variable `a` is greater than the value assigned to `b`, so the first `if` condition is not true, also the `elif` condition is not true, so we ultimatively go to the `else` condition and print that "a is greater than b".

# ### Exercises:

# We recommend you to try the following exercises as part of the lab:
# 
# **1. Write a Python decision structure that prints all the numbers from 0 to 6 except 3 and 6.**
# 
# > Write a Python decision structure that prints a number if it doesn't equal to 3 and 6. If the number equals 3 or 6 it should print 'forbidden number'.

# **2. Write a Python decision structure that evaluates if a number is a multiple of 5 and 7.**
# 
# > Write a Python decision structure that evaluates if number is a multiple of 5 and 7. Hint: You may want to use Python's percentage `%` operator as part of your case evaluation.

# Dictionary comprehensions: These are similar to list comprehensions, but allow you to easily construct dictionaries. For example:

# In[61]:


nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)


# #### 4.3 Python Functions 

# A function is a block of organized, reusable code that is used to perform a single, related action. Functions provide better modularity for your application and allow for high degree of reusable code. As you already saw, Python provides you with many built-in functions such as `print()`, etc. but you can also create your own functions. These functions are called user-defined functions.
# 
# A function is a block of code which only runs when it is called. You can pass data, known as parameters, into a function. A function can return data as a result. Python functions are defined using the `def` keyword.

# <img align="mid" style="width: 400px; height: auto" src="images/python-function.svg">

# (Source: https://swcarpentry.github.io/python-novice-inflammation/06-func/index.html.) 
# 
# Let's define our first function that takes a string as input parameter and prints it:

# In[62]:


# defines a printme function
def print_me(characters):
    
   # this prints a passed string into this function
   print(characters)


# Now, we can call our newly defined function using the function name followed by the arguments that we aim to pass in parenthesis:

# In[63]:


print_me(characters="I'm first call to user defined function!")
print_me(characters="Again second call to the same function")


# Isn't that fantastic? 

# Now that we understood the syntax to create customized functions we can create even more complex functions. Let's implement a function that determines if a given integer number is positive, negative or zero using a decision structure and prints the result accordingly:

# In[63]:


# defines a sign evaluation function
def sign(x):
    
    # case: positive value
    if x > 0:
        
        # return the string 'positive'
        return 'positive'
    
    # case: negative value
    elif x < 0:
        
        # return the string 'negative'
        return 'negative'
    
    # else: other value
    else:
        
        # return the string 'zero'
        return 'zero'


# Now we call our function and print the result of sign evaluation for distinct values:

# In[64]:


print(sign(x=-1))
print(sign(x=0))
print(sign(x=1))


# We will often define functions to take optional keyword arguments. An optional argument is an argument that assumes a default value if a value is not provided in the function call for that argument. The following example provides an idea on default arguments, it prints the characters given in upper case if not specified different, like this:

# In[65]:


def hello(characters, loud=1633):
    
    # case: default - loud print enabled
    if loud:
        print('HELLO, %s' % characters.upper())
        
    # case: non-loud print enabled
    else:
        print('Hello, %s!' % characters)


# In[66]:


hello(characters='Helloween', loud=1000)


# In[67]:


hello(characters='Helloween', loud=False)


# ### Exercises:

# We recommend you to try the following exercises as part of the lab:
# 
# **1. Write a Python function to calculate the length of a string.**
# 
# >Write a Python function named **"string_length"** to calculate the length of a string. The function should take an arbitrary string as an input and count the number of its characters. 
# 
# >Test your function accordingly using various string values and print the results, e.g., input: 'Halloween', expected result: 9.

# In[74]:





# **2. Write a Python program to get the largest number from a list.**
# 
# >Write a Python function named **"max_num_in_list"** to get the largest number from a list. The function should take an arbitrary list of integer values as an input and should return the integer that corresponds to the highest value. 
# 
# >Test your function accordingly using various string values and print the results, e.g., input: [1, 5, 8, 3], expected result: 8.

# In[75]:





# **3. Write a Python program to count the number of characters (character frequency) in a string.**
# 
# >Write a Python function named **"char_frequency"** to count the number of distinct characters occuring in it. The function should take an arbitrary string as an input and should return the count of occurance each individual character. 
# 
# >Test your function accordingly using various string values and print the results, e.g., input: 'Happy Halllllloweeeeeen!', expected result: {'a': 2, ' ': 1, 'e': 6, 'H': 2, 'l': 6, 'o': 1, 'n': 1, 'p': 2, '!': 1, 'w': 1, 'y': 1}

# In[79]:





# **Bonus: Write a Python function that takes a list of words and returns the one exhibiting the most characters.**
# 
# >Write a Python function named **find_longest_word** that takes a list of words and returns the length of the longest one. The function should take an arbitrary list of string values (words) as an input and should return the word that exhibits the most characters. 
# 
# >Test your function accordingly using various lists of string values and print the results, e.g., input: ['Happy', 'Halloween', '2018'], expected result: 'Halloween'.

# In[77]:





# ### [BONUS] 5. Basic Python Image Manipulations

# The Python Imaging Library (abbreviated as PIL, in newer versions known as Pillow) is a free library for the Python programming language that adds support for opening, manipulating, and saving many different image file formats. Using the PIL library in combination with the Numpy library allows for a great deal image analysis and manipulation:
# 
# Pls. note, that prior to using PIL you need to install the Python Image package. A brief installation instruction can be obtained from: https://pillow.readthedocs.io/en/5.3.x/installation.html.

# Upon successfully installation we can now import the PIL package:

# In[68]:


# use this statement to show plots inside your notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# import the pil and the matplotlib libraries
from PIL import Image 
import numpy as np
import matplotlib.pyplot as plt


# PIL supports a wide variety of image file formats. To read files from disk, use the open() function in the Image module. You don’t have to know the file format to open a file. The library automatically determines the format based on the contents of the file. Let's open and plot the 'halloween.jpg' image file:

# #### 5.1. Image Loading and Plotting

# In[69]:


image = np.asarray(Image.open("images/halloween.jpg"))
plot = plt.imshow(image, cmap=plt.cm.gray, vmin=0, vmax=255)


# Let's have a look at the exact shape of the image:

# In[70]:


h, w, c = image.shape

print(h)   # prints the image height
print(w)   # prints the image width
print(c)   # prints the number of image color channels = 3 for red, green, blue (rgb)


# #### 5.2. Image Cropping

# Use PIL to extract the upper pumpkin of the image:

# In[71]:


upper_pumpkin = image[20:250,20:300,:]
plot = plt.imshow(upper_pumpkin, cmap=plt.cm.gray, vmin=0, vmax=255)


# #### 5.3. Image Manipulation

# Greyscale the image by setting the third image channel to mean of all color values:

# In[72]:


gray = upper_pumpkin.mean(axis=2)
plot = plt.imshow(gray, cmap=plt.cm.gray, vmin=0, vmax=255)


# To learn more about PIL and its capabilites visit: https://pillow.readthedocs.io.

# ### Lab Summary:

# In this second lab, the basic data types and containers of the Python programming language are presented. The code and exercises presented in this lab may serves as a starting point for more complex and tailored analytics. 

# You may want to execute the content of your lab outside of the Jupyter notebook environment e.g. on a compute node or a server. The cell below converts the lab notebook into a standalone and executable python script. Pls. note, that in order to convert the notebook you need to install the Python **nbconvert** package available via: https://pypi.org/project/nbconvert/5.3.1/ prior to running the command below.

# In[73]:


get_ipython().system('jupyter nbconvert --to script cfds_lab_02.ipynb')

