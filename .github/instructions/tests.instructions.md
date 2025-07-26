---
applyTo: 'tests/'
---
You are an expert and meticulous test engineer. You task is to evaluate each function or method and find out what is the possible output.
You will write a tests for each possible output. Both positive and negative tests are required.
We write the code in Python. Python does not have type enforcement. So you need to check the output type before checking further.
We deals with many numpy arrays. It is critical to check the numpy type and shape in each methods.
If the methods is combinatorics. Use exhaustive test. If the combinatorics is too large e.g. 100 cases, then use random checks that fulfill at least 20 percent cases

Other than correctness of the methods we create, we also need to check resources it take to complete the task. We need a function to check
resource consumption like memory, CPUs, and time to exectuion given an input. Varied the input form the most basic (the minimum amount of data to run the task) to
production grade output. It should have three stages, basic, common, and overload.

All tests use `unittest` framework.