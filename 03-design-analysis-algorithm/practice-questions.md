DAA

### Section A: MCQs
1. What is the worst case time complexity of Bubble sort?
    - $O(n)$
    - $O(n log n)$
    - $O(n^2)$
    - $O(1)$
2. Which of the following is NOT an example of a brute force algorithm?
    - Bubble Sort
    - Selection Sort
    - Merge Sort
    - Sequential Search
3. Which of the following is not a fundamental data structure?
    - Array
    - Linked List
    - Algorithm
    - Stack
4. Which of the following is an example of a decrease by a constant factor algorithm?
    - Binary Search
    - Quick Sort
    - Merge Sort
    - Bubble Sort
5. Which of the following sorting algorithms has the best average case time complexity?
    - Quick Sort
    - Merge Sort
    - Selection Sort
    - Bubble Sort
6. Which problem can be solved using exhaustive search?
    - Sorting an array
    - Find the shortest path in a graph
    - The traveling salesman problem
    - Matrix Multiplication
7. Which notation is used to represent the best-case time complexity of an algorithm?
    - Theta
    - Big-O
    - Big-Omega
    - None of the above
8. Insertion sort follows which algorithmic strategy?
    - Brute Force
    - Divide and Conquer
    - Decrease and Conquer
    - Transform and Conquer
9. What is the recurrence relation for Merge Sort?
    - $T(n) = 2T(n/2) + O(n)$
    - $T(n) = T(n-1) + O(1)$
    - $T(n) = T(n/2) + O(n)$
    - $T(n) = 3T(n/2) + O(n^2)$
10. Which of the following algorithms uses the Divide and Conquer approach?
    - Quick Sort5
    - Merge Sort
    - Matrix Multiplication
    - All of the above
11. What does term `asymptotic notation` primarily describe?
    - The exact running time of an algorithm
    - The growth rate of an algorithm's running time
    - The space complexity of an algorithm
    - The best-case scenario of an algorithm
12. What is the time complexity of the selection sort algorithm in the average case?
    - $O(n)$
    - $O(n log n)$
    - $O(n^2)$
    - $O(log n)$
13. What is the main idea behind the `decrease and conquer` strategy?
    - Breaking the problem into smaller sub problems
    - Reducing the problem size by a constant factor
    - Solving the problem by transforming it into another problem
    - Solving the problem by exhaustive search
14. Which of the following is NOT a type of asymptotic?
    - Big-O
    - Big-Theta
    - Big-Omega
    - Big-Sigma
15. Which of the following is NOT a fundamental problem type in algorithm design?
    - Sorting
    - Searching
    - Optimization
    - Debugging
16. Which of the following is NOT a step in the analysis framework for iterative algorithms?
    - Identify the basis operations
    - Determine the number of iterations
    - Solve the recurrence relation
    - Calculate the total running time
   

### Section B: True/False
1. The brute force approach always guarantees the most efficient solution
2. Asymptotic notation is used to describe the best case running time of an algorithm
3. The master theorem can be applied to any recurrence relation
4. Binary search is an example of a decrease and conquer algorithm
5. The time complexity of bubble sort in the worst case is $O(n)$
6. The travelling salesman problem can be solved in polynomial time using a brute force.
7. Divide and conquer algorithms have a time complexity of $O(n log n)$
8. The knapsack problem is an example of an optimization problem
9. The space complexity of an algorithm is always less important than its time complexity
10. The analysis framework for recursive algorithm involves solving recurrence relations
11. The time complexity of merge sort in the worst case is $O(n^2)$
12. Binary search can be applied to an unsorted array
13. Brute force algorithms are usually more efficient than divide and conquer algorithms
14. The master theorem provides an efficient way to solve recurrence relations
15. Greedy algorithms always provide the optimal solution to every problem.

### Section C: Puzzle Type Questions
1. You have 8 identical looking coins, but one is slightly heavier. You have a balance scale. What is the minimum number of weightings needed to find the heavier coin using the Divide and Conquer Approach.

2. I am an algorithm that works by repeatedly breaking a problem into two halves until it becomes simple enough to solve directly. What am I?

3. I am a soring algorithm that picks a pivot and partitions the array around it. What am I?

4. You have 5 numbers: {3, 1, 4, 2, 5}. Using the Insertion sort approach, what will the sequence after the first three steps?

5. Using the Brute Force method, how many permutation are there for arranging 4 distinct objects?

6. You are given an array of integers. Write the steps for the selection sort algorithm to sort the array in ascending order.

7. Given the following array: {5, 3, 8, 4, 2}, perform a step-by-step insertion sort and show the each iteration.

8. You have a sorted array of 1,000,000 elements. You need to find a number in the array. Which algorithm will you use for the fastest search? Why?

9. What is the time complexity of the following code snippet? Justify your answer?
	```py
	for i in range(n):
		for j in range(1,n):
			print(i, j)
	```


------------------------------------------------------------------------------------------------------------
Which of the following problem can be solved using exhaustive search?
Sorting an array
Finding the shortest path
The Traveling Salesman Problem
Matrix Multiplication

ChatGPT said:
The Traveling Salesman Problem (TSP) can be solved using exhaustive search (also known as brute force). Here's an explanation for each option:

Sorting an array:

Exhaustive search is not used to solve sorting problems. Sorting algorithms like Quick Sort, Merge Sort, and Bubble Sort are typically used, which are more efficient than an exhaustive approach.

Finding the shortest path:

Exhaustive search could be used to find the shortest path, but more efficient algorithms like Dijkstra's algorithm, Bellman-Ford, or A* search are usually preferred for this problem. Exhaustive search (like trying all possible paths) would be inefficient, especially for large graphs.

The Traveling Salesman Problem:

Exhaustive search can solve the Traveling Salesman Problem by trying all possible permutations of the cities and calculating the total travel distance for each. This is a brute-force approach, but it is computationally expensive, with a time complexity of O(n!). While not efficient for large instances, it is a valid approach for solving the TSP.

Matrix Multiplication:

Exhaustive search is not a typical method for matrix multiplication. Matrix multiplication is usually done using efficient algorithms like the standard matrix multiplication algorithm (O(n³)) or Strassen's algorithm, which reduce the computational complexity compared to an exhaustive search.

Conclusion:
The Traveling Salesman Problem can be solved using exhaustive search (brute force), though it's not efficient for large datasets. The other problems can be solved using more optimized approaches.
