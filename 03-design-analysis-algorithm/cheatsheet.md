## **Week 1: Introduction and Fundamentals**

This week covers the introduction to algorithms, fundamental problem types, essential data structures, and the basics of algorithm efficiency analysis[cite: 6].

### **Topic 1.1: Introduction - Fundamentals of Algorithmic Problem Solving**

**What is an Algorithm?**
An algorithm is a well-defined computational procedure that takes some value, or set of values, as input and produces some value, or set of values, as output. It is a sequence of unambiguous instructions for solving a problem, i.e., for obtaining a required output for any legitimate input in a finite amount of time.

**Why study algorithms?**
* **Correctness & Efficiency:** To ensure programs are correct and run efficiently, especially with large inputs or in resource-constrained environments.
* **Problem Solving:** Provides a framework for thinking about and solving computational problems.
* **Innovation:** Many technological advancements stem from new and improved algorithms (e.g., search engines, machine learning, bioinformatics).
* **Foundation of CS:** Algorithms are the core of computer science. Understanding them is crucial for any serious programmer or computer scientist.
* **Interview Preparation:** A significant part of technical interviews for software engineering roles.

**Fundamentals of Algorithmic Problem Solving Process:**
1.  **Understanding the Problem:**
    * Read the problem description carefully. Identify inputs, outputs, and constraints.
    * Work through small examples manually.
    * Clarify any ambiguities.
2.  **Deciding on Exact vs. Approximate Problem Solving:**
    * Can we find an exact solution?
    * Is an approximate solution acceptable or necessary (e.g., for NP-hard problems)?
3.  **Algorithm Design Techniques:**
    * Choose an appropriate strategy (e.g., brute force, divide and conquer, greedy, dynamic programming). This course will cover these techniques[cite: 4].
4.  **Designing an Algorithm and Data Structures:**
    * Develop a step-by-step procedure.
    * Choose appropriate data structures to store and manage data efficiently. The choice of data structures is crucial and often intertwined with algorithm design.
5.  **Proving Correctness:**
    * Demonstrate that the algorithm yields the required output for every legitimate input in a finite amount of time. This can be done through logical reasoning, mathematical induction, or loop invariants.
6.  **Analyzing an Algorithm:**
    * **Efficiency:** Estimate time complexity (how fast it runs) and space complexity (how much memory it uses), typically as a function of input size.
    * **Simplicity/Clarity:** Is the algorithm easy to understand and implement?
    * **Generality:** Does it solve a broad range of problems or just a specific case?
7.  **Coding and Testing the Algorithm:**
    * Implement the algorithm in a programming language.
    * Test with various inputs, including typical cases, edge cases, and large inputs, to ensure correctness and performance.

**Important Concepts:**
* **Input/Output:** Clearly defined.
* **Definiteness:** Each step must be precisely defined.
* **Finiteness:** The algorithm must terminate after a finite number of steps.
* **Effectiveness:** Each step must be basic enough to be carried out, in principle, by a person using only pencil and paper.
* **Correctness:** The algorithm should produce the correct output for all valid inputs.

**Use Cases/Real-World Examples:**
* **GPS Navigation:** Uses shortest path algorithms (like Dijkstra's or A*) to find the best route.
* **Search Engines:** Employ complex algorithms for indexing web pages and ranking search results.
* **Sorting Data:** Algorithms like Quicksort or Merge Sort are used in databases to sort records, on e-commerce sites to sort products, etc.

**Edge Cases to Consider in Problem Solving:**
* Empty input (e.g., an empty list, zero).
* Single-element input.
* Inputs that are already sorted or in reverse order (for sorting algorithms).
* Inputs that cause worst-case behavior.
* Inputs with duplicate values.
* Very large inputs (to test scalability).

---

**Exam-Style Questions (Topic 1.1):**

1.  **Conceptual:** What are the five main criteria that an algorithm must satisfy? Explain each briefly.
    * **Answer:**
        1.  **Input:** Zero or more quantities are externally supplied.
        2.  **Output:** At least one quantity is produced.
        3.  **Definiteness:** Each instruction is clear and unambiguous.
        4.  **Finiteness:** The algorithm terminates after a finite number of steps for all cases.
        5.  **Effectiveness:** Every instruction must be basic enough to be carried out, in principle, by a person using only pencil and paper.

2.  **Scenario-Based:** You are tasked with developing a feature to find the shortest driving route between two points in a city. Describe the algorithmic problem-solving steps you would take, from understanding the problem to having a working solution.
    * **Answer:**
        1.  **Understand the Problem:** Input: two locations (start, end), map data (roads, intersections, distances, speed limits, one-way streets, current traffic). Output: the shortest (or fastest) path.
        2.  **Exact vs. Approximate:** An exact shortest path is usually desired.
        3.  **Algorithm Design Technique:** Graph algorithms are suitable. Specifically, shortest path algorithms like Dijkstra's or A* (if heuristics like straight-line distance are available and admissible).
        4.  **Design Algorithm & Data Structures:** Represent the city map as a graph (intersections as nodes, roads as weighted edges). Use a priority queue for Dijkstra's or A*.
        5.  **Prove Correctness:** Standard proofs for Dijkstra's/A* demonstrate they find the shortest path in graphs with non-negative edge weights.
        6.  **Analyze Algorithm:** Analyze time and space complexity based on the number of intersections and roads.
        7.  **Code & Test:** Implement using a suitable language. Test with various city pairs, edge cases (e.g., disconnected locations, one-way streets creating no path), and compare with known good routes.

3.  **Practical Application:** Why is "finiteness" a crucial characteristic of an algorithm? What could happen if an algorithm is not finite?
    * **Answer:** Finiteness ensures that the algorithm will eventually terminate and produce a result. If an algorithm is not finite (i.e., it enters an infinite loop), it will never complete its task, consume resources indefinitely (CPU time, memory), and fail to provide an output, making it practically useless. For example, a web server request handled by a non-finite algorithm would cause the request to hang forever, potentially crashing the server or making it unresponsive.

---

### **Topic 1.2: Important Problem Types**

Understanding common problem types helps in categorizing new problems and recognizing existing algorithmic solutions or design patterns.

1.  **Sorting:**
    * **Description:** Rearrange items of a given list in non-decreasing (or non-increasing) order.
    * **Why important:** A fundamental problem, often a subroutine in other algorithms (e.g., searching, data compression, scheduling).
    * **Examples:** Sorting a list of numbers, names, student records by ID.
    * **Key Considerations:** Stability (preserving relative order of equal elements), in-place (minimal extra space).

2.  **Searching:**
    * **Description:** Find a given value (search key) in a given set of items, or determine that it's not present.
    * **Why important:** Essential for data retrieval.
    * **Examples:** Looking up a word in a dictionary, finding a specific file on a computer, checking if a user exists in a database.
    * **Key Considerations:** Static vs. dynamic sets, efficiency based on data organization (e.g., sorted array, hash table).

3.  **String Processing:**
    * **Description:** Problems involving sequences of characters (strings).
    * **Why important:** Text manipulation is ubiquitous (word processors, web search, bioinformatics).
    * **Examples:** String matching (finding a pattern in a text), sequence alignment (bioinformatics), text compression.
    * **Key Considerations:** Character sets (ASCII, Unicode), efficient pattern matching algorithms.

4.  **Graph Problems:**
    * **Description:** Problems modeled using graphs (a collection of vertices and edges).
    * **Why important:** Model many real-world relationships and networks.
    * **Examples:**
        * Shortest path (GPS navigation, network routing).
        * Traveling Salesman Problem (TSP): Find the shortest tour visiting all cities.
        * Graph traversal (web crawling, social network analysis).
        * Topological sorting (task scheduling).
    * **Key Considerations:** Directed vs. undirected graphs, weighted vs. unweighted edges, sparse vs. dense graphs.

5.  **Combinatorial Problems:**
    * **Description:** Find a combinatorial object (e.g., a permutation, combination, or subset) that satisfies certain constraints and has some desired property (e.g., maximizes a value).
    * **Why important:** Often very hard (NP-hard), requiring clever algorithms or approximations.
    * **Examples:** TSP, knapsack problem (select items with maximum value given weight constraint), scheduling problems.
    * **Key Considerations:** Feasibility, optimality, dealing with large search spaces.

6.  **Geometric Problems:**
    * **Description:** Problems involving geometric objects (points, lines, polygons).
    * **Why important:** Computer graphics, robotics, geographic information systems (GIS).
    * **Examples:** Closest-pair problem (find two closest points), convex hull problem (find the smallest convex polygon enclosing a set of points).
    * **Key Considerations:** Representing geometric objects, handling floating-point precision.

7.  **Numerical Problems:**
    * **Description:** Problems involving mathematical objects like equations, integrals, functions. Often require continuous mathematics.
    * **Why important:** Scientific computing, engineering, finance.
    * **Examples:** Solving systems of linear equations, computing definite integrals, finding roots of equations.
    * **Key Considerations:** Accuracy of results, stability of algorithms, computational cost.

**Why categorize problems?**
* **Recognition:** If a new problem fits a known type, existing algorithms or techniques might be applicable.
* **Difficulty:** Certain problem types are known to be harder than others (e.g., many combinatorial problems are NP-hard).
* **Algorithm Selection:** The type of problem influences the choice of algorithm design strategy.

---

**Exam-Style Questions (Topic 1.2):**

1.  **Conceptual:** Explain the difference between a sorting problem and a searching problem. Why are both fundamental in computer science?
    * **Answer:**
        * **Sorting:** Arranges a collection of items in a specific order (e.g., ascending or descending). The output is the entire collection, reordered.
        * **Searching:** Finds the location or presence of a specific item (key) within a collection. The output is typically the item's position or a boolean indicating its presence.
        * Both are fundamental because sorting often facilitates faster searching and is a common preprocessing step. Searching is essential for information retrieval, a core task in most applications.

2.  **Scenario-Based:** A logistics company wants to optimize its delivery routes. Each day, a driver has a list of locations to visit, starting and ending at the depot. The goal is to minimize the total distance traveled. What type of algorithmic problem does this represent? Name a specific famous problem of this type.
    * **Answer:** This represents a **graph problem** or more specifically, a **combinatorial problem** on a graph. The famous problem of this type is the **Traveling Salesman Problem (TSP)**.

3.  **Practical Application:** You are designing a system for a library. List at least three different "important problem types" that you might encounter and provide a specific example for each in the library context.
    * **Answer:**
        1.  **Searching Problem:** Finding a specific book by its title, author, or ISBN. Example: A user types "Introduction to Algorithms" in the search bar, and the system needs to find all matching books.
        2.  **Sorting Problem:** Displaying search results for books sorted by relevance, publication date, or author's last name. Example: After searching for books on "Data Structures," the results are shown sorted by the newest publication date first.
        3.  **String Processing Problem:** Implementing a feature that suggests corrections for misspelled author names or book titles during a search. Example: If a user types "Algoithms" the system might suggest "Algorithms" (related to pattern matching or edit distance).
        * (Bonus) **Graph Problem (less common but possible):** Analyzing citation networks between academic papers in the library's digital collection to find influential papers (nodes are papers, edges are citations).

---

### **Topic 1.3: Fundamental Data Structures**

(As per the outline, this refers to Cormen/Chapter 10 and Levitin/Chapter 1[cite: 6]. Chapter 10 of Cormen (3rd ed.) covers elementary data structures.)

Data structures are ways of organizing and storing data to perform operations on them efficiently. The choice of data structure is a critical part of algorithm design.

1.  **Arrays:**
    * **Description:** A sequence of items of the same type stored in contiguous memory locations. Accessed by an index.
    * **Operations:** Access (O(1)), Search (O(n) unsorted, O(log n) sorted), Insertion/Deletion (O(n) in general, as it may require shifting elements).
    * **Pros:** Fast random access. Simple.
    * **Cons:** Fixed size (in many languages, or requires resizing which can be costly). Insertions/deletions can be slow.
    * **Use Cases:** Storing lists of items where size is known or access speed is critical (e.g., lookup tables).
    * **Real-World Example:** A list of scores for students in a class.

2.  **Linked Lists:**
    * **Description:** A sequence of nodes, where each node contains data and a pointer (or link) to the next node. Singly linked (next pointer) vs. Doubly linked (next and previous pointers).
    * **Operations:** Insertion/Deletion (O(1) if at known location/ends, O(n) to find location), Access/Search (O(n)).
    * **Pros:** Dynamic size. Efficient insertions/deletions at any point once the position is known.
    * **Cons:** Slow random access (requires traversal). Requires extra space for pointers.
    * **Use Cases:** Implementing stacks, queues, managing dynamic lists where insertions/deletions are frequent (e.g., a playlist manager).
    * **Real-World Example:** The "undo" functionality in a text editor might use a list-like structure to keep track of actions.

3.  **Stacks:**
    * **Description:** A Last-In, First-Out (LIFO) data structure. Operations are Push (add to top) and Pop (remove from top).
    * **Implementation:** Can be implemented using arrays or linked lists.
    * **Operations (Typical):** Push (O(1)), Pop (O(1)), Peek/Top (O(1)).
    * **Pros:** Simple, efficient for LIFO access pattern.
    * **Use Cases:** Function call management (call stack), expression evaluation (infix to postfix), backtracking algorithms.
    * **Real-World Example:** The call stack that manages function calls and local variables in most programming languages.

4.  **Queues:**
    * **Description:** A First-In, First-Out (FIFO) data structure. Operations are Enqueue (add to rear) and Dequeue (remove from front).
    * **Implementation:** Can be implemented using arrays (circular queue) or linked lists.
    * **Operations (Typical):** Enqueue (O(1)), Dequeue (O(1)), Peek/Front (O(1)).
    * **Pros:** Simple, efficient for FIFO access pattern.
    * **Use Cases:** Managing requests (e.g., print queue, server requests), breadth-first search in graphs, task scheduling.
    * **Real-World Example:** A printer queue where print jobs are processed in the order they are received.

5.  **Trees:**
    * **Description:** A hierarchical data structure consisting of nodes connected by edges. Each tree has a root node, and every node (except the root) has exactly one parent.
    * **Types:** Binary Trees, Binary Search Trees (BSTs), Balanced Trees (AVL, Red-Black), Heaps, B-Trees, etc.
    * **Operations (vary by type):** Insertion, Deletion, Search, Traversal (pre-order, in-order, post-order, level-order).
    * **Pros:** Efficient for hierarchical data, searching (in BSTs/balanced trees), sorting (heaps).
    * **Cons:** Can be complex to implement and maintain balance (for BSTs).
    * **Use Cases:** File systems, organization charts, syntax trees in compilers, search applications (BSTs), priority queues (Heaps).
    * **Real-World Example:** The directory structure of a file system.

6.  **Graphs:**
    * **Description:** A set of vertices (nodes) and a set of edges connecting pairs of vertices. Can be directed/undirected, weighted/unweighted.
    * **Representations:** Adjacency Matrix, Adjacency List.
    * **Operations:** Add/Remove vertex/edge, Traversal (BFS, DFS), Find path, Check connectivity.
    * **Pros:** Model complex relationships and networks.
    * **Cons:** Some graph algorithms can be complex and computationally intensive.
    * **Use Cases:** Social networks, road networks, web page linking, network analysis, circuit design.
    * **Real-World Example:** A social network like Facebook, where users are vertices and friendships are edges.

7.  **Hash Tables (Dictionaries/Maps):**
    * **Description:** A data structure that maps keys to values for highly efficient lookup. Uses a hash function to compute an index into an array of buckets or slots.
    * **Operations:** Insert (Average O(1)), Delete (Average O(1)), Search (Average O(1)). Worst case O(n) with many collisions.
    * **Pros:** Extremely fast average-case lookup, insertion, and deletion.
    * **Cons:** Worst-case performance can be bad (due to collisions). Order is not generally preserved. Requires a good hash function.
    * **Use Cases:** Implementing associative arrays, database indexing, caching, symbol tables in compilers.
    * **Real-World Example:** A phone book where you can quickly look up a phone number (value) using a person's name (key).

8.  **Heaps (Priority Queues):**
    * **Description:** A specialized tree-based data structure that satisfies the heap property (e.g., in a min-heap, for any given node C, if P is a parent node of C, then the key of P is less than or equal to the key of C). Often used to implement priority queues.
    * **Types:** Min-heap, Max-heap.
    * **Operations:** Insert (O(log n)), Delete-Min/Max (O(log n)), Get-Min/Max (O(1)).
    * **Pros:** Efficiently supports finding and removing min/max element, good for priority management.
    * **Use Cases:** Implementing priority queues (e.g., in task scheduling, Dijkstra's algorithm, Prim's algorithm), heapsort.
    * **Real-World Example:** In an operating system, managing tasks with different priorities. Tasks with higher priority are processed before tasks with lower priority.

**Why are data structures fundamental?**
The choice of data structure can significantly impact an algorithm's efficiency (time and space). An inappropriate choice can lead to very poor performance. Understanding their properties, strengths, and weaknesses is essential for designing effective algorithms.

---

**Exam-Style Questions (Topic 1.3):**

1.  **Conceptual:** Explain the primary difference in performance characteristics between an array and a linked list for (a) accessing an element at a specific index, and (b) inserting an element at the beginning of the list.
    * **Answer:**
        * **(a) Accessing an element:**
            * **Array:** O(1) (constant time) because elements are in contiguous memory, and the address can be calculated directly from the index.
            * **Linked List:** O(n) (linear time) in the worst case, as it requires traversing the list from the head to reach the desired index.
        * **(b) Inserting an element at the beginning:**
            * **Array:** O(n) because all existing elements need to be shifted one position to make space for the new element.
            * **Linked List:** O(1) because it only involves updating a few pointers (creating a new node and adjusting the head pointer).

2.  **Scenario-Based:** You are designing a system to manage a waiting list for a popular restaurant. Customers are added to the list and seated in the order they arrive. Which fundamental data structure would be most appropriate for this, and why? What are the key operations you would need?
    * **Answer:**
        * **Data Structure:** A **Queue** would be most appropriate.
        * **Why:** A queue follows the First-In, First-Out (FIFO) principle, which matches the requirement that customers are seated in the order they arrive.
        * **Key Operations:**
            * `Enqueue`: To add a customer to the end of the waiting list.
            * `Dequeue`: To remove a customer from the front of the list when a table is ready.
            * `Peek` (or `Front`): To see who is next without removing them.
            * `IsEmpty`: To check if the waiting list is empty.

3.  **Practical Application:** When would you prefer to use a Hash Table over a Binary Search Tree (BST) for storing and retrieving data? What is a potential downside of Hash Tables you must consider?
    * **Answer:**
        * **Preference for Hash Table:** You would prefer a Hash Table when average-case O(1) time complexity for search, insertion, and deletion is paramount, and the order of elements does not need to be maintained. For example, implementing a cache where quick lookups are critical.
        * **Potential Downside of Hash Tables:** The main downside is that in the worst-case (e.g., many collisions, poor hash function), the performance can degrade to O(n) for these operations. Also, hash tables do not inherently maintain data in a sorted order, which BSTs can do (via in-order traversal).

---

### **Topic 1.4: Fundamentals of the Analysis of Algorithm Efficiency**

(As per the outline, this refers to Cormen/Chapter 3[cite: 6]. Chapter 3 of Cormen (3rd ed.) covers Growth of Functions and Asymptotic Notations.)

Algorithm analysis is about predicting the resources an algorithm requires, primarily:
* **Time Complexity:** How much time an algorithm takes to run as a function of its input size.
* **Space Complexity:** How much memory an algorithm uses as a function of its input size.

**Why Analyze Efficiency?**
* **Performance Comparison:** To compare different algorithms for the same problem.
* **Predictability:** To predict performance for larger inputs.
* **Optimization:** To identify bottlenecks and areas for improvement.
* **Feasibility:** To determine if an algorithm is practical for a given problem size and hardware.

**Key Concepts:**

1.  **Input Size (n):**
    * The primary parameter used to measure efficiency.
    * Examples: Number of items in an array to be sorted, number of nodes in a graph, number of bits in a numerical input.

2.  **Running Time:**
    * The number of primitive operations (basic computations like assignments, comparisons, arithmetic operations) executed by the algorithm.
    * We assume each primitive operation takes a constant amount of time.

3.  **Orders of Growth (Growth Rates):**
    * We are most interested in how the running time grows as the input size `n` becomes very large.
    * Focus on the dominant term in the running time function, ignoring constant factors and lower-order terms. For example, if `T(n) = 3n^2 + 10n + 5`, we say its order of growth is `n^2`.

4.  **Cases for Analysis:**
    * **Worst-Case Analysis (Most Common):**
        * The longest running time for *any* input of size `n`.
        * Provides an upper bound on performance. Often the most crucial because it guarantees performance.
        * Example: Searching for an item not present in an unsorted array, or present only at the very end.
    * **Average-Case Analysis:**
        * The expected running time over all possible inputs of size `n`.
        * Requires assumptions about the statistical distribution of inputs.
        * Can be more realistic but is often harder to compute.
        * Example: Average number of comparisons in Quicksort.
    * **Best-Case Analysis:**
        * The shortest running time for *any* input of size `n`.
        * Provides a lower bound, but not very useful for overall performance guarantee as best cases might be rare.
        * Example: Searching for an item at the very beginning of an array.

**Asymptotic Notations (Introduced more formally in Week 2 topics[cite: 6], but fundamental here):**
These are mathematical tools to describe the limiting behavior of functions, used to classify algorithms by their growth rates.
* **O-notation (Big O):** Upper bound. `f(n) = O(g(n))` means `f(n)` grows no faster than `g(n)`. Used for worst-case complexity.
* **$\Omega$-notation (Big Omega):** Lower bound. `f(n) = \Omega(g(n))` means `f(n)` grows at least as fast as `g(n)`. Used for best-case complexity or lower bounds of problems.
* **$\Theta$-notation (Big Theta):** Tight bound. `f(n) = \Theta(g(n))` means `f(n)` grows at the same rate as `g(n)`. Used when best and worst cases have the same growth rate.

**Example: Analyzing a Simple Loop**
```
sum = 0
for i = 1 to n:      // Loop runs n times
    sum = sum + i    // Constant time operation (c1)
return sum
```
* The operation inside the loop takes constant time, say `c1`.
* The loop executes `n` times.
* Total time `T(n) = c1 * n`.
* This algorithm has a time complexity of `O(n)` (linear growth).

**Space Complexity:**
* **Auxiliary Space:** The extra space or temporary space used by the algorithm, excluding input space.
* **Total Space:** Auxiliary space + space used by input.
* Often analyzed similarly to time complexity (e.g., `O(1)` for constant space, `O(n)` for linear space).

**Why Asymptotic Analysis?**
* **Abstracts away machine-dependent constants:** Focuses on the intrinsic efficiency related to the input size.
* **Simplifies analysis:** Ignores lower-order terms and constant factors, making it easier to compare algorithms fundamentally.
* **Focuses on scalability:** Crucial for understanding how algorithms perform as data grows.

---

**Exam-Style Questions (Topic 1.4):**

1.  **Conceptual:** What is the difference between worst-case and average-case analysis of an algorithm? Why is worst-case analysis generally preferred?
    * **Answer:**
        * **Worst-Case Analysis:** Determines the maximum running time of an algorithm for any possible input of a given size `n`. It provides an upper bound on performance.
        * **Average-Case Analysis:** Determines the expected running time of an algorithm averaged over all possible inputs of size `n`, assuming a certain probability distribution of inputs.
        * **Preference for Worst-Case:** Worst-case analysis is generally preferred because:
            1.  It provides a guarantee: the algorithm will never take longer than this bound. This is crucial for critical applications.
            2.  Average-case analysis can be complex as it requires knowing or assuming the distribution of inputs, which might not be realistic or easy to determine.
            3.  For some algorithms, the worst-case occurs fairly often.

2.  **Scenario-Based:** Consider an algorithm with a time complexity function `T(n) = 5n^3 + 100n^2 + 200n + 1000`. What is its asymptotic time complexity using Big O notation? Explain why we can simplify the function to this notation.
    * **Answer:**
        * The asymptotic time complexity is `O(n^3)`.
        * **Explanation:** In asymptotic analysis, we are interested in the growth rate for large values of `n`.
            1.  **Dominant Term:** The term `5n^3` grows much faster than `100n^2`, `200n`, and `1000` as `n` becomes large. So, we focus on the `n^3` term.
            2.  **Ignore Constant Coefficients:** The constant factor `5` does not fundamentally change the growth rate relative to `n^3`. Whether it's `5n^3` or `n^3`, the algorithm's time still grows cubically with `n`.
            3.  **Ignore Lower-Order Terms:** `100n^2`, `200n`, and `1000` become insignificant compared to `5n^3` for large `n`.
            Thus, we simplify `T(n)` to its highest order term without the constant coefficient, resulting in `O(n^3)`.

3.  **Practical Application:** You have two algorithms to solve the same problem. Algorithm A has a worst-case time complexity of `O(n log n)` and Algorithm B has a worst-case time complexity of `O(n^2)`.
    * (a) Which algorithm would you generally prefer for large input sizes and why?
    * (b) Can Algorithm B ever be faster than Algorithm A? If so, under what circumstances?
    * **Answer:**
        * **(a) Preference for Large Inputs:** For large input sizes (`n`), Algorithm A with `O(n log n)` complexity would generally be preferred. This is because `n log n` grows slower than `n^2`. As `n` increases, the difference in performance will become more significant, with Algorithm A being much faster.
        * **(b) Algorithm B Faster?:** Yes, Algorithm B (`O(n^2)`) *could* be faster than Algorithm A (`O(n log n)`) under certain circumstances:
            1.  **Small Input Sizes (n):** Asymptotic notation describes behavior for large `n`. For small `n`, the constant factors and lower-order terms hidden by Big O notation can dominate. Algorithm B might have smaller constant factors, making it faster for small `n`. For example, if Algorithm A is `100 * n log n` and Algorithm B is `1 * n^2`, for very small `n`, `n^2` might be smaller.
            2.  **Specific Inputs (Not Worst-Case for B):** If Algorithm B has a much better best-case or average-case performance than `O(n^2)` for certain types of inputs, and those inputs are common, while Algorithm A consistently performs at `O(n log n)`. However, the question specifies worst-case, making this less relevant unless comparing actual runtimes not just worst-case bounds.
            3.  **Simpler Implementation:** Sometimes an algorithm with a theoretically worse complexity (like `O(n^2)` Bubble Sort) might be implemented so simply with fewer overheads that it outperforms a more complex `O(n log n)` algorithm for very small, fixed-size inputs.
