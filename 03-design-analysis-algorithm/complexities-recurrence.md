
### **Validated Table of Recurrence Relations**

| **Algorithm**              | **Recurrence Relation**                              | **Best Case**          | **Average Case**       | **Worst Case**         | **Notes** |
|----------------------------|------------------------------------------------------|------------------------|------------------------|------------------------|-----------|
| **Linear Search**           | $T(n) = T(n-1) + O(1)$                          | $O(1)$             | $O(n)$             | $O(n)$             | Found on first try (best case). |
| **Binary Search**           | $T(n) = T(n/2) + O(1)$                          | $O(1)$             | $O(\log n)$        | $O(\log n)$        | Assumes sorted input. |
| **Bubble Sort**             | $T(n) = T(n-1) + O(n)$                          | $O(n)$             | $O(n^2)$           | $O(n^2)$           | Best case with early termination on sorted input. |
| **Selection Sort**          | $T(n) = T(n-1) + O(n)$                          | $O(n^2)$           | $O(n^2)$           | $O(n^2)$           | No best-case optimization. |
| **Insertion Sort**          | $T(n) = T(n-1) + O(n)$                          | $O(n)$             | $O(n^2)$           | $O(n^2)$           | Best case for already sorted input. |
| **Merge Sort**              | $T(n) = 2T(n/2) + O(n)$                         | $O(n \log n)$      | $O(n \log n)$      | $O(n \log n)$      | Always divides perfectly. |
| **Quick Sort**              | $T(n) = T(k) + T(n-k-1) + O(n)$                 | $O(n \log n)$      | $O(n \log n)$      | $O(n^2)$           | Worst case when pivot is smallest/largest (unbalanced partitions). |
| **Heap Sort**               | $T(n) = O(n) + \sum_{i=1}^n O(\log i)$          | $O(n \log n)$      | $O(n \log n)$      | $O(n \log n)$      | Build heap: $O(n)$, then $n \times$ heapify. |
| **Fibonacci (Naive)**       | $T(n) = T(n-1) + T(n-2) + O(1)$                 | $O(1)$             | $O(\phi^n)$        | $O(\phi^n)$        | $\phi = \frac{1+\sqrt{5}}{2}$. Exponential time. |
| **Tower of Hanoi**          | $T(n) = 2T(n-1) + O(1)$                         | $O(1)$             | $O(2^n)$           | $O(2^n)$           | No best-case optimization. |
| **Factorial (Recursive)**   | $T(n) = T(n-1) + O(1)$                          | $O(1)$             | $O(n)$             | $O(n)$             | Tail recursion optimizable to $O(1)$ space. |
| **TSP (Brute Force)**       | $T(n) = (n-1) \cdot T(n-1) + O(n)$              | $O(n!)$            | $O(n!)$            | $O(n!)$            | Visits all permutations. |
| **Greedy (e.g., Dijkstra)** | $T(n) = T(n-1) + O(\text{heap op})$             | $O((V+E)\log V)$   | $O((V+E)\log V)$   | $O((V+E)\log V)$   | Depends on priority queue implementation. |
| **Brute Force (Subset Sum)**| $T(n) = 2T(n-1) + O(1)$                         | $O(1)$             | $O(2^n)$           | $O(2^n)$           | Decision problem variant. |
| **Exhaustive Search**       | Problem-specific (e.g., $O(2^n)$)               | $O(1)$             | $O(2^n)$           | $O(2^n)$           | General case for NP problems. |
| **Decrease and Conquer**    | Examples: $T(n) = T(n/2) + O(1)$ (Binary Search) | $O(1)$            | $O(\log n)$        | $O(\log n)$        | Reduces problem size by a factor. |
| **Divide and Conquer**      | Examples: $T(n) = 2T(n/2) + O(n)$ (Merge Sort)  | $O(n \log n)$      | $O(n \log n)$      | $O(n \log n)$      | Splits into equal subproblems. |

---
<!--
### **Key Corrections/Clarifications:**
1. **Heap Sort**:  
   - Recurrence is better expressed as:  
     - Build heap: $O(n)$.  
     - Heapify $n$ times: $O(\log n)$ per operation.  
     - Total: $O(n \log n)$.  

2. **Quick Sort**:  
   - Average case assumes balanced partitions ($k \approx n/2$).  
   - Worst case occurs when partitions are highly unbalanced (e.g., already sorted input with pivot at end).  

3. **Fibonacci**:  
   - Naive recursive solution is exponential ($O(\phi^n)$).  
   - With memoization: $O(n)$.  

4. **Greedy Algorithms**:  
   - Time complexity varies (e.g., Dijkstra’s is $O((V+E)\log V)$ with a binary heap).  

5. **Brute Force/Exhaustive Search**:  
   - Generalizes to $O(2^n)$ or $O(n!)$ depending on the problem.  

---
-->

### **Common Misconceptions:**
- **Best Case for Selection Sort**:  
  - Always $O(n^2)$ because it scans the entire unsorted portion each time.  
- **Best Case for Bubble Sort**:  
  - $O(n)$ only if optimized to terminate early when no swaps occur (not reflected in the recurrence).  
- **Quick Sort’s Recurrence**:  
  - Average case assumes $k \approx n/2$, but the recurrence is written for a general pivot position $k$.  

---

### **Final Notes:**
- **Master Theorem**: Useful for solving divide-and-conquer recurrences like Merge Sort ($a=2, b=2, f(n)=O(n)$).  
- **Dynamic Programming**: Recurrences like Fibonacci can be optimized from $O(2^n)$ to $O(n)$ with memoization.  
- **Space Complexity**: Not included here but important (e.g., Merge Sort uses $O(n)$ auxiliary space).  

