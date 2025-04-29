##  Problem 1:
**`a | a | a | b | b | b | ε`**, which is:  
> $w = aaabbb$

Language: $L = \{ a^n b^n \mid n \geq 0 \}$

Accepts by **empty stack**.  
### Flow:
- **Push `A`** for each `a`  
- **Pop `A`** for each `b`  
- Accept when stack is empty (after reading all input)

![image](https://github.com/user-attachments/assets/8d9fb343-2a39-4c75-b3f4-54c3dabb8257)

##  Stack Simulation for `aaabbb`:

Initial Stack: `Z0`  
Initial State: `q_0`

| Step | Input Symbol | Top of Stack | Action Taken | Stack After |
|------|---------------|---------------|----------------|--------------|
| 1    | a             | Z0            | Push A         | A Z0         |
| 2    | a             | A             | Push A         | A A Z0       |
| 3    | a             | A             | Push A         | A A A Z0     |
| 4    | b             | A             | Pop A, move to q₁ | A A Z0   |
| 5    | b             | A             | Pop A          | A Z0         |
| 6    | b             | A             | Pop A          | Z0           |
| 7    | ε             | Z0            | Accept (Z0 left) | Z0         |

**Input `aaabbb` is accepted.**


### PDA Transitions :
1. $\delta(q_0, a, Z_0) = (q_0, AZ_0)$  
2. $\delta(q_0, a, A) = (q_0, AA)$  
3. $\delta(q_0, b, A) = (q_1, \varepsilon)$  
4. $\delta(q_1, b, A) = (q_1, \varepsilon)$  
5. $\delta(q_1, \varepsilon, Z_0) = (q_{accept}, Z_0)$

---


##  **Problem 2**
Language:  
$L = \{ a^n b^m c^{n+m} \mid n, m \geq 0 \}$

input: **`a a b c c c`**, i.e. $a^2 b^1 c^3$

---

### Flow:
- Push `A` for each `a`
- Push `B` for each `b`
- For each `c`, pop `A` or `B`
- Accept if the stack becomes empty

![image](https://github.com/user-attachments/assets/2af56d64-1d2f-4e96-8291-2977e32fcdaa)

### Stack Simulation for `a a b c c c`:

Initial Stack: `Z0`  
Start State: `q0`

| Step | Input Symbol | Stack Top | Action | Stack After |
|------|--------------|-----------|--------|-------------|
| 1    | a            | Z0        | Push A | A Z0        |
| 2    | a            | A         | Push A | A A Z0      |
| 3    | b            | A         | Push B | B A A Z0    |
| 4    | c            | B         | Pop B  | A A Z0      |
| 5    | c            | A         | Pop A  | A Z0        |
| 6    | c            | A         | Pop A  | Z0          |
| 7    | ε            | Z0        | Accept | Z0          |

Accepted! Stack is empty after all input is processed.

### PDA Transitions

1. $\delta(q_0, a, Z_0) = (q_0, AZ_0)$  
2. $\delta(q_0, a, A) = (q_0, AA)$  
3. $\delta(q_0, b, A) = (q_0, BA)$  
4. $\delta(q_0, c, B) = (q_1, \varepsilon)$  
5. $\delta(q_1, c, A) = (q_1, \varepsilon)$  
6. $\delta(q_1, c, A) = (q_1, \varepsilon)$  
7. $\delta(q_1, \varepsilon, Z_0) = (q_{accept}, Z_0)$


---

##  **Problem 3**

**Language:**  $L = \{ a^n b^m \mid n \gt m \}$

### Flow:
- Push 1 `A` for each `a`
- Pop 1 `A` for each `b`
- If **stack still has `A`s** after input is done ⇒ Accept  
- Do **not** accept if the stack is empty (that would be $n = m$)

![image](https://github.com/user-attachments/assets/86e08620-c7bb-4e55-8bb6-c132be2d5931)

### Example: `a a a b` (n=3, m=1 ⇒ n > m)

| Step | Input | Stack Top | Action     | Stack After |
|------|--------|------------|------------|--------------|
| 1    | a      | Z0         | Push A     | A Z0         |
| 2    | a      | A          | Push A     | A A Z0       |
| 3    | a      | A          | Push A     | A A A Z0     |
| 4    | b      | A          | Pop A      | A A Z0       |
| 5    | ε      | A          | Accept     | A A Z0       |

Accepted, since extra `A`s left on stack ⇒ $n > m$

### Transitions:
1. $\delta(q_0, a, Z_0) = (q_0, AZ_0)$  
2. $\delta(q_0, a, A) = (q_0, AA)$  
3. $\delta(q_0, b, A) = (q_0, \varepsilon)$  
4. $\delta(q_0, \varepsilon, A) = (q_{\text{accept}}, A)$   // Accept if A remains








