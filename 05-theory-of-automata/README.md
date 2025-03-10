# Theory of Automata

## Week 01: Jan 22, 2025

- Why dubai is famous: Diversification;

- Automatic: Follows certain rules ()
- Autonomous: Take deciesion by itself (AI)

- Automata is the plural of automaton, and it means "Something that works automatically"

### Why do we study this subject?
It allows us to think systematically about machine. Desgining of theroetical models for machines.

- Vending Machine: Input process output based on `predefined rules`.

- Informal Languages: same things having different meanings in different contexts.
- Formal Language: 2 Parameters: Set of alphabets + Rules based.


## Introduction to Languages
1. Formal Languages (Syntactic): 
    - Follows fixed set of rules (ambiguity)
    - Is normally defined by an alphabet and formation rules.
    - 
2. Informal Languages (Semantic): 
    - Not strict rules

#### Alphabets, letters, digits, operators (113 letters in algol)
#### String

- Empty String: Lambda
- Null String
- Words: words are string belonging to language; words vs. string

- Write down some words that end with a: aba, aa, aaba, ...
    - Every word is a string but not every string is a word

- Valid / Invalid Alphabets
    - Suppose we can have the language
    - While defining an alphabet, an alphabet may contain letters consiting

$\sum = \{a,b\}$ 
Write two valid and invalid strings: ab, ba; CB, BC
    - Remarks: one letter should not be the prefix of another.

- Length of Strings
s = ababa; len = |s| = 5;
- Strings can be tokenized like:
    - tokenized String = (a), (aB), (aba), (d); len is 4

- How to create languages?
SPoken languagews like english have all words with some emaning so they are defined.

but languages we are talking about are computer languages used to communicate computer

we have created ourselves not naturally evaluated

so we deine rules ourselves

we can deine lang by applying some conditions


## Defining languages
The languages can be defined in different ways, such as
1. Descriptive definition,
2. Recursive definition, using
    - Reg exp
    - Using fininte automata 

Descriptive Ex: Write a string that ends with a;



## Recap Lecture 1
- Intro to course title
- Formal and In-formal languages
- Alphabets
- Strings, Null string,
- Words, valid and in-valid alphabets
- length of a string
- reverse of a string
- defining languages
- descriptive definition of languages
- EQUAL, EVEN-EVEN, INTEGER, EVEN, $\{a^n b^n\}, \{a^n b^n c^n\}$
- FACTORIAL, DOUBLEFACTORIAL, SQUARE, DOUBLESQUARE, PRIME, PALINDROME

Language: Start with b and end with b<br>
Word: That belong to a particular language<br>
baab | Word <br>
bab | Word<br>
abb | String<br>
baa | String<br>

- Three methods of defining languages: Descriptive, Reucrsive, Regex


## Jan 30
- Double Factorial: Even, and Odd Double Factorial
- Even Double Factorial: 8!! = 8 * 6 * 4 * 2
- Odd Double Factorial: 9!! =  9 * 7 * 5 * 3 * 1 


## Task:


### Plus
$\sum^+$


1. $(S^+)^* = (S^*)^*$
2. $(S^+)^+ = (S)^+$
3. $(S^*)^+ = (S^+)^*$

## Recursive Definitions: A Language that Follows Rules
The following three steps are used in recursive definition
1. Some basic words are specified in the language.
2. Rules for constructing more words are defined in the language
3. No String except those


### Example:
Defining language of INTEGER<br>
Step1: 1 is in INTEGER<br>
Step2: IF x is in INTEGER then x + 1 and x - 1 are also in INTEGER<br>
Step3: No String those constructed in above, are allowed to be in INTEGER.<br>

### Example



## Jan
- Kleen Star Closure
- Kleen Plus
- Concatenation: (ab)*
- Union: OR (a+b)*



(a + b)2

a2b2



aa, bb, aabb, bbaa

(a+b)2+(a+b)2 

(a*+b*)*

- ((a+b)* (a+b)*)* ✅ Even
- ((a+b)* (a+b)*)* + (a+b) ✅ Odd

-((a+b)* + (a+b)*)*


-  Syntactic (Grammar/Structure) – Making sure the sentence follows proper rules.
-  Semantic (Meaning) – Making sure the sentence makes sense.






### **How to Distinguish Between Valid and Invalid Alphabets?**  

In **automata theory**, an **alphabet (Σ)** is a **finite set of atomic (indivisible) symbols** used to form strings.  

---

### ✅ **Valid Alphabet**  
A **valid alphabet** must:  
1. **Be a finite set** → It cannot be infinite.  
2. **Contain atomic symbols** → Each element must be treated as a single, indivisible unit.  

#### **Example 1 (Valid Alphabet)**  
**Σ₁ = {B, aB, bab, d}**  
- Here, "B", "aB", "bab", and "d" are **all considered single symbols**.  
- Strings can be formed using only these symbols, e.g., **"BaBbab"** → ✅ (Valid string: "BaB", "bab")  

#### **Example 2 (Valid Alphabet)**  
**Σ₂ = {0, 1, 10}**  
- "0", "1", and "10" are atomic symbols.  
- Valid string: **"011010"** → ✅ ("0", "1", "10", "10")  

---

### ❌ **Invalid Alphabet**  
An **invalid alphabet** has:  
1. **Infinite symbols** → Not allowed.  
2. **Symbols that can be broken into smaller parts without defining atomicity**.  

#### **Example 1 (Invalid Alphabet)**  
**Σ = {a, b, ab, abc, abcd, abcde, ...}**  
- ❌ **Infinite symbols** (keeps growing) → **INVALID**.  

#### **Example 2 (Invalid Alphabet)**  
**Σ = {a, b, c, ab}**  
- If "ab" is an atomic symbol but **"a" and "b" are also separate symbols**, this can cause confusion.  
- If we see "ab", should we treat it as **"ab"** (single symbol) or **"a" and "b"** separately?  
- **Automata theory requires clear rules**, so this is **invalid unless explicitly defined**.  

---

### **Summary Table**  

| Alphabet (Σ) | Valid or Invalid? | Reason |
|-------------|----------------|--------|
| {B, aB, bab, d} | ✅ Valid | Finite, atomic symbols |
| {0, 1, 10} | ✅ Valid | Finite, atomic symbols |
| {a, b, ab, abc, abcd, ...} | ❌ Invalid | Infinite set |
| {a, b, c, ab} | ❌ Invalid | Unclear atomicity of "ab" |

---
### **What Does "Atomic Symbols" and "Indivisible Units" Mean?**  

In **automata theory**, an **atomic symbol** is a **basic unit** that **cannot be broken down further** within the given alphabet (Σ).  

### **Key Points:**  
- **Atomic symbol = Indivisible unit** → It must be treated as a **single entity**.  
- **You cannot split it into smaller parts** when forming strings.  
- **Defined in the alphabet (Σ)** → If a symbol is in Σ, it must be used **as it is**.  

---

### **Examples**  

#### ✅ **Valid Atomic Symbols**  
If Σ = {A, BB, C}  
- "A" → **Atomic** ✅  
- "BB" → **Atomic** ✅ (Even though it has two 'B's, it is defined as one unit.)  
- "C" → **Atomic** ✅  

**Valid String:** "ABBBC" ✅ (Tokens: A, BB, C)  

---

#### ❌ **Invalid Case (Breaking Atomicity)**  
If Σ = {A, BB, C}, and we try:  
- "B" alone ❌ (Not in Σ)  
- "B B C" ❌ (BB must be treated as a **single** unit)  
- "ABBC" ✅ (Correct, because BB is one atomic unit)  

---

### **Real-Life Analogy (For Kids 😊)**  
Think of **atomic symbols** like **LEGO bricks**:  
- A **single LEGO piece** (even if big) must be used **as a whole**.  
- You **cannot** break a LEGO block into smaller pieces unless it was already separate.  

For example:  
- If "BB" is a single LEGO brick, you **must** use "BB" as one piece.  
- If "B" was a separate piece, then only "B" would be allowed alone.  

---

### **Quick Test: Valid or Not?**  

1️⃣ Σ = {X, YZ, ZZ}, is "XYZZ" valid?  
✅ **Yes!** (X, YZ, ZZ are atomic units.)  

2️⃣ Σ = {M, NO, P}, is "NOP" valid?  
❌ **No!** ("NO" is atomic, so "NOP" would mean "NO P", not "N O P".)  



