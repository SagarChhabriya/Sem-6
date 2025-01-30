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
