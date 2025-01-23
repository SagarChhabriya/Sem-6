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

